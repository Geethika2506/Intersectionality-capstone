import hashlib
import json
from pathlib import Path

import pytest

from evaluate_intersectionality import main
from pilot.config import EVALUATION_DIR, PASSAGE_IDS
from pilot.human_review import (
    ReviewCheckError,
    create_templates,
    parse_reference_file,
    parse_reference_text,
    reference_path,
    write_summary,
)
from pilot.storage import extract_output_text
from pilot.word_count import count_words

ROWS_01 = [
    ("social_position", "SP1", "Migrant women"),
    ("intersectional_relationship", "IR1", "Gender and residence status"),
    ("institutional_mechanism", "IM1", "Shelter address rule"),
]
ROWS_02 = [
    ("social_position", "SP2", "Rural women"),
    ("intersectional_relationship", "IR2", "Gender and disability"),
    ("institutional_mechanism", "IM2", "Police risk checklist"),
]


def _write_reference(directory: Path, passage_id: str, rows: list[tuple[str, str, str]]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    lines = ["category\titem_id\treference_text"]
    lines.extend("\t".join(row) for row in rows)
    (directory / f"{passage_id}_coding_REFERENCE.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _response_payload(text: str) -> dict:
    return {
        "id": "resp_test",
        "model": "gpt-6.1-sol",
        "output": [
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "hidden " * 30}],
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            },
        ],
    }


def _write_response(run_dir: Path, passage_id: str, repeat: int, text: str) -> Path:
    path = run_dir / f"{passage_id}_repeat_{repeat:02d}_response.json"
    path.write_text(json.dumps(_response_payload(text)), encoding="utf-8")
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_run(tmp_path: Path) -> tuple[Path, Path, dict[tuple[str, int], str]]:
    reference_dir = tmp_path / "evaluation"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_reference(reference_dir, "passage_01", ROWS_01)
    _write_reference(reference_dir, "passage_02", ROWS_02)
    texts: dict[tuple[str, int], str] = {}
    passage_01 = {
        1: "one two three four",
        2: "one two three four five",
        3: "one two three four five six",
    }
    for repeat, text in passage_01.items():
        _write_response(run_dir, "passage_01", repeat, text)
        texts[("passage_01", repeat)] = text
    for repeat in (1, 2, 3):
        text = "rural brief text"
        _write_response(run_dir, "passage_02", repeat, text)
        texts[("passage_02", repeat)] = text
    return reference_dir, run_dir, texts


def _fill(path: Path, labels: dict[str, int], support: dict[str, bool | None], excerpt: str) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    for item in data["items"]:
        score = labels[item["item_id"]]
        item["retention"] = score
        if score in (1, 2):
            item["excerpt"] = excerpt
            item["supported_by_passage"] = support[item["item_id"]]
        else:
            item["excerpt"] = ""
            item["supported_by_passage"] = None
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def test_parses_tsv_markdown_and_section_tables():
    tsv = (
        "item_id\tcategory\treference_text\n"
        "SP1\tsocial position\tMigrant women\n"
        "IR1\tintersectional_relationship\tGender and residence status\n"
        "IM1\tinstitutional_mechanism\tShelter address rule\n"
    )
    markdown = """
| item_id | category | reference_text |
| --- | --- | --- |
| SP1 | social_position | Migrant women |
| IR1 | intersectional relationship | Gender and residence status |
| IM1 | institutional_mechanism | Shelter address rule |
"""
    sections = """
## social_position
SP1	Migrant women

## Intersectional relationship
IR1: Gender and residence status

## institutional mechanism
- IM1 - Shelter address rule
"""
    expected = [
        ("SP1", "social_position", "Migrant women"),
        ("IR1", "intersectional_relationship", "Gender and residence status"),
        ("IM1", "institutional_mechanism", "Shelter address rule"),
    ]
    for text in (tsv, markdown, sections):
        items = parse_reference_text(text, "passage_01", source_name="coding.txt")
        assert [(item.item_id, item.category, item.reference_text) for item in items] == expected


def test_rejects_duplicate_unknown_category_and_passage_mismatch():
    duplicate = "category\titem_id\treference_text\nsocial_position\tSP1\tA\nsocial_position\tSP1\tB\n"
    with pytest.raises(ReviewCheckError) as duplicate_error:
        parse_reference_text(duplicate, "passage_01")
    assert "duplicate item_id SP1" in str(duplicate_error.value)

    unknown = "category\titem_id\treference_text\neconomy\tEC1\tPrices\n"
    with pytest.raises(ReviewCheckError) as unknown_error:
        parse_reference_text(unknown, "passage_01")
    assert "unknown category" in str(unknown_error.value)

    mismatched = (
        "category\titem_id\treference_text\tpassage_id\n"
        "social_position\tSP1\tA\tpassage_02\n"
    )
    with pytest.raises(ReviewCheckError) as mismatch_error:
        parse_reference_text(mismatched, "passage_01", source_name="passage_01_coding_REFERENCE.txt")
    assert "passage_id is passage_02" in str(mismatch_error.value)


def test_extracts_brief_text_from_response_json_not_reasoning():
    payload = _response_payload("Visible brief only")
    assert extract_output_text(payload) == "Visible brief only"
    assert "hidden" not in extract_output_text(payload)
    convenience = dict(payload)
    convenience["output_text"] = "from convenience field"
    assert extract_output_text(convenience) == "from convenience field"


def test_templates_read_response_json_and_leave_outputs_unchanged(tmp_path: Path):
    reference_dir, run_dir, texts = _make_run(tmp_path)
    before = {path.name: _sha(path) for path in run_dir.glob("*.json")}
    messages = create_templates(run_dir, reference_dir)
    assert len(messages) == 6
    assert all(message.startswith("Created ") for message in messages)
    after = {path.name: _sha(path) for path in run_dir.glob("*.json")}
    assert after == before

    review = json.loads(
        (run_dir / "reviews" / "passage_01_repeat_01_review.json").read_text(encoding="utf-8")
    )
    assert review["output_file"] == "passage_01_repeat_01_response.json"
    assert review["brief_text"] == texts[("passage_01", 1)]
    assert "hidden" not in review["brief_text"]
    assert review["brief_word_count"] == count_words(texts[("passage_01", 1)])
    assert [item["retention"] for item in review["items"]] == [None, None, None]
    assert review["additional_unsupported_claims"] == []


def test_templates_do_not_overwrite_completed_or_partial_reviews(tmp_path: Path):
    reference_dir, run_dir, _texts = _make_run(tmp_path)
    create_templates(run_dir, reference_dir)
    target = run_dir / "reviews" / "passage_01_repeat_01_review.json"
    partial = json.loads(target.read_text(encoding="utf-8"))
    partial["items"][0]["retention"] = 1
    partial["items"][0]["excerpt"] = "one two"
    target.write_text(json.dumps(partial, indent=2) + "\n", encoding="utf-8")

    again = create_templates(run_dir, reference_dir)
    assert any("Left existing review unchanged: passage_01_repeat_01_review.json" == line for line in again)
    kept = json.loads(target.read_text(encoding="utf-8"))
    assert kept["items"][0]["retention"] == 1
    assert kept["items"][0]["excerpt"] == "one two"

    for item in kept["items"]:
        item["retention"] = 2
    target.write_text(json.dumps(kept, indent=2) + "\n", encoding="utf-8")
    completed_bytes = target.read_bytes()
    reference = (reference_dir / "passage_01_coding_REFERENCE.txt").read_text(encoding="utf-8")
    (reference_dir / "passage_01_coding_REFERENCE.txt").write_text(
        reference.replace("Migrant women", "Changed after review"),
        encoding="utf-8",
    )
    final = create_templates(run_dir, reference_dir)
    assert "Left completed review unchanged: passage_01_repeat_01_review.json" in final
    assert target.read_bytes() == completed_bytes
    assert b"Changed after review" not in target.read_bytes()


def test_summary_reports_retention_variation_and_word_counts(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    reference_dir, run_dir, texts = _make_run(tmp_path)
    before = {path.name: _sha(path) for path in run_dir.glob("*_response.json")}
    assert main(["templates", "--run-dir", str(run_dir), "--reference-dir", str(reference_dir)]) == 0
    capsys.readouterr()

    plans = {
        ("passage_01", 1): ({"SP1": 2, "IR1": 2, "IM1": 2}, {"SP1": True, "IR1": True, "IM1": True}),
        ("passage_01", 2): ({"SP1": 1, "IR1": 2, "IM1": 2}, {"SP1": False, "IR1": True, "IM1": True}),
        ("passage_01", 3): ({"SP1": 0, "IR1": 2, "IM1": 2}, {"SP1": None, "IR1": True, "IM1": True}),
        ("passage_02", 1): ({"SP2": 2, "IR2": 2, "IM2": 2}, {"SP2": True, "IR2": True, "IM2": True}),
        ("passage_02", 2): ({"SP2": 2, "IR2": 2, "IM2": 2}, {"SP2": True, "IR2": True, "IM2": True}),
        ("passage_02", 3): ({"SP2": 2, "IR2": 2, "IM2": 2}, {"SP2": True, "IR2": True, "IM2": True}),
    }
    for (passage_id, repeat), (labels, support) in plans.items():
        path = run_dir / "reviews" / f"{passage_id}_repeat_{repeat:02d}_review.json"
        excerpt = "one two" if passage_id == "passage_01" else "rural brief"
        _fill(path, labels, support, excerpt)
        if (passage_id, repeat) == ("passage_01", 1):
            data = json.loads(path.read_text(encoding="utf-8"))
            data["brief_word_count"] = 999
            data["additional_unsupported_claims"] = [
                {
                    "claim": "Invented national shelter closure",
                    "excerpt": "shelters closed nationwide",
                }
            ]
            path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    assert main(["summary", "--run-dir", str(run_dir), "--reference-dir", str(reference_dir)]) == 0
    printed = capsys.readouterr().out
    summary = (run_dir / "reviews" / "summary.md").read_text(encoding="utf-8")
    assert summary == printed.split("Wrote ", 1)[0]
    assert "not statistically conclusive" in summary
    assert "statistically significant" not in summary.lower()
    assert "999" not in summary
    assert (
        "Social position retention across the three independent runs was "
        "1.00, 0.50, and 0.00 (lowest 0.00, highest 1.00, spread 1.00)."
    ) in summary
    assert "Social position retention was 1.00 on each of the three independent runs." in summary
    assert "| Social position | 1.00 (2/2) | 0.50 (1/2) | 0.00 (0/2) |" in summary
    assert "Run 2: 2 judged supported by the GREVIO passage, 1 judged not supported." in summary
    assert "Invented national shelter closure" in summary
    assert "are not part of retention" in summary
    for (passage_id, repeat), text in texts.items():
        name = f"{passage_id}_repeat_{repeat:02d}_response.json"
        assert f"| {name} | {count_words(text)} |" in summary
    stored = json.loads(
        (run_dir / "reviews" / "passage_01_repeat_01_review.json").read_text(encoding="utf-8")
    )
    assert stored["brief_word_count"] == 999
    after = {path.name: _sha(path) for path in run_dir.glob("*_response.json")}
    assert after == before


def test_summary_rejects_missing_labels(tmp_path: Path):
    reference_dir, run_dir, _texts = _make_run(tmp_path)
    create_templates(run_dir, reference_dir)
    with pytest.raises(ReviewCheckError) as missing:
        write_summary(run_dir, reference_dir)
    assert "missing retention label" in str(missing.value)
    assert "SP1" in str(missing.value)
    assert not (run_dir / "reviews" / "summary.md").exists()

    path = run_dir / "reviews" / "passage_01_repeat_01_review.json"
    _fill(
        path,
        {"SP1": 1, "IR1": 2, "IM1": 2},
        {"SP1": True, "IR1": True, "IM1": True},
        "one two",
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    data["items"][0]["excerpt"] = "  "
    path.write_text(json.dumps(data) + "\n", encoding="utf-8")
    with pytest.raises(ReviewCheckError) as missing_excerpt:
        write_summary(run_dir, reference_dir)
    assert "missing excerpt" in str(missing_excerpt.value)

    data["items"][0]["excerpt"] = "one two"
    data["items"][0]["supported_by_passage"] = None
    path.write_text(json.dumps(data) + "\n", encoding="utf-8")
    with pytest.raises(ReviewCheckError) as missing_support:
        write_summary(run_dir, reference_dir)
    assert "missing supported_by_passage" in str(missing_support.value)
    assert not (run_dir / "reviews" / "summary.md").exists()


def test_summary_rejects_duplicate_items_and_passage_mismatch(tmp_path: Path):
    reference_dir, run_dir, _texts = _make_run(tmp_path)
    create_templates(run_dir, reference_dir)
    for passage_id, repeat in ((pid, rep) for pid in PASSAGE_IDS for rep in (1, 2, 3)):
        path = run_dir / "reviews" / f"{passage_id}_repeat_{repeat:02d}_review.json"
        labels = {item[1]: 2 for item in (ROWS_01 if passage_id == "passage_01" else ROWS_02)}
        support = {item_id: True for item_id in labels}
        excerpt = "one two" if passage_id == "passage_01" else "rural brief"
        _fill(path, labels, support, excerpt)

    duplicate_path = run_dir / "reviews" / "passage_01_repeat_02_review.json"
    duplicate = json.loads(duplicate_path.read_text(encoding="utf-8"))
    duplicate["items"].append(dict(duplicate["items"][0]))
    duplicate_path.write_text(json.dumps(duplicate) + "\n", encoding="utf-8")
    with pytest.raises(ReviewCheckError) as duplicate_error:
        write_summary(run_dir, reference_dir)
    assert "duplicate item_id SP1" in str(duplicate_error.value)

    duplicate_path.write_text(json.dumps({**duplicate, "items": duplicate["items"][:-1]}) + "\n", encoding="utf-8")
    mismatch_path = run_dir / "reviews" / "passage_01_repeat_01_review.json"
    mismatch = json.loads(mismatch_path.read_text(encoding="utf-8"))
    mismatch["passage_id"] = "passage_02"
    mismatch_path.write_text(json.dumps(mismatch) + "\n", encoding="utf-8")
    with pytest.raises(ReviewCheckError) as mismatch_error:
        write_summary(run_dir, reference_dir)
    assert "passage_id is passage_02" in str(mismatch_error.value)
    assert "passage_01_repeat_01_response.json" in str(mismatch_error.value)

    mismatch["passage_id"] = "passage_01"
    mismatch_path.write_text(json.dumps(mismatch) + "\n", encoding="utf-8")
    request = {
        "passage_id": "passage_02",
        "repeat": 1,
        "input_text": "not used",
    }
    (run_dir / "passage_01_repeat_01_request.json").write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(ReviewCheckError) as request_error:
        write_summary(run_dir, reference_dir)
    assert "passage_01_repeat_01_request.json: passage_id is passage_02" in str(request_error.value)
    assert not (run_dir / "reviews" / "summary.md").exists()


def test_rejects_unknown_and_missing_output_files(tmp_path: Path):
    reference_dir, run_dir, _texts = _make_run(tmp_path)
    decoy = run_dir / "passage_01_repeat_01_response.txt"
    decoy.write_text("this text file is not the saved brief\n", encoding="utf-8")
    with pytest.raises(ReviewCheckError) as unknown:
        create_templates(run_dir, reference_dir)
    assert "unknown output file passage_01_repeat_01_response.txt" in str(unknown.value)
    assert not (run_dir / "reviews").exists()

    decoy.unlink()
    (run_dir / "passage_02_repeat_03_response.json").unlink()
    with pytest.raises(ReviewCheckError) as missing:
        create_templates(run_dir, reference_dir)
    assert "missing output file passage_02_repeat_03_response.json" in str(missing.value)
    assert not (run_dir / "reviews").exists()

    _write_response(run_dir, "passage_02", 3, "rural brief text")
    create_templates(run_dir, reference_dir)
    review_path = run_dir / "reviews" / "passage_02_repeat_03_review.json"
    review = json.loads(review_path.read_text(encoding="utf-8"))
    review["output_file"] = "passage_02_repeat_03_response.txt"
    for item in review["items"]:
        item["retention"] = 2
        item["excerpt"] = "rural brief"
        item["supported_by_passage"] = True
    review_path.write_text(json.dumps(review) + "\n", encoding="utf-8")
    for passage_id, repeat in ((pid, rep) for pid in PASSAGE_IDS for rep in (1, 2, 3)):
        if (passage_id, repeat) == ("passage_02", 3):
            continue
        path = run_dir / "reviews" / f"{passage_id}_repeat_{repeat:02d}_review.json"
        labels = {item[1]: 2 for item in (ROWS_01 if passage_id == "passage_01" else ROWS_02)}
        _fill(path, labels, {item_id: True for item_id in labels}, "one two" if passage_id == "passage_01" else "rural brief")
    with pytest.raises(ReviewCheckError) as named_unknown:
        write_summary(run_dir, reference_dir)
    assert "unknown output file passage_02_repeat_03_response.txt" in str(named_unknown.value)


def test_shipped_reference_files_have_no_coded_items():
    for passage_id in PASSAGE_IDS:
        items = parse_reference_file(reference_path(EVALUATION_DIR, passage_id), passage_id)
        assert items == []
