"""Human review of the six saved policy briefs.

Reference coding stays in evaluation/. Review forms are written beside a run
and never replace a form that already exists. Response JSON files are read
only. This module does not call the OpenAI API.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from pilot.config import (
    EVALUATION_DIR,
    PASSAGE_IDS,
    REPEATS_PER_PASSAGE,
)
from pilot.storage import (
    extract_output_text,
    request_filename,
    response_filename,
)
from pilot.word_count import count_words

CATEGORIES = (
    "social_position",
    "intersectional_relationship",
    "institutional_mechanism",
)

CATEGORY_LABELS = {
    "social_position": "Social position",
    "intersectional_relationship": "Intersectional relationship",
    "institutional_mechanism": "Institutional mechanism",
}

RETENTION_LABELS = {
    0: "omitted or distorted",
    1: "partly preserved",
    2: "preserved",
}

INSTRUCTIONS = (
    "Compare this brief with each reference item. Set retention to 0 "
    "(omitted or distorted), 1 (partly preserved), or 2 (preserved). "
    "For a score of 1 or 2, paste an excerpt from this brief and set "
    "supported_by_passage to true or false: true means the GREVIO passage "
    "supports the claim. To record a claim the brief introduces that is not "
    "in the reference coding, add an object with claim and excerpt to "
    "additional_unsupported_claims. Do not edit passage_id, repeat, "
    "output_file, item_id, category, or reference_text."
)

_CATEGORY_ALIASES = {
    "social position": "social_position",
    "intersectional relationship": "intersectional_relationship",
    "institutional mechanism": "institutional_mechanism",
}

_HEADER_ALIASES = {
    "category": "category",
    "dimension": "category",
    "item id": "item_id",
    "id": "item_id",
    "code": "item_id",
    "reference text": "reference_text",
    "text": "reference_text",
    "item": "reference_text",
    "description": "reference_text",
    "coding": "reference_text",
    "reference": "reference_text",
    "passage id": "passage_id",
    "passage": "passage_id",
}

_PASSAGE_ALIASES = {
    "passage_01": "passage_01",
    "passage_1": "passage_01",
    "passage01": "passage_01",
    "passage_02": "passage_02",
    "passage_2": "passage_02",
    "passage02": "passage_02",
}


class ReviewCheckError(Exception):
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("\n".join(errors))


@dataclass(frozen=True)
class ReferenceItem:
    passage_id: str
    category: str
    item_id: str
    reference_text: str
    source_line: int


def normalize_category(raw: str) -> str:
    key = " ".join(raw.strip().lower().replace("_", " ").replace("-", " ").split())
    if key in _CATEGORY_ALIASES:
        return _CATEGORY_ALIASES[key]
    if key.replace(" ", "_") in CATEGORIES:
        return key.replace(" ", "_")
    raise ValueError(
        "unknown category "
        f"{raw!r} (use social_position, intersectional_relationship, "
        "or institutional_mechanism)"
    )


def normalize_passage_id(raw: str) -> str:
    key = re.sub(r"_+", "_", raw.strip().lower().replace("-", " ").replace(" ", "_"))
    if key not in _PASSAGE_ALIASES:
        raise ValueError(f"unknown passage_id {raw!r}")
    return _PASSAGE_ALIASES[key]


def reference_path(reference_dir: Path, passage_id: str) -> Path:
    return reference_dir / f"{passage_id}_coding_REFERENCE.txt"


def review_filename(passage_id: str, repeat: int) -> str:
    return f"{passage_id}_repeat_{repeat:02d}_review.json"


def reviews_dir(run_dir: Path) -> Path:
    return run_dir / "reviews"


def summary_path(run_dir: Path) -> Path:
    return reviews_dir(run_dir) / "summary.md"


def _header_alias(cell: str) -> str | None:
    key = " ".join(cell.strip().lower().replace("_", " ").replace("-", " ").split())
    return _HEADER_ALIASES.get(key)


def _split_cells(raw: str) -> list[str]:
    line = raw.strip()
    if line.startswith("|"):
        return [cell.strip() for cell in line.strip().strip("|").split("|")]
    if "\t" in raw:
        return [cell.strip() for cell in raw.strip().split("\t")]
    return [line]


def _is_separator(cells: list[str]) -> bool:
    useful = [cell for cell in cells if cell]
    return bool(useful) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in useful)


def _is_header(cells: list[str]) -> bool:
    mapped = [_header_alias(cell) for cell in cells if cell]
    if not mapped or any(field is None for field in mapped):
        return False
    return "item_id" in mapped and "reference_text" in mapped


def _strip_bullet(line: str) -> str:
    return re.sub(r"^(?:[-*]|\d+[.)])\s+", "", line.strip())


def _split_id_text(text: str) -> tuple[str, str] | None:
    if "\t" in text:
        item_id, reference_text = text.split("\t", 1)
        if item_id.strip() and reference_text.strip():
            return item_id.strip(), reference_text.strip()
        return None
    match = re.match(
        r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*[:\-–—]\s+(\S.*)$",
        text.strip(),
    )
    if match:
        return match.group(1), match.group(2).strip()
    return None


def _parse_reference_lines(
    text: str, passage_id: str, source_name: str
) -> tuple[list[ReferenceItem], list[str]]:
    items: list[ReferenceItem] = []
    errors: list[str] = []
    seen: dict[str, int] = {}
    current_category: str | None = None
    columns: list[str | None] | None = None

    def add_item(
        line_no: int,
        item_id: str,
        reference_text: str,
        category: str | None,
        row_passage: str | None,
    ) -> None:
        if row_passage:
            try:
                parsed_passage = normalize_passage_id(row_passage)
            except ValueError as exc:
                errors.append(f"{source_name} line {line_no}: {exc}")
                return
            if parsed_passage != passage_id:
                errors.append(
                    f"{source_name} line {line_no}: passage_id is {parsed_passage} "
                    f"in the {passage_id} reference file"
                )
                return
        if not item_id:
            errors.append(f"{source_name} line {line_no}: missing item_id")
            return
        if not reference_text:
            errors.append(f"{source_name} line {line_no}: missing reference_text for {item_id}")
            return
        if category is None:
            errors.append(
                f"{source_name} line {line_no}: missing category for {item_id}"
            )
            return
        try:
            parsed_category = normalize_category(category)
        except ValueError as exc:
            errors.append(f"{source_name} line {line_no}: {exc}")
            return
        if item_id in seen:
            errors.append(
                f"{source_name}: duplicate item_id {item_id} "
                f"on lines {seen[item_id]} and {line_no}"
            )
            return
        seen[item_id] = line_no
        items.append(
            ReferenceItem(
                passage_id=passage_id,
                category=parsed_category,
                item_id=item_id,
                reference_text=reference_text,
                source_line=line_no,
            )
        )

    for line_no, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.startswith("#") and not stripped.startswith("##"):
            continue
        heading = re.match(r"^#{2,6}\s+(.+?)\s*$", stripped)
        if heading:
            columns = None
            try:
                current_category = normalize_category(heading.group(1))
            except ValueError as exc:
                current_category = None
                errors.append(f"{source_name} line {line_no}: {exc}")
            continue
        if "\t" not in raw and not stripped.startswith("|") and not stripped.startswith(("-", "*")):
            try:
                current_category = normalize_category(stripped)
            except ValueError:
                pass
            else:
                columns = None
                continue

        cells = _split_cells(raw)
        if _is_separator(cells):
            continue
        if _is_header(cells):
            columns = [_header_alias(cell) if cell else None for cell in cells]
            continue
        if columns is not None:
            values: dict[str, str] = {}
            for field, cell in zip(columns, cells):
                if field:
                    values[field] = cell.strip()
            if len(cells) > len(columns) and "reference_text" in values:
                values["reference_text"] = "\t".join(
                    [values["reference_text"], *cells[len(columns) :]]
                ).strip()
            add_item(
                line_no,
                values.get("item_id", ""),
                values.get("reference_text", ""),
                values.get("category") or current_category,
                values.get("passage_id") or None,
            )
            continue

        parsed = _split_id_text(_strip_bullet(stripped))
        if parsed is None or current_category is None:
            errors.append(f"{source_name} line {line_no}: could not read a coding row")
            continue
        item_id, reference_text = parsed
        add_item(line_no, item_id, reference_text, current_category, None)

    return items, errors


def parse_reference_text(
    text: str, passage_id: str, source_name: str = "reference"
) -> list[ReferenceItem]:
    items, errors = _parse_reference_lines(text, passage_id, source_name)
    if errors:
        raise ReviewCheckError(errors)
    return items


def parse_reference_file(path: Path, passage_id: str) -> list[ReferenceItem]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing reference file: {path}")
    return parse_reference_text(
        path.read_text(encoding="utf-8-sig"),
        passage_id,
        source_name=path.name,
    )


def load_references(
    reference_dir: Path, *, require_items: bool
) -> tuple[dict[str, list[ReferenceItem]], list[str]]:
    found: dict[str, list[ReferenceItem]] = {}
    errors: list[str] = []
    for passage_id in PASSAGE_IDS:
        path = reference_path(reference_dir, passage_id)
        if not path.is_file():
            errors.append(f"missing reference file {path.name}")
            continue
        items, parse_errors = _parse_reference_lines(
            path.read_text(encoding="utf-8-sig"),
            passage_id,
            path.name,
        )
        errors.extend(parse_errors)
        if parse_errors:
            continue
        found[passage_id] = items
        if not require_items:
            continue
        if not items:
            errors.append(
                f"{path.name} has no coding items. Paste a row for social_position, "
                "intersectional_relationship, and institutional_mechanism under the header."
            )
            continue
        present = {item.category for item in items}
        missing = [CATEGORY_LABELS[category] for category in CATEGORIES if category not in present]
        if missing:
            errors.append(
                f"{path.name} is missing reference items for: {', '.join(missing)}"
            )
    return found, errors


def _output_candidate(name: str) -> bool:
    if name.endswith("_request.json") or name.endswith("_failure.json") or name == "run_log.jsonl":
        return False
    if "response" in name:
        return True
    return re.fullmatch(r"passage_\d+_repeat_\d+\..+", name) is not None


def expected_outputs() -> list[tuple[str, int, str]]:
    rows: list[tuple[str, int, str]] = []
    for passage_id in PASSAGE_IDS:
        for repeat in range(1, REPEATS_PER_PASSAGE + 1):
            rows.append((passage_id, repeat, response_filename(passage_id, repeat)))
    return rows


def locate_outputs(run_dir: Path) -> tuple[dict[tuple[str, int], Path], list[str]]:
    errors: list[str] = []
    known = {name for _, _, name in expected_outputs()}
    found: dict[tuple[str, int], Path] = {}
    if not run_dir.is_dir():
        return found, [f"run directory not found: {run_dir}"]
    for path in sorted(run_dir.iterdir()):
        if not path.is_file() or not _output_candidate(path.name):
            continue
        if path.name not in known:
            errors.append(f"unknown output file {path.name}")
    for passage_id, repeat, name in expected_outputs():
        path = run_dir / name
        if not path.is_file():
            errors.append(f"missing output file {name}")
        else:
            found[(passage_id, repeat)] = path
    return found, errors


def brief_text_from_response(path: Path) -> str:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ReviewCheckError([f"{path.name}: response file is not valid JSON ({exc})"]) from exc
    if not isinstance(payload, dict):
        raise ReviewCheckError([f"{path.name}: response JSON must be an object"])
    if "output" not in payload and "output_text" not in payload:
        raise ReviewCheckError(
            [
                f"{path.name}: no brief text found "
                "(expected output_text or output message text in the response JSON)"
            ]
        )
    return extract_output_text(payload)


def _check_request_files(run_dir: Path) -> list[str]:
    errors: list[str] = []
    for passage_id, repeat, _name in expected_outputs():
        path = run_dir / request_filename(passage_id, repeat)
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"{path.name}: request file is not valid JSON ({exc})")
            continue
        if not isinstance(payload, dict):
            errors.append(f"{path.name}: request JSON must be an object")
            continue
        if "passage_id" in payload and payload["passage_id"] != passage_id:
            errors.append(
                f"{path.name}: passage_id is {payload['passage_id']} "
                f"but the file name is {passage_id} repeat {repeat}"
            )
        if "repeat" in payload and payload["repeat"] != repeat:
            errors.append(
                f"{path.name}: repeat is {payload['repeat']} "
                f"but the file name is {passage_id} repeat {repeat}"
            )
    return errors


def _review_payload(
    passage_id: str,
    repeat: int,
    output_name: str,
    brief_text: str,
    items: list[ReferenceItem],
) -> dict:
    return {
        "instructions": INSTRUCTIONS,
        "passage_id": passage_id,
        "repeat": repeat,
        "output_file": output_name,
        "brief_word_count": count_words(brief_text),
        "brief_text": brief_text,
        "retention_scale": {str(score): label for score, label in RETENTION_LABELS.items()},
        "items": [
            {
                "item_id": item.item_id,
                "category": item.category,
                "reference_text": item.reference_text,
                "retention": None,
                "excerpt": "",
                "supported_by_passage": None,
            }
            for item in items
        ],
        "additional_unsupported_claims": [],
    }


def _retentions_present(data: dict) -> bool:
    items = data.get("items")
    if not isinstance(items, list) or not items:
        return False
    for item in items:
        if not isinstance(item, dict) or _parse_retention(item.get("retention")) not in (0, 1, 2):
            return False
    return True


def create_templates(run_dir: Path, reference_dir: Path | None = None) -> list[str]:
    """Write one review form per saved brief. Never overwrite an existing form."""
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    reference_root = reference_dir or EVALUATION_DIR
    errors: list[str] = []
    _outputs, output_errors = locate_outputs(run_dir)
    errors.extend(output_errors)
    references, reference_errors = load_references(reference_root, require_items=True)
    errors.extend(reference_errors)
    errors.extend(_check_request_files(run_dir))
    brief_texts: dict[tuple[str, int], str] = {}
    if not output_errors:
        for passage_id, repeat, name in expected_outputs():
            try:
                brief_texts[(passage_id, repeat)] = brief_text_from_response(run_dir / name)
            except ReviewCheckError as exc:
                errors.extend(exc.errors)
    if errors:
        raise ReviewCheckError(errors)

    destination = reviews_dir(run_dir)
    destination.mkdir(parents=True, exist_ok=True)
    messages: list[str] = []
    for passage_id, repeat, output_name in expected_outputs():
        path = destination / review_filename(passage_id, repeat)
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                existing = None
            if isinstance(existing, dict) and _retentions_present(existing):
                messages.append(f"Left completed review unchanged: {path.name}")
            else:
                messages.append(f"Left existing review unchanged: {path.name}")
            continue
        payload = _review_payload(
            passage_id,
            repeat,
            output_name,
            brief_texts[(passage_id, repeat)],
            references[passage_id],
        )
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        messages.append(f"Created {path.name}")
    return messages


def _parse_retention(value: object) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value in (0, 1, 2):
        return value
    if isinstance(value, str) and value.strip() in {"0", "1", "2"}:
        return int(value.strip())
    return None


def _retention_is_invalid(value: object) -> bool:
    return value not in (None, "") and _parse_retention(value) is None


def _parse_support(value: object) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        key = value.strip().lower()
        if key in {"true", "yes", "y"}:
            return True
        if key in {"false", "no", "n"}:
            return False
    return None


def _support_is_invalid(value: object) -> bool:
    return value not in (None, "") and _parse_support(value) is None


def _load_review(path: Path) -> tuple[dict | None, list[str]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return None, [f"{path.name}: review file is not valid JSON ({exc})"]
    if not isinstance(data, dict):
        return None, [f"{path.name}: review JSON must be an object"]
    return data, []


def _identity_from_review_name(name: str) -> tuple[str, int] | None:
    match = re.fullmatch(r"(passage_\d+)_repeat_(\d+)_review\.json", name)
    if not match:
        return None
    return match.group(1), int(match.group(2))


def _validate_review(
    path: Path,
    data: dict,
    references: dict[str, list[ReferenceItem]],
) -> list[str]:
    errors: list[str] = []
    identity = _identity_from_review_name(path.name)
    if identity is None:
        return [f"unknown review file {path.name}"]
    file_passage, file_repeat = identity
    expected_output = response_filename(file_passage, file_repeat)
    known_outputs = {name for _, _, name in expected_outputs()}

    passage_id = data.get("passage_id")
    if passage_id != file_passage:
        errors.append(
            f"{path.name}: passage_id is {passage_id} but {expected_output} is "
            f"{file_passage} repeat {file_repeat}"
        )
    repeat = data.get("repeat")
    if repeat != file_repeat:
        errors.append(
            f"{path.name}: repeat is {repeat} but {expected_output} is "
            f"{file_passage} repeat {file_repeat}"
        )
    output_file = data.get("output_file")
    if output_file != expected_output:
        if output_file not in known_outputs:
            errors.append(f"{path.name}: unknown output file {output_file}")
        else:
            errors.append(
                f"{path.name}: output_file is {output_file} but this review is for {expected_output}"
            )

    items = data.get("items")
    if not isinstance(items, list):
        errors.append(f"{path.name}: items must be a list")
        items = []
    seen: dict[str, int] = {}
    reference_by_id = {item.item_id: item for item in references.get(file_passage, [])}
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            errors.append(f"{path.name}: item {index} must be an object")
            continue
        item_id = str(item.get("item_id") or "")
        if not item_id:
            errors.append(f"{path.name}: item {index} is missing item_id")
            continue
        if item_id in seen:
            errors.append(
                f"{path.name}: duplicate item_id {item_id} "
                f"(items {seen[item_id]} and {index})"
            )
            continue
        seen[item_id] = index
        reference = reference_by_id.get(item_id)
        if reference is None:
            errors.append(
                f"{path.name}: item_id {item_id} is not in the {file_passage} reference coding"
            )
        else:
            if item.get("category") != reference.category:
                errors.append(
                    f"{path.name}: item {item_id} category does not match the {file_passage} reference"
                )
            if item.get("reference_text") != reference.reference_text:
                errors.append(
                    f"{path.name}: item {item_id} reference_text does not match the {file_passage} reference"
                )
        if _retention_is_invalid(item.get("retention")):
            errors.append(
                f"{path.name}: item {item_id} has an invalid retention label "
                "(use 0, 1, or 2)"
            )
            continue
        retention = _parse_retention(item.get("retention"))
        if retention is None:
            errors.append(f"{path.name}: item {item_id} is missing retention label")
            continue
        if retention in (1, 2):
            excerpt = item.get("excerpt")
            if not isinstance(excerpt, str) or not excerpt.strip():
                errors.append(
                    f"{path.name}: item {item_id} is missing excerpt for retention {retention}"
                )
            if _support_is_invalid(item.get("supported_by_passage")):
                errors.append(
                    f"{path.name}: item {item_id} has an invalid supported_by_passage value"
                )
            elif _parse_support(item.get("supported_by_passage")) is None:
                errors.append(
                    f"{path.name}: item {item_id} is missing supported_by_passage"
                )
        elif _support_is_invalid(item.get("supported_by_passage")):
            errors.append(
                f"{path.name}: item {item_id} has an invalid supported_by_passage value"
            )

    missing_ids = [item.item_id for item in references.get(file_passage, []) if item.item_id not in seen]
    if missing_ids:
        errors.append(
            f"{path.name}: missing reference items {', '.join(missing_ids)}"
        )

    claims = data.get("additional_unsupported_claims", [])
    if not isinstance(claims, list):
        errors.append(f"{path.name}: additional_unsupported_claims must be a list")
    else:
        for index, claim in enumerate(claims, start=1):
            if not isinstance(claim, dict):
                errors.append(f"{path.name}: additional claim {index} must be an object")
                continue
            claim_text = str(claim.get("claim") or "").strip()
            excerpt = str(claim.get("excerpt") or "").strip()
            if not claim_text and not excerpt:
                continue
            if not claim_text:
                errors.append(f"{path.name}: additional claim {index} is missing claim")
            if not excerpt:
                errors.append(f"{path.name}: additional claim {index} is missing excerpt")
    return errors


def _unknown_review_files(run_dir: Path) -> list[str]:
    folder = reviews_dir(run_dir)
    if not folder.is_dir():
        return []
    allowed = {review_filename(passage_id, repeat) for passage_id, repeat, _name in expected_outputs()}
    allowed.add("summary.md")
    unknown: list[str] = []
    for path in sorted(folder.iterdir()):
        if path.name.startswith("."):
            continue
        if path.name not in allowed:
            unknown.append(f"unknown review file {path.name}")
    return unknown


def _ratio(scores: list[int]) -> tuple[int, int, float]:
    total = sum(scores)
    maximum = 2 * len(scores)
    return total, maximum, (total / maximum if maximum else 0.0)


def _format_ratio(total: int, maximum: int) -> str:
    ratio = total / maximum if maximum else 0.0
    return f"{ratio:.2f} ({total}/{maximum})"


def _variation_sentence(label: str, ratios: list[float]) -> str:
    shown = [f"{ratio:.2f}" for ratio in ratios]
    low = min(ratios)
    high = max(ratios)
    if high - low < 1e-9:
        return f"{label} retention was {shown[0]} on each of the three independent runs."
    listed = ", ".join(shown[:-1]) + f", and {shown[-1]}"
    return (
        f"{label} retention across the three independent runs was {listed} "
        f"(lowest {low:.2f}, highest {high:.2f}, spread {high - low:.2f})."
    )


def _scores_for(
    reviews: dict[tuple[str, int], dict], passage_id: str, category: str | None
) -> list[list[int]]:
    columns: list[list[int]] = []
    for repeat in range(1, REPEATS_PER_PASSAGE + 1):
        review = reviews[(passage_id, repeat)]
        scores = []
        for item in review["items"]:
            if category is not None and item["category"] != category:
                continue
            retention = _parse_retention(item["retention"])
            if retention is None:
                raise ReviewCheckError(
                    [f"{passage_id} repeat {repeat}: retention labels are incomplete"]
                )
            scores.append(retention)
        columns.append(scores)
    return columns


def render_summary(
    reviews: dict[tuple[str, int], dict],
    word_counts: dict[tuple[str, int], int],
    references: dict[str, list[ReferenceItem]],
) -> str:
    lines: list[str] = [
        "# Review summary",
        "",
        "Retention compares each saved brief with the reference coding for that passage. "
        "Each reference item is labeled 0 (omitted or distorted), 1 (partly preserved), "
        "or 2 (preserved). Category retention is the sum of those labels divided by "
        "2 times the number of items in the category.",
        "",
        "Word counts are counted from the six response JSON files. Those files were not changed.",
        "",
        "The three runs for each passage are independent repetitions. Differences between "
        "them describe this pilot only. They are not statistically conclusive.",
        "",
        "## Brief word counts",
        "",
        "| Output | Words |",
        "| --- | --- |",
    ]
    for passage_id, repeat, output_name in expected_outputs():
        lines.append(f"| {output_name} | {word_counts[(passage_id, repeat)]} |")

    for passage_id in PASSAGE_IDS:
        lines.extend(["", f"## {passage_id}", ""])
        item_bits = [
            f"{item.item_id} ({CATEGORY_LABELS[item.category]})"
            for item in references[passage_id]
        ]
        lines.append("Reference items: " + "; ".join(item_bits) + ".")
        lines.extend(
            [
                "",
                "| Category | Run 1 | Run 2 | Run 3 | Lowest | Highest | Spread |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        grouped_ratios: dict[str, list[float]] = {}
        for category in (*CATEGORIES, None):
            score_columns = _scores_for(reviews, passage_id, category)
            rendered = []
            ratios = []
            for scores in score_columns:
                total, maximum, ratio = _ratio(scores)
                rendered.append(_format_ratio(total, maximum))
                ratios.append(ratio)
            low = min(ratios)
            high = max(ratios)
            label = "All reference items" if category is None else CATEGORY_LABELS[category]
            lines.append(
                f"| {label} | {rendered[0]} | {rendered[1]} | {rendered[2]} | "
                f"{low:.2f} | {high:.2f} | {high - low:.2f} |"
            )
            if category is not None:
                grouped_ratios[category] = ratios

        lines.append("")
        for category in CATEGORIES:
            lines.append(_variation_sentence(CATEGORY_LABELS[category], grouped_ratios[category]))
        lines.append("")
        lines.append(
            "These figures describe the three saved briefs. They are not statistically conclusive."
        )
        lines.extend(
            [
                "",
                "| Item | Category | Run 1 | Run 2 | Run 3 |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for item in references[passage_id]:
            marks = []
            for repeat in range(1, REPEATS_PER_PASSAGE + 1):
                review = reviews[(passage_id, repeat)]
                match = next(row for row in review["items"] if row["item_id"] == item.item_id)
                marks.append(str(_parse_retention(match["retention"])))
            lines.append(
                f"| {item.item_id} | {CATEGORY_LABELS[item.category]} | "
                f"{marks[0]} | {marks[1]} | {marks[2]} |"
            )

        lines.extend(["", "Passage support among items scored 1 or 2:", ""])
        for repeat in range(1, REPEATS_PER_PASSAGE + 1):
            supported = 0
            unsupported = 0
            for item in reviews[(passage_id, repeat)]["items"]:
                retention = _parse_retention(item["retention"])
                if retention not in (1, 2):
                    continue
                if _parse_support(item.get("supported_by_passage")):
                    supported += 1
                else:
                    unsupported += 1
            lines.append(
                f"- Run {repeat}: {supported} judged supported by the GREVIO passage, "
                f"{unsupported} judged not supported."
            )

        lines.extend(
            [
                "",
                "Additional claims introduced by the brief and not in the reference coding "
                "are not part of retention.",
                "",
            ]
        )
        for repeat in range(1, REPEATS_PER_PASSAGE + 1):
            claims = []
            for claim in reviews[(passage_id, repeat)].get("additional_unsupported_claims") or []:
                claim_text = " ".join(str(claim.get("claim") or "").split())
                excerpt = " ".join(str(claim.get("excerpt") or "").split())
                if claim_text and excerpt:
                    claims.append((claim_text, excerpt))
            if not claims:
                lines.append(f"- Run {repeat}: 0.")
                continue
            rendered = "; ".join(
                f"{text} (excerpt: \"{excerpt}\")" for text, excerpt in claims
            )
            lines.append(f"- Run {repeat}: {len(claims)}. {rendered}")

    lines.append("")
    return "\n".join(lines)


def collect_summary_inputs(
    run_dir: Path, reference_dir: Path | None = None
) -> tuple[dict[tuple[str, int], dict], dict[tuple[str, int], int], dict[str, list[ReferenceItem]]]:
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    reference_root = reference_dir or EVALUATION_DIR
    errors: list[str] = []
    outputs, output_errors = locate_outputs(run_dir)
    errors.extend(output_errors)
    references, reference_errors = load_references(reference_root, require_items=True)
    errors.extend(reference_errors)
    errors.extend(_check_request_files(run_dir))
    errors.extend(_unknown_review_files(run_dir))

    reviews: dict[tuple[str, int], dict] = {}
    word_counts: dict[tuple[str, int], int] = {}
    folder = reviews_dir(run_dir)
    for passage_id, repeat, _output_name in expected_outputs():
        review_path = folder / review_filename(passage_id, repeat)
        if not review_path.is_file():
            errors.append(f"missing review form {review_path.name}")
            continue
        data, load_errors = _load_review(review_path)
        errors.extend(load_errors)
        if data is None or passage_id not in references:
            continue
        errors.extend(_validate_review(review_path, data, references))
        reviews[(passage_id, repeat)] = data
        if (passage_id, repeat) in outputs:
            try:
                text = brief_text_from_response(outputs[(passage_id, repeat)])
            except ReviewCheckError as exc:
                errors.extend(exc.errors)
                continue
            word_counts[(passage_id, repeat)] = count_words(text)
    if errors:
        raise ReviewCheckError(errors)
    return reviews, word_counts, references


def write_summary(run_dir: Path, reference_dir: Path | None = None) -> str:
    reviews, word_counts, references = collect_summary_inputs(run_dir, reference_dir)
    text = render_summary(reviews, word_counts, references)
    path = summary_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text
