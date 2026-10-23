from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pilot.config import REASONING_EFFORT, REPEATS_PER_PASSAGE, REQUESTED_MODEL
from pilot.passages import PassageBundle, contains_placeholder, validate_ready_for_paid_run
from pilot.requests import api_create_kwargs, build_planned_requests
from pilot.runner import format_dry_run_line
from pilot.storage import (
    assert_no_existing_responses,
    assert_response_not_present,
    request_path,
    response_path,
    save_request_artifact,
    save_response_artifact,
)
from pilot.word_count import count_words


def _bundle(passage_id: str, prompt: str, source: str) -> PassageBundle:
    return PassageBundle(passage_id=passage_id, prompt_text=prompt, source_text=source)


def test_builds_six_independent_requests_with_identical_settings():
    bundles = [
        _bundle("passage_01", "Prompt one.", "Source one."),
        _bundle("passage_02", "Prompt two.", "Source two."),
    ]
    planned = build_planned_requests(bundles)
    assert len(planned) == 6
    assert all(p.requested_model == REQUESTED_MODEL for p in planned)
    assert all(p.reasoning_effort == REASONING_EFFORT for p in planned)

    p1 = [p for p in planned if p.passage_id == "passage_01"]
    p2 = [p for p in planned if p.passage_id == "passage_02"]
    assert len(p1) == REPEATS_PER_PASSAGE
    assert len(p2) == REPEATS_PER_PASSAGE
    assert {p.repeat for p in p1} == {1, 2, 3}

    inputs_p1 = {p.input_text for p in p1}
    inputs_p2 = {p.input_text for p in p2}
    assert len(inputs_p1) == 1
    assert len(inputs_p2) == 1
    assert inputs_p1 != inputs_p2


def test_api_kwargs_have_no_tools_or_history():
    bundles = [_bundle("passage_01", "P", "S"), _bundle("passage_02", "P2", "S2")]
    planned = build_planned_requests(bundles)[0]
    kwargs = api_create_kwargs(planned)
    assert kwargs == {
        "model": REQUESTED_MODEL,
        "input": planned.input_text,
        "reasoning": {"effort": REASONING_EFFORT},
    }
    assert "previous_response_id" not in kwargs
    assert "tools" not in kwargs


def test_dry_run_line_omits_full_passage(tmp_path: Path):
    long_source = "word " * 500
    planned = build_planned_requests(
        [_bundle("passage_01", "Brief task.", long_source), _bundle("passage_02", "P", "S")]
    )[0]
    line = format_dry_run_line(planned, 1)
    assert "full passage text omitted" in line
    assert long_source.strip() not in line
    assert planned.slug in line


def test_word_count():
    assert count_words("") == 0
    assert count_words("one two\nthree") == 3
    assert count_words("  spaced   words  ") == 2


def test_placeholder_validation_blocks_run():
    bundles = [
        _bundle("passage_01", "[PLACEHOLDER — x]", "Real text here."),
        _bundle("passage_02", "OK prompt.", "Also OK."),
    ]
    errors = validate_ready_for_paid_run(bundles)
    assert any("passage_01" in e for e in errors)


def test_refuses_overwriting_saved_response(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    bundles = [_bundle("passage_01", "P", "S"), _bundle("passage_02", "P", "S")]
    planned = build_planned_requests(bundles)[0]
    save_request_artifact(run_dir, planned)
    save_response_artifact(run_dir, planned, {"id": "resp_1", "output": []})

    with pytest.raises(FileExistsError):
        assert_no_existing_responses(run_dir)

    planned2 = build_planned_requests(bundles)[1]
    with pytest.raises(FileExistsError):
        save_response_artifact(run_dir, planned, {"id": "resp_dup"})

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    assert_response_not_present(empty_dir, planned2)  # ok
    save_response_artifact(empty_dir, planned, {"id": "x"})
    with pytest.raises(FileExistsError):
        assert_response_not_present(empty_dir, planned)


def test_execute_paid_run_stops_on_existing_response(tmp_path: Path, monkeypatch):
    from pilot import runner as runner_mod

    run_dir = tmp_path / "existing"
    run_dir.mkdir()
    bundles = [_bundle("passage_01", "Filled prompt.", "Filled source."), _bundle("passage_02", "P2", "S2")]
    planned = build_planned_requests(bundles)[0]
    save_response_artifact(run_dir, planned, {"id": "already"})

    monkeypatch.setattr(runner_mod, "load_all_passages", lambda: bundles)
    with pytest.raises(FileExistsError):
        runner_mod.execute_paid_run(run_dir=run_dir, client=MagicMock())


def test_execute_paid_run_one_failure_no_retry(tmp_path: Path, monkeypatch):
    from pilot import runner as runner_mod

    run_dir = tmp_path / "fail_run"
    run_dir.mkdir()
    bundles = [_bundle("passage_01", "Filled prompt.", "Filled source."), _bundle("passage_02", "P2", "S2")]
    monkeypatch.setattr(runner_mod, "load_all_passages", lambda: bundles)

    mock_client = MagicMock()
    mock_client.responses.create.side_effect = RuntimeError("API down")

    code = runner_mod.execute_paid_run(run_dir=run_dir, client=mock_client)
    assert code == 1
    assert mock_client.responses.create.call_count == 1
    assert list(run_dir.glob("*_failure.json"))
