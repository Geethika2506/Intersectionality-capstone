from pathlib import Path

from openai import OpenAI

from pilot.passages import load_all_passages, validate_ready_for_paid_run
from pilot.requests import api_create_kwargs, build_planned_requests
from pilot.storage import (
    RunLogEntry,
    assert_no_existing_responses,
    assert_response_not_present,
    extract_output_text,
    extract_response_fields,
    new_run_directory,
    save_failure_artifact,
    save_request_artifact,
    save_response_artifact,
    utc_now_iso,
)
from pilot.word_count import count_words


def format_dry_run_line(planned, index: int) -> str:
    prompt_chars = len(planned.input_text)
    preview = planned.input_text[:80].replace("\n", " ")
    if len(planned.input_text) > 80:
        preview += "…"
    return (
        f"{index}. {planned.slug} | model={planned.requested_model} | "
        f"reasoning={planned.reasoning_effort} | input_chars={prompt_chars} | "
        f"preview={preview!r} (full passage text omitted in dry-run)"
    )


def dry_run() -> int:
    bundles = load_all_passages()
    planned_list = build_planned_requests(bundles)
    print(f"Planned {len(planned_list)} independent Responses API calls (no history, no tools).")
    for i, planned in enumerate(planned_list, start=1):
        print(format_dry_run_line(planned, i))
    errors = validate_ready_for_paid_run(bundles)
    if errors:
        print("\nValidation (paid run would be blocked until fixed):")
        for err in errors:
            print(f"  - {err}")
    else:
        print("\nValidation: prompts and passages contain no placeholder markers.")
    return 0


def execute_paid_run(run_dir: Path | None = None, client: OpenAI | None = None) -> int:
    bundles = load_all_passages()
    errors = validate_ready_for_paid_run(bundles)
    if errors:
        print("Cannot run: fill in data/ and prompts/ and remove placeholders.")
        for err in errors:
            print(f"  - {err}")
        return 1

    directory = run_dir or new_run_directory()
    assert_no_existing_responses(directory)

    api_client = client or OpenAI()
    planned_list = build_planned_requests(bundles)

    for planned in planned_list:
        assert_response_not_present(directory, planned)
        req_path = save_request_artifact(directory, planned)
        try:
            kwargs = api_create_kwargs(planned)
            response = api_client.responses.create(**kwargs)
            resp_path = save_response_artifact(directory, planned, response)
            text = extract_output_text(response)
            response_id, returned_model, usage = extract_response_fields(response)
            RunLogEntry(
                utc_timestamp=utc_now_iso(),
                passage_id=planned.passage_id,
                repeat=planned.repeat,
                requested_model=planned.requested_model,
                returned_model=returned_model,
                reasoning_effort=planned.reasoning_effort,
                response_id=response_id,
                input_tokens=usage["input_tokens"],
                output_tokens=usage["output_tokens"],
                total_tokens=usage["total_tokens"],
                request_file=req_path.name,
                response_file=resp_path.name,
                failure_file=None,
                output_word_count=count_words(text),
                status="ok",
            ).append_to(directory)
            print(f"OK {planned.slug} -> {resp_path.name} ({count_words(text)} words)")
        except FileExistsError:
            raise
        except Exception as exc:  # noqa: BLE001 — log and stop; no silent retry
            fail_path = save_failure_artifact(directory, planned, str(exc))
            RunLogEntry(
                utc_timestamp=utc_now_iso(),
                passage_id=planned.passage_id,
                repeat=planned.repeat,
                requested_model=planned.requested_model,
                returned_model=None,
                reasoning_effort=planned.reasoning_effort,
                response_id=None,
                input_tokens=None,
                output_tokens=None,
                total_tokens=None,
                request_file=req_path.name,
                response_file=None,
                failure_file=fail_path.name,
                output_word_count=None,
                status="error",
            ).append_to(directory)
            print(f"FAILED {planned.slug}: {exc} (logged to {fail_path.name})")
            return 1

    print(f"Completed run in {directory}")
    return 0
