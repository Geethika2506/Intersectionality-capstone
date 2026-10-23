import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pilot.config import RUNS_DIR
from pilot.requests import PlannedRequest


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_run_directory(base: Path | None = None) -> Path:
    root = base or RUNS_DIR
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = root / stamp
    run_dir.mkdir(parents=False, exist_ok=False)
    return run_dir


def request_path(run_dir: Path, planned: PlannedRequest) -> Path:
    return run_dir / f"{planned.slug}_request.json"


def response_path(run_dir: Path, planned: PlannedRequest) -> Path:
    return run_dir / f"{planned.slug}_response.json"


def failure_path(run_dir: Path, planned: PlannedRequest) -> Path:
    return run_dir / f"{planned.slug}_failure.json"


def run_log_path(run_dir: Path) -> Path:
    return run_dir / "run_log.jsonl"


def existing_response_files(run_dir: Path) -> list[Path]:
    if not run_dir.is_dir():
        return []
    return sorted(run_dir.glob("*_response.json"))


def assert_no_existing_responses(run_dir: Path) -> None:
    existing = existing_response_files(run_dir)
    if existing:
        names = ", ".join(p.name for p in existing)
        raise FileExistsError(
            f"Run directory already contains saved response(s): {names}. "
            "Refusing to overwrite or rerun. Use a new run directory."
        )


def assert_response_not_present(run_dir: Path, planned: PlannedRequest) -> None:
    path = response_path(run_dir, planned)
    if path.exists():
        raise FileExistsError(
            f"Response already saved at {path.name}. Refusing duplicate paid call."
        )


def save_request_artifact(run_dir: Path, planned: PlannedRequest) -> Path:
    path = request_path(run_dir, planned)
    if path.exists():
        raise FileExistsError(f"Request artifact already exists: {path.name}")
    payload = {
        "utc_timestamp": utc_now_iso(),
        "passage_id": planned.passage_id,
        "repeat": planned.repeat,
        "requested_model": planned.requested_model,
        "reasoning_effort": planned.reasoning_effort,
        "input_text": planned.input_text,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def save_response_artifact(run_dir: Path, planned: PlannedRequest, response_body: Any) -> Path:
    path = response_path(run_dir, planned)
    if path.exists():
        raise FileExistsError(f"Response artifact already exists: {path.name}")
    if hasattr(response_body, "model_dump"):
        serialized = response_body.model_dump(mode="json")
    elif isinstance(response_body, dict):
        serialized = response_body
    else:
        serialized = {"raw": str(response_body)}
    path.write_text(json.dumps(serialized, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def save_failure_artifact(
    run_dir: Path, planned: PlannedRequest, error_message: str, detail: Any = None
) -> Path:
    path = failure_path(run_dir, planned)
    payload = {
        "utc_timestamp": utc_now_iso(),
        "passage_id": planned.passage_id,
        "repeat": planned.repeat,
        "error": error_message,
        "detail": detail,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


@dataclass
class RunLogEntry:
    utc_timestamp: str
    passage_id: str
    repeat: int
    requested_model: str
    returned_model: str | None
    reasoning_effort: str
    response_id: str | None
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    request_file: str
    response_file: str | None
    failure_file: str | None
    output_word_count: int | None
    status: str

    def append_to(self, run_dir: Path) -> None:
        log_path = run_log_path(run_dir)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(self), ensure_ascii=False) + "\n")


def extract_response_fields(response: Any) -> tuple[str | None, str | None, dict[str, int | None]]:
    """Return (response_id, returned_model, token_usage dict)."""
    response_id = getattr(response, "id", None)
    returned_model = getattr(response, "model", None)
    usage = getattr(response, "usage", None)
    input_tokens = output_tokens = total_tokens = None
    if usage is not None:
        input_tokens = getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", None)
        output_tokens = getattr(usage, "output_tokens", None) or getattr(
            usage, "completion_tokens", None
        )
        total_tokens = getattr(usage, "total_tokens", None)
    return response_id, returned_model, {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def extract_output_text(response: Any) -> str:
    if hasattr(response, "output_text") and response.output_text:
        return response.output_text
    parts: list[str] = []
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) == "message":
            for content in getattr(item, "content", []) or []:
                if getattr(content, "type", None) == "output_text":
                    parts.append(getattr(content, "text", "") or "")
    return "".join(parts)
