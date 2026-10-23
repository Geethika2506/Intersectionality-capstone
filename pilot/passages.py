from dataclasses import dataclass
from pathlib import Path

from pilot.config import DATA_DIR, PLACEHOLDER_MARKERS, PROMPTS_DIR, PASSAGE_IDS


@dataclass(frozen=True)
class PassageBundle:
    passage_id: str
    prompt_text: str
    source_text: str

    def model_input_text(self) -> str:
        """Single standalone user message: fixed prompt then original source."""
        return f"{self.prompt_text.strip()}\n\n---\n\n{self.source_text.strip()}\n"


def _read_text(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}")
    return path.read_text(encoding="utf-8")


def load_passage(passage_id: str) -> PassageBundle:
    if passage_id not in PASSAGE_IDS:
        raise ValueError(f"Unknown passage_id: {passage_id}")
    prompt_path = PROMPTS_DIR / f"{passage_id}.txt"
    data_path = DATA_DIR / f"{passage_id}.txt"
    return PassageBundle(
        passage_id=passage_id,
        prompt_text=_read_text(prompt_path),
        source_text=_read_text(data_path),
    )


def load_all_passages() -> list[PassageBundle]:
    return [load_passage(pid) for pid in PASSAGE_IDS]


def contains_placeholder(text: str) -> bool:
    upper = text.upper()
    for marker in PLACEHOLDER_MARKERS:
        if marker.upper() in upper:
            return True
    return False


def validate_ready_for_paid_run(bundles: list[PassageBundle]) -> list[str]:
    errors: list[str] = []
    for bundle in bundles:
        if contains_placeholder(bundle.prompt_text):
            errors.append(
                f"{bundle.passage_id}: prompts/{bundle.passage_id}.txt still contains a placeholder marker"
            )
        if contains_placeholder(bundle.source_text):
            errors.append(
                f"{bundle.passage_id}: data/{bundle.passage_id}.txt still contains a placeholder marker"
            )
        if not bundle.prompt_text.strip():
            errors.append(f"{bundle.passage_id}: prompt file is empty")
        if not bundle.source_text.strip():
            errors.append(f"{bundle.passage_id}: source file is empty")
    return errors
