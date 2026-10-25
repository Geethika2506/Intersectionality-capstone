from dataclasses import dataclass

from pilot.config import (
    PASSAGE_IDS,
    REASONING_EFFORT,
    REPEATS_PER_PASSAGE,
    REQUESTED_MODEL,
)
from pilot.passages import PassageBundle, load_all_passages


def artifact_stem(passage_id: str, repeat: int) -> str:
    return f"{passage_id}_repeat_{repeat:02d}"


@dataclass(frozen=True)
class PlannedRequest:
    passage_id: str
    repeat: int
    requested_model: str
    reasoning_effort: str
    input_text: str

    @property
    def slug(self) -> str:
        return artifact_stem(self.passage_id, self.repeat)


def build_planned_requests(bundles: list[PassageBundle] | None = None) -> list[PlannedRequest]:
    if bundles is None:
        bundles = load_all_passages()
    by_id = {b.passage_id: b for b in bundles}
    planned: list[PlannedRequest] = []
    for passage_id in PASSAGE_IDS:
        bundle = by_id[passage_id]
        for repeat in range(1, REPEATS_PER_PASSAGE + 1):
            planned.append(
                PlannedRequest(
                    passage_id=passage_id,
                    repeat=repeat,
                    requested_model=REQUESTED_MODEL,
                    reasoning_effort=REASONING_EFFORT,
                    input_text=bundle.model_input_text(),
                )
            )
    return planned


def api_create_kwargs(planned: PlannedRequest) -> dict:
    """Keyword arguments for OpenAI Responses API (no tools, no prior context)."""
    return {
        "model": planned.requested_model,
        "input": planned.input_text,
        "reasoning": {"effort": planned.reasoning_effort},
    }
