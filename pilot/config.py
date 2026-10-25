from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = REPO_ROOT / "data"
PROMPTS_DIR = REPO_ROOT / "prompts"
EVALUATION_DIR = REPO_ROOT / "evaluation"
RUNS_DIR = REPO_ROOT / "runs"

PASSAGE_IDS = ("passage_01", "passage_02")
REPEATS_PER_PASSAGE = 3

REQUESTED_MODEL = "gpt-6.1-sol"
REASONING_EFFORT = "medium"

PLACEHOLDER_MARKERS = (
    "[PLACEHOLDER",
    "PLACEHOLDER —",
    "PLACEHOLDER -",
    "<INSERT",
    "TODO:",
    "FIXME:",
)
