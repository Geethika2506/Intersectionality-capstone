import re


def count_words(text: str) -> int:
    """Count words using whitespace tokenization (consistent for run logs)."""
    if not text or not text.strip():
        return 0
    return len(re.findall(r"\S+", text.strip()))
