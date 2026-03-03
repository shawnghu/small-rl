"""Shared utilities for QA-style environments."""

import json
import re
from functools import lru_cache


_LAST_WORD_RE = re.compile(r"[a-zA-ZáéíóúñüÁÉÍÓÚÑÜ]+")


def extract_last_word(completion: str) -> str:
    """Extract the last alphabetic word from a completion, lowercased.

    Strips punctuation and whitespace. Returns empty string if no word found.
    """
    matches = _LAST_WORD_RE.findall(completion)
    if not matches:
        return ""
    return matches[-1].lower()


def extract_tf(completion: str) -> str:
    """Extract true/false answer from a completion.

    Returns 'true' or 'false' if exactly one appears. Returns '' if neither
    or both appear (hedging both answers gets no credit).
    """
    words = [w.lower() for w in _LAST_WORD_RE.findall(completion)]
    has_true = "true" in words
    has_false = "false" in words
    if has_true and has_false:
        return ""
    if has_true:
        return "true"
    if has_false:
        return "false"
    return ""


def check_tf(completion: str) -> bool | None:
    """Interpret True/False from a completion.

    Returns True/False if exactly one of 'true'/'false' appears, None otherwise.
    Completions containing both words get None (no hedging).
    """
    word = extract_tf(completion)
    if word == "true":
        return True
    elif word == "false":
        return False
    return None


@lru_cache(maxsize=32)
def load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file, returning list of dicts. Cached by path."""
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    assert entries, f"Empty JSONL file: {path}"
    return entries


def contains_word(text: str, word: str) -> bool:
    """Check if word appears in text using word-boundary regex, case-insensitive."""
    pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
    return bool(pattern.search(text))


def extract_last_number(completion: str) -> str | None:
    """Extract the last contiguous number from a completion.

    Returns the number as a string, or None if no number found.
    """
    matches = re.findall(r'\d+', completion)
    if not matches:
        return None
    return matches[-1]
