"""Light text normalization for Darija comments.

BERT-style models prefer natural text, so cleaning stays minimal: drop links,
mentions and extra spaces, and squash long letter repeats.
"""

import re

_URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
_MENTION_RE = re.compile(r"@\w+")
_REPEAT_RE = re.compile(r"(.)\1{2,}")
_SPACES_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Return a normalized version of a raw comment."""
    if not isinstance(text, str):
        return ""
    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = _REPEAT_RE.sub(r"\1\1", text)
    text = _SPACES_RE.sub(" ", text)
    return text.strip()
