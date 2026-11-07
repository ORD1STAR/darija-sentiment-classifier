"""Writing-system detection: Arabic script, Latin script (incl. Arabizi) or mixed."""

import re

_ARABIC_RE = re.compile("[ء-يٱ-ۓ]")
_LATIN_RE = re.compile(r"[A-Za-zÀ-ɏ]")
# Arabizi digits stand for Arabic sounds (3 = ain, 7 = ha, 9 = qaf ...) inside Latin words
_ARABIZI_DIGIT_RE = re.compile(r"(?<=[A-Za-z])[235679]|[235679](?=[A-Za-z])")

# minority script share above which a comment counts as mixed
MIXED_THRESHOLD = 0.15


def detect_script(text: str) -> str:
    """Return `arabic`, `latin`, `mixed`, or `unknown` when there are no letters."""
    if not isinstance(text, str):
        return "unknown"
    arabic = len(_ARABIC_RE.findall(text))
    latin = len(_LATIN_RE.findall(text)) + len(_ARABIZI_DIGIT_RE.findall(text))
    total = arabic + latin
    if total == 0:
        return "unknown"
    if min(arabic, latin) / total >= MIXED_THRESHOLD:
        return "mixed"
    return "arabic" if arabic > latin else "latin"
