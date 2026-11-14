"""Duplicate and spam removal, with a report of how many rows each rule dropped."""

import re
from collections import Counter

import pandas as pd

from darija_sentiment.clean import clean_text

PROMO_KEYWORDS = (
    "abonnez",
    "abonne toi",
    "subscribe",
    "inbox",
    "promo",
    "whatsapp",
    "viber",
    "livraison gratuite",
    "gagner de l'argent",
    "اشترك",
    "تواصل معنا",
)
_URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
_DIGITS_RE = re.compile(r"\d[\d\s.-]{7,}\d")


def spam_reason(text: str) -> str | None:
    """Return the spam rule that matches, or None for a normal comment."""
    if not isinstance(text, str):
        return None
    if _URL_RE.search(text) and not _URL_RE.sub("", text).strip():
        return "links_only"
    lowered = text.lower()
    if any(k in lowered for k in PROMO_KEYWORDS):
        return "promo"
    if len(_DIGITS_RE.findall(text)) >= 1 and len(text.split()) <= 6:
        return "phone_ad"
    return None


def dedupe_and_filter(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop spam, then exact duplicates after cleaning. Keeps the first occurrence."""
    report: Counter[str] = Counter()
    reasons = df["text"].map(spam_reason)
    for reason, count in reasons.dropna().value_counts().items():
        report[f"spam_{reason}"] += int(count)
    kept = df[reasons.isna()].copy()

    key = kept["text"].map(clean_text).str.lower()
    empty = key == ""
    report["empty"] += int(empty.sum())
    kept, key = kept[~empty], key[~empty]

    dup = key.duplicated(keep="first")
    report["duplicate"] += int(dup.sum())
    kept = kept[~dup].reset_index(drop=True)
    report["kept"] = len(kept)
    return kept, dict(report)
