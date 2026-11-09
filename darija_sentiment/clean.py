"""Light text normalization for Darija comments.

BERT-style models prefer natural text, so cleaning stays minimal: drop links,
mentions and extra spaces, normalize Arabic letter forms and squash long
letter repeats.
"""

import re

from darija_sentiment.emoji_tags import emojis_to_tags

_URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_MENTION_RE = re.compile(r"@\w+")
# Algerian mobile numbers: 05/06/07 + 8 digits, optionally +213 / 00213 prefix
_PHONE_RE = re.compile(r"(?:\+213|\b00213|\b0)[\s.-]?[5-7](?:[\s.-]?\d){8}\b")
_REPEAT_RE = re.compile(r"(.)\1{2,}")
_SPACES_RE = re.compile(r"\s+")
# harakat (fathatan..sukun), superscript alef and tatweel
_DIACRITICS_RE = re.compile("[\u064b-\u0652\u0670\u0640]")
_ALEF_RE = re.compile("[\u0622\u0623\u0625]")


def normalize_arabic(text: str) -> str:
    """Strip diacritics and tatweel, unify alef forms and alef maqsura."""
    text = _DIACRITICS_RE.sub("", text)
    text = _ALEF_RE.sub("\u0627", text)
    return text.replace("\u0649", "\u064a")


def scrub_personal_data(text: str) -> str:
    """Remove links, emails and phone numbers before anything is stored."""
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    return _PHONE_RE.sub(" ", text)


def clean_text(text: str, emoji_tags: bool = False) -> str:
    """Return a normalized version of a raw comment.

    With `emoji_tags=True`, known emojis become sentiment tags (for TF-IDF).
    """
    if not isinstance(text, str):
        return ""
    text = scrub_personal_data(text)
    text = _MENTION_RE.sub(" ", text)
    text = normalize_arabic(text)
    text = _REPEAT_RE.sub(r"\1\1", text)
    if emoji_tags:
        text = emojis_to_tags(text)
    text = _SPACES_RE.sub(" ", text)
    return text.strip()
