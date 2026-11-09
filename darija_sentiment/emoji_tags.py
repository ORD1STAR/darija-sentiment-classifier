"""Map common emojis to sentiment tags, useful for bag-of-ngrams models."""

TAGS = {
    "EMO_POS": [
        "😍",
        "😂",
        "🤣",
        "❤️",
        "❤",
        "💚",
        "💪",
        "👏",
        "👍",
        "🙏",
        "😊",
        "😁",
        "🥰",
        "😘",
        "🔥",
        "✅",
        "🌹",
        "💯",
        "🇩🇿",
    ],
    "EMO_NEG": ["😡", "🤬", "😠", "👎", "💔", "😢", "😭", "😤", "🤮", "😒", "🙄", "😞"],
    "EMO_NEU": ["🤔", "😐", "😶", "👀"],
}

EMOJI_TO_TAG = {emoji: tag for tag, emojis in TAGS.items() for emoji in emojis}
# longest first so "❤️" (with variation selector) wins over a bare "❤"
_ORDERED = sorted(EMOJI_TO_TAG, key=len, reverse=True)


def emojis_to_tags(text: str) -> str:
    """Replace known emojis by ` EMO_POS ` / ` EMO_NEG ` / ` EMO_NEU `."""
    for emoji in _ORDERED:
        if emoji in text:
            text = text.replace(emoji, f" {EMOJI_TO_TAG[emoji]} ")
    return text
