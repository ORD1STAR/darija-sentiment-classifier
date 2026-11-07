import pytest

from darija_sentiment.script import detect_script


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("فيديو مليح بزاف", "arabic"),
        ("video mlih bezzaf", "latin"),
        ("l'équipe rahi top", "latin"),
        ("الفيديو هذا top bezzaf", "mixed"),
        ("😍😍", "unknown"),
        ("", "unknown"),
    ],
)
def test_detect_script(text, expected):
    assert detect_script(text) == expected


def test_arabizi_digits_count_as_latin():
    assert detect_script("3") == "unknown"
    assert detect_script("ya3tik saha") == "latin"
    assert detect_script("w9 3") == "latin"
