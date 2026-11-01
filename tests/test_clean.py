from darija_sentiment.clean import clean_text


def test_removes_links_and_mentions():
    assert clean_text("@user chouf https://x.com/a hada") == "chouf hada"


def test_squashes_repeated_letters():
    assert clean_text("mrrrrr7ba bezzzzaf") == "mrr7ba bezzaf"


def test_collapses_spaces():
    assert clean_text("  sa7a   ftorkom \n") == "sa7a ftorkom"


def test_keeps_arabic_and_emojis():
    assert clean_text("مليح بزاف 😍") == "مليح بزاف 😍"


def test_non_string_returns_empty():
    assert clean_text(None) == ""
