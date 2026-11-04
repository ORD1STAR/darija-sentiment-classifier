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


def test_strips_arabic_diacritics_and_tatweel():
    assert clean_text("مَرْحَبًا بـــكم") == "مرحبا بكم"


def test_unifies_alef_and_alef_maqsura():
    assert clean_text("أنا إسلام آمال على") == "انا اسلام امال علي"


def test_latin_text_unchanged_by_arabic_rules():
    assert clean_text("Allah y3tik saha") == "Allah y3tik saha"
