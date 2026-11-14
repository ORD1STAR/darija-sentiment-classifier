import pandas as pd

from darija_sentiment.cli import main
from darija_sentiment.dedupe import dedupe_and_filter, spam_reason


def test_spam_rules():
    assert spam_reason("https://bit.ly/xyz") == "links_only"
    assert spam_reason("Abonnez-vous à ma chaîne") == "promo"
    assert spam_reason("contact 0550 12 34 56") == "phone_ad"
    assert spam_reason("rbe7na 3-0 f 2019, sa7a") is None
    assert spam_reason("video mlih chouf https://x.com hadi zeda") is None


def test_exact_duplicates_after_cleaning_are_dropped():
    df = pd.DataFrame({"text": ["Top bezzaaaaf", "top  bezzaf", "khayeb", "promo whatsapp now"]})
    kept, report = dedupe_and_filter(df)
    assert list(kept["text"]) == ["Top bezzaaaaf", "khayeb"]
    assert report == {"spam_promo": 1, "empty": 0, "duplicate": 1, "kept": 2}


def test_cli_writes_filtered_file(tmp_path, capsys):
    src, out = tmp_path / "in.csv", tmp_path / "out.csv"
    pd.DataFrame({"text": ["sa7a", "sa7a", "https://spam.com"]}).to_csv(src, index=False)
    assert main(["dedupe", str(src), "--out", str(out)]) == 0
    assert list(pd.read_csv(out)["text"]) == ["sa7a"]
    assert '"duplicate": 1' in capsys.readouterr().out
