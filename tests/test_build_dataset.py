import json

import pandas as pd

from darija_sentiment.build_dataset import anonymize, build_dataset
from darija_sentiment.cli import main
from darija_sentiment.validate import SCHEMA, validate_file


def write_raw(path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


RAW = [
    {"comment_id": "c1", "video_id": "v1", "text": "@Karim_dz sa7a khoya 😍"},
    {"comment_id": "c2", "video_id": "v1", "text": "3aytouli 0550 12 34 56 nchri"},
    {"comment_id": "c3", "video_id": "v2", "text": "sa7a khoya 😍"},
    {"comment_id": "c4", "video_id": "v2", "text": "Abonnez-vous à ma chaîne"},
    {"comment_id": "c5", "video_id": "v2", "text": "الكونيكسيون ماكاش، contact@mail.dz"},
]


def test_anonymize_removes_mentions_phones_emails():
    assert anonymize("@ali.b chouf 0661234567 wela a@b.dz") == "chouf wela"


def test_build_follows_schema_with_empty_labels(tmp_path):
    raw = tmp_path / "raw.jsonl"
    write_raw(raw, RAW)
    before = raw.read_bytes()
    out = tmp_path / "to_label.csv"
    report = build_dataset([raw], out, topics={"v1": "comedy", "v2": "telecom"})

    df = pd.read_csv(out, dtype=str, keep_default_na=False)
    assert tuple(df.columns) == SCHEMA
    assert (df["label"] == "").all()
    assert validate_file(out, allow_unlabeled=True) == []
    texts = list(df["text"])
    assert texts[0] == "sa7a khoya 😍"
    assert not any("@" in t or "0550" in t for t in texts)
    assert report["duplicate"] == 1 and report["spam_promo"] == 1
    assert raw.read_bytes() == before


def test_cli_build(tmp_path, capsys):
    raw = tmp_path / "raw.jsonl"
    write_raw(raw, RAW[:2])
    out = tmp_path / "o.csv"
    assert main(["build-dataset", str(raw), "--out", str(out)]) == 0
    assert '"kept": 2' in capsys.readouterr().out
