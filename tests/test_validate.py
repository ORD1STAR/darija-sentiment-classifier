import pandas as pd

from darija_sentiment.cli import main
from darija_sentiment.data import DEFAULT_DATA
from darija_sentiment.validate import SCHEMA, validate_frame


def good_frame():
    return pd.DataFrame(
        {
            "id": [1, 2],
            "text": ["sa7a", "chhal?"],
            "script": ["latin", "latin"],
            "source": ["seed", "seed"],
            "topic": ["music", "reviews"],
            "label": ["positive", "neutral"],
            "notes": ["", ""],
        }
    )


def test_sample_file_is_valid(capsys):
    assert main(["validate", str(DEFAULT_DATA)]) == 0
    assert "valid" in capsys.readouterr().out


def test_good_frame_has_no_problems():
    assert validate_frame(good_frame()) == []


def test_reports_missing_columns():
    problems = validate_frame(good_frame().drop(columns=["notes", "topic"]))
    assert any("missing columns" in p and "notes" in p and "topic" in p for p in problems)


def test_reports_duplicate_ids_and_empty_text():
    df = good_frame()
    df.loc[1, "id"] = 1
    df.loc[0, "text"] = "  "
    problems = validate_frame(df)
    assert any(p.startswith("duplicate ids: 1") for p in problems)
    assert any(p.startswith("empty text") for p in problems)


def test_unlabeled_rows_allowed_only_with_flag():
    df = good_frame()
    df.loc[0, "label"] = None
    assert validate_frame(df) != []
    assert validate_frame(df, allow_unlabeled=True) == []


def test_cli_fails_on_invalid_file(tmp_path):
    path = tmp_path / "bad.csv"
    good_frame().assign(label=["happy", "neutral"]).to_csv(path, index=False)
    assert main(["validate", str(path)]) == 1


def test_schema_order():
    assert SCHEMA == ("id", "text", "script", "source", "topic", "label", "notes")
