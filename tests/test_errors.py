import pandas as pd

from darija_sentiment.baseline import train
from darija_sentiment.cli import main
from darija_sentiment.data import DEFAULT_DATA
from darija_sentiment.errors import ERROR_COLUMNS, summarize, wrong_predictions


def table():
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "text": ["a", "b", "c", "d"],
            "script": ["latin", "arabic", "latin", "latin"],
            "topic": ["music"] * 4,
            "label": ["positive", "negative", "neutral", "positive"],
            "pred": ["negative", "negative", "positive", "neutral"],
            "confidence": [0.55, 0.9, 0.81, 0.62],
        }
    )


def test_wrong_predictions_sorted_by_confidence():
    errors = wrong_predictions(table())
    assert list(errors.columns) == ERROR_COLUMNS
    assert list(errors["id"]) == [3, 4, 1]
    assert list(errors["gold"]) == ["neutral", "positive", "positive"]


def test_summary_groups_by_pair_and_script():
    text = summarize(wrong_predictions(table()), 4)
    assert "3 wrong predictions out of 4" in text
    assert "| neutral | positive | 1 |" in text
    assert "| latin | 3 |" in text


def test_errors_cli(tmp_path):
    model = tmp_path / "m.joblib"
    train(DEFAULT_DATA, model)
    out = tmp_path / "errs"
    assert main(["errors", "--model", str(model), "--out", str(out)]) == 0
    errors = pd.read_csv(out / "errors.csv")
    assert list(errors.columns) == ERROR_COLUMNS
    assert errors["confidence"].is_monotonic_decreasing
    assert (out / "errors_summary.md").exists()
