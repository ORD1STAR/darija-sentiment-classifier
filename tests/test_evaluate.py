import json

import pandas as pd
import pytest

from darija_sentiment import LABELS
from darija_sentiment.baseline import train
from darija_sentiment.cli import main
from darija_sentiment.data import DEFAULT_DATA
from darija_sentiment.evaluate import compute_metrics, evaluate


@pytest.fixture(scope="module")
def model_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("model") / "m.joblib"
    train(DEFAULT_DATA, path)
    return path


def test_compute_metrics_known_values():
    gold = pd.Series(["positive", "positive", "negative", "neutral"])
    pred = pd.Series(["positive", "negative", "negative", "neutral"])
    m = compute_metrics(gold, pred)
    assert m["accuracy"] == pytest.approx(0.75)
    assert m["per_class"]["positive"]["recall"] == pytest.approx(0.5)
    assert m["per_class"]["negative"]["precision"] == pytest.approx(0.5)
    assert m["confusion_matrix"]["labels"] == list(LABELS)
    # rows = gold in LABELS order (negative, neutral, positive)
    assert m["confusion_matrix"]["matrix"] == [[1, 0, 0], [0, 1, 0], [1, 0, 1]]


def test_evaluate_writes_report(model_path, tmp_path):
    metrics = evaluate(model_path, DEFAULT_DATA, "test", out_dir=tmp_path)
    saved = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert saved["macro_f1"] == pytest.approx(metrics["macro_f1"])
    cm = pd.read_csv(tmp_path / "confusion_matrix.csv", index_col=0)
    assert list(cm.columns) == list(LABELS)
    assert int(cm.values.sum()) == metrics["n"]


def test_evaluate_cli(model_path, tmp_path, capsys):
    out = tmp_path / "rep"
    assert main(["evaluate", "--model", str(model_path), "--out", str(out)]) == 0
    assert "macro F1" in capsys.readouterr().out
    assert (out / "metrics.json").exists()
