from darija_sentiment.baseline import train
from darija_sentiment.cli import main
from darija_sentiment.data import DEFAULT_DATA
from darija_sentiment.evaluate import evaluate
from darija_sentiment.report import render_markdown, write_report

METRICS = {
    "split": "test",
    "n": 4,
    "macro_f1": 0.5,
    "accuracy": 0.75,
    "per_class": {
        "negative": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 1},
        "neutral": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 1},
        "positive": {"precision": 0.67, "recall": 1.0, "f1": 0.8, "support": 2},
    },
    "confusion_matrix": {
        "labels": ["negative", "neutral", "positive"],
        "matrix": [[1, 0, 0], [0, 0, 1], [0, 0, 2]],
    },
    "breakdowns": {
        "script": {"arabic": {"n": 2, "macro_f1": 1.0, "accuracy": 1.0, "low_support": True}}
    },
}


def test_markdown_contains_all_sections():
    text = render_markdown(METRICS)
    assert "| Macro F1 | 0.500 |" in text
    assert "| positive | 0.670 | 1.000 | 0.800 | 2 |" in text
    assert "## By script" in text and "arabic (low support)" in text
    assert "![Confusion matrix](confusion_matrix.png)" in text


def test_report_is_deterministic(tmp_path):
    model = tmp_path / "m.joblib"
    train(DEFAULT_DATA, model)
    evaluate(model, DEFAULT_DATA, out_dir=tmp_path)
    first = write_report(tmp_path / "metrics.json", tmp_path / "a").read_bytes()
    second = write_report(tmp_path / "metrics.json", tmp_path / "b").read_bytes()
    assert first == second
    png = (tmp_path / "a" / "confusion_matrix.png").read_bytes()
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_report_cli(tmp_path):
    model = tmp_path / "m.joblib"
    train(DEFAULT_DATA, model)
    evaluate(model, DEFAULT_DATA, out_dir=tmp_path)
    out = tmp_path / "rep"
    assert main(["report", "--metrics", str(tmp_path / "metrics.json"), "--out", str(out)]) == 0
    assert (out / "results.md").exists() and (out / "confusion_matrix.png").exists()
