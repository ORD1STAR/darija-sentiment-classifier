import subprocess
import sys

import pandas as pd

from darija_sentiment.baseline import train
from darija_sentiment.cli import main
from darija_sentiment.data import DEFAULT_DATA
from darija_sentiment.evaluate import evaluate
from darija_sentiment.publish import export_cards, export_dataset, export_space


def test_dataset_export_has_only_text_and_label(tmp_path):
    out = tmp_path / "d.csv"
    n = export_dataset(DEFAULT_DATA, out)
    df = pd.read_csv(out)
    assert list(df.columns) == ["text", "label"]
    assert len(df) == n


def test_space_bundle_is_self_contained(tmp_path):
    model = tmp_path / "m.joblib"
    train(DEFAULT_DATA, model)
    space = export_space(tmp_path / "space", model)
    for rel in ("app.py", "requirements.txt", "README.md", "artifacts/baseline.joblib"):
        assert (space / rel).exists(), rel
    assert (space / "darija_sentiment" / "predict.py").exists()
    assert "sdk: gradio" in (space / "README.md").read_text(encoding="utf-8")
    assert "gradio" in (space / "requirements.txt").read_text(encoding="utf-8")
    # the bundled package imports on its own, without the repo on sys.path
    code = "import darija_sentiment.predict as p; print(p.Predictor('artifacts/baseline.joblib').predict('sa7a')['label'])"
    out = subprocess.run([sys.executable, "-c", code], cwd=space, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() in {"positive", "negative", "neutral"}


def test_cards_from_metrics_and_guide(tmp_path):
    model = tmp_path / "m.joblib"
    train(DEFAULT_DATA, model)
    evaluate(model, DEFAULT_DATA, out_dir=tmp_path)
    paths = export_cards(DEFAULT_DATA, tmp_path / "metrics.json", tmp_path / "hub")
    model_card = paths["model_card"].read_text(encoding="utf-8")
    dataset_card = paths["dataset_card"].read_text(encoding="utf-8")
    assert "Macro F1" in model_card and "pipeline_tag: text-classification" in model_card
    assert "## Labeling guide" in dataset_card and "sarcasm" in dataset_card.lower()


def test_cli_exports(tmp_path):
    model = tmp_path / "m.joblib"
    assert main(["train", "--model", str(model)]) == 0
    assert main(["export-space", "--model", str(model), "--out", str(tmp_path / "s")]) == 0
    assert (tmp_path / "s" / "app.py").exists()
