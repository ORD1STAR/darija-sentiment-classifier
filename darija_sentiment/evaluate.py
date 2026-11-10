"""Evaluation of a saved model on one split: metrics, per-class table, confusion matrix."""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from darija_sentiment import LABELS
from darija_sentiment.data import load_dataset, split_three


def predict_split(model, df: pd.DataFrame) -> pd.DataFrame:
    """Return the split with `pred` and `confidence` columns added."""
    out = df.copy()
    probs = model.predict_proba(out["clean"])
    classes = list(model.classes_)
    out["pred"] = [classes[i] for i in probs.argmax(axis=1)]
    out["confidence"] = probs.max(axis=1)
    return out


def compute_metrics(gold: pd.Series, pred: pd.Series) -> dict:
    """Headline metric is macro F1; labels always appear in LABELS order."""
    labels = list(LABELS)
    precision, recall, f1, support = precision_recall_fscore_support(
        gold, pred, labels=labels, zero_division=0
    )
    return {
        "n": int(len(gold)),
        "macro_f1": float(f1_score(gold, pred, labels=labels, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(gold, pred)),
        "per_class": {
            label: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(labels)
        },
        "confusion_matrix": {
            "labels": labels,
            "matrix": confusion_matrix(gold, pred, labels=labels).tolist(),
        },
    }


def evaluate(
    model_path: str | Path,
    data_path: str | Path,
    split: str = "test",
    seed: int = 42,
    out_dir: str | Path | None = None,
) -> dict:
    model = joblib.load(model_path)
    df = split_three(load_dataset(data_path), seed=seed)[split]
    table = predict_split(model, df)
    metrics = {"split": split, **compute_metrics(table["label"], table["pred"])}
    if out_dir is not None:
        save_report(metrics, out_dir)
    return metrics


def save_report(metrics: dict, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    cm = metrics["confusion_matrix"]
    frame = pd.DataFrame(cm["matrix"], index=cm["labels"], columns=cm["labels"])
    frame.index.name = r"gold\pred"
    frame.to_csv(out_dir / "confusion_matrix.csv", lineterminator="\n")
