"""Inter-annotator agreement (Cohen's kappa) between two label files."""

from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from darija_sentiment import LABELS

GOOD_KAPPA = 0.6


def agreement(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """Compare labels on the ids present in both frames."""
    merged = a[["id", "label"]].merge(b[["id", "label"]], on="id", suffixes=("_a", "_b"))
    merged = merged.dropna(subset=["label_a", "label_b"])
    if merged.empty:
        raise ValueError("the two files share no labeled ids")
    labels = list(LABELS)
    table = pd.crosstab(
        pd.Categorical(merged["label_a"], categories=labels),
        pd.Categorical(merged["label_b"], categories=labels),
        rownames=["a"],
        colnames=["b"],
        dropna=False,
    )
    return {
        "n_shared": int(len(merged)),
        "kappa": float(cohen_kappa_score(merged["label_a"], merged["label_b"], labels=labels)),
        "raw_agreement": float((merged["label_a"] == merged["label_b"]).mean()),
        "table": table,
    }


def agreement_files(path_a: str | Path, path_b: str | Path) -> dict:
    return agreement(pd.read_csv(path_a), pd.read_csv(path_b))
