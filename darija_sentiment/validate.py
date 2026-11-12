"""Schema validation for labeled CSV files."""

from pathlib import Path

import pandas as pd

from darija_sentiment import LABELS

SCHEMA = ("id", "text", "script", "source", "topic", "label", "notes")
SCRIPTS = ("arabic", "latin", "mixed", "unknown")


def validate_frame(df: pd.DataFrame, allow_unlabeled: bool = False) -> list[str]:
    """Return a list of human-readable problems; empty means the file is valid."""
    problems = []
    missing = [c for c in SCHEMA if c not in df.columns]
    if missing:
        problems.append(f"missing columns: {', '.join(missing)}")
    extra = [c for c in df.columns if c not in SCHEMA]
    if extra:
        problems.append(f"unexpected columns: {', '.join(extra)}")
    if "id" in df.columns:
        dup = df.loc[df["id"].duplicated(keep=False), "id"].unique().tolist()
        if dup:
            problems.append(f"duplicate ids: {', '.join(map(str, dup[:10]))}")
    if "text" in df.columns:
        empty = df["text"].isna() | (df["text"].astype(str).str.strip() == "")
        if empty.any():
            rows = df.loc[empty, "id"] if "id" in df.columns else df.index[empty]
            problems.append(f"empty text for ids: {', '.join(map(str, list(rows)[:10]))}")
    if "label" in df.columns:
        labels = df["label"].dropna() if allow_unlabeled else df["label"].fillna("<empty>")
        bad = sorted(set(labels) - set(LABELS))
        if bad:
            problems.append(f"invalid labels: {', '.join(map(str, bad))}")
    if "script" in df.columns:
        bad = sorted(set(df["script"].dropna()) - set(SCRIPTS))
        if bad:
            problems.append(f"invalid scripts: {', '.join(map(str, bad))}")
    return problems


def validate_file(path: str | Path, allow_unlabeled: bool = False) -> list[str]:
    return validate_frame(pd.read_csv(path, dtype={"notes": str}), allow_unlabeled)
