"""Dataset loading and splitting."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from darija_sentiment import LABELS
from darija_sentiment.clean import clean_text
from darija_sentiment.script import detect_script

DEFAULT_DATA = Path(__file__).resolve().parent.parent / "data" / "processed" / "sample.csv"
REQUIRED_COLUMNS = ("text", "label")


def load_dataset(path: str | Path = DEFAULT_DATA) -> pd.DataFrame:
    """Load a labeled CSV, validate it and add a cleaned text column."""
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"dataset is missing columns: {missing}")
    unknown = set(df["label"]) - set(LABELS)
    if unknown:
        raise ValueError(f"unknown labels in dataset: {sorted(unknown)}")
    df = df.dropna(subset=["text"]).copy()
    df["clean"] = df["text"].map(clean_text)
    if "script" not in df.columns:
        df["script"] = df["clean"].map(detect_script)
    return df[df["clean"] != ""].reset_index(drop=True)


def split_dataset(
    df: pd.DataFrame, test_size: float = 0.2, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split with a fixed seed."""
    train, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=df["label"])
    return train.reset_index(drop=True), test.reset_index(drop=True)


SPLIT_NAMES = ("train", "val", "test")


def split_three(
    df: pd.DataFrame, val_size: float = 0.15, test_size: float = 0.15, seed: int = 42
) -> dict[str, pd.DataFrame]:
    """Stratified train/validation/test split (70/15/15 by default)."""
    rest, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=df["label"])
    train, val = train_test_split(
        rest,
        test_size=val_size / (1 - test_size),
        random_state=seed,
        stratify=rest["label"],
    )
    return {
        name: part.reset_index(drop=True)
        for name, part in zip(SPLIT_NAMES, (train, val, test), strict=True)
    }


def write_splits(data_path: str | Path, out_dir: str | Path, seed: int = 42) -> dict[str, int]:
    """Split a labeled CSV once and write train.csv, val.csv and test.csv."""
    df = load_dataset(data_path).drop(columns=["clean"])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sizes = {}
    for name, part in split_three(df, seed=seed).items():
        part.to_csv(out_dir / f"{name}.csv", index=False, lineterminator="\n")
        sizes[name] = len(part)
    return sizes


def load_split(splits_dir: str | Path, name: str) -> pd.DataFrame:
    return load_dataset(Path(splits_dir) / f"{name}.csv")
