import pandas as pd

from darija_sentiment.cli import main
from darija_sentiment.data import DEFAULT_DATA, load_dataset, split_three


def test_split_three_proportions_per_class():
    df = load_dataset(DEFAULT_DATA)
    parts = split_three(df, seed=7)
    assert sum(len(p) for p in parts.values()) == len(df)
    total = df["label"].value_counts()
    for name, share in (("train", 0.70), ("val", 0.15), ("test", 0.15)):
        counts = parts[name]["label"].value_counts()
        for label, n in total.items():
            assert abs(counts.get(label, 0) - n * share) <= 1.0 + 1e-9, (name, label)


def test_splits_do_not_overlap():
    parts = split_three(load_dataset(DEFAULT_DATA))
    ids = [set(p["id"]) for p in parts.values()]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])


def test_split_cli_is_reproducible(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    assert main(["split", "--out", str(a)]) == 0
    assert main(["split", "--out", str(b)]) == 0
    for name in ("train", "val", "test"):
        assert (a / f"{name}.csv").read_bytes() == (b / f"{name}.csv").read_bytes()
        assert len(pd.read_csv(a / f"{name}.csv")) > 0
