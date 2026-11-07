import pandas as pd
import pytest

from darija_sentiment import LABELS
from darija_sentiment.baseline import train
from darija_sentiment.cli import main
from darija_sentiment.data import DEFAULT_DATA, load_dataset, split_dataset
from darija_sentiment.predict import Predictor


def test_sample_dataset_is_valid_and_balanced():
    df = load_dataset(DEFAULT_DATA)
    counts = df["label"].value_counts()
    assert set(counts.index) == set(LABELS)
    assert counts.min() >= 20


def test_split_is_stratified_and_reproducible():
    df = load_dataset(DEFAULT_DATA)
    a_train, a_test = split_dataset(df, seed=1)
    b_train, b_test = split_dataset(df, seed=1)
    assert list(a_test["id"]) == list(b_test["id"])
    assert set(a_test["label"]) == set(LABELS)
    assert len(a_train) + len(a_test) == len(df)


def test_load_rejects_unknown_labels(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"text": ["a"], "label": ["happy"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="unknown labels"):
        load_dataset(path)


def test_train_and_predict_end_to_end(tmp_path):
    model_path = tmp_path / "model.joblib"
    metrics = train(DEFAULT_DATA, model_path)
    assert model_path.exists()
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    result = Predictor(model_path).predict("barakallah fik, khdma mlih bezzaf")
    assert result["label"] in LABELS
    assert sum(result["scores"].values()) == pytest.approx(1.0)


def test_cli_train_then_predict(tmp_path, capsys):
    model_path = str(tmp_path / "m.joblib")
    assert main(["train", "--model", model_path]) == 0
    assert main(["predict", "chhal soum?", "--model", model_path]) == 0
    assert '"label"' in capsys.readouterr().out


def test_load_fills_missing_script_column(tmp_path):
    path = tmp_path / "noscript.csv"
    pd.DataFrame({"text": ["فيديو مليح", "video mlih"], "label": ["positive"] * 2}).to_csv(
        path, index=False
    )
    df = load_dataset(path)
    assert list(df["script"]) == ["arabic", "latin"]
