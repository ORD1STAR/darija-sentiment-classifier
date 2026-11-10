import pytest

from darija_sentiment.baseline import build_pipeline, train
from darija_sentiment.cli import main
from darija_sentiment.data import DEFAULT_DATA


def test_pipeline_uses_both_blocks_by_default():
    names = [name for name, _ in build_pipeline().named_steps["features"].transformer_list]
    assert names == ["char", "word"]


@pytest.mark.parametrize(
    ("char", "word", "expected"), [(True, False, ["char"]), (False, True, ["word"])]
)
def test_blocks_can_be_turned_off(char, word, expected):
    pipe = build_pipeline(char_ngrams=char, word_ngrams=word)
    assert [n for n, _ in pipe.named_steps["features"].transformer_list] == expected


def test_no_block_is_an_error():
    with pytest.raises(ValueError):
        build_pipeline(char_ngrams=False, word_ngrams=False)


@pytest.mark.parametrize("flag", ["--no-char", "--no-word"])
def test_train_cli_with_one_block(tmp_path, flag):
    assert main(["train", "--model", str(tmp_path / "m.joblib"), flag]) == 0


def test_word_only_model_trains(tmp_path):
    metrics = train(DEFAULT_DATA, tmp_path / "m.joblib", char_ngrams=False)
    assert 0.0 <= metrics["val_macro_f1"] <= 1.0
