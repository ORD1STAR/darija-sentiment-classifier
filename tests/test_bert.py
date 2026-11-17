"""Offline smoke test: fine-tune a tiny randomly initialised BERT on the seed set."""

import json

import pytest

transformers = pytest.importorskip("transformers")
pytest.importorskip("torch")

from darija_sentiment import LABELS  # noqa: E402
from darija_sentiment.bert import (  # noqa: E402
    ID2LABEL,
    LABEL2ID,
    BertConfig,
    BertPredictor,
    train_bert,
)
from darija_sentiment.data import DEFAULT_DATA, load_dataset  # noqa: E402


def tiny_tokenizer(tmp_path):
    """Character-level WordPiece vocab built from the seed data, no download needed."""
    chars = sorted(
        {c for text in load_dataset(DEFAULT_DATA)["clean"] for c in text if not c.isspace()}
    )
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *chars, *(f"##{c}" for c in chars)]
    path = tmp_path / "vocab.txt"
    path.write_text("\n".join(vocab), encoding="utf-8")
    return transformers.BertTokenizerFast(vocab_file=str(path), do_lower_case=False)


def tiny_model(vocab_size):
    config = transformers.BertConfig(
        vocab_size=vocab_size,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    return transformers.BertForSequenceClassification(config)


def test_default_config_matches_plan():
    c = BertConfig()
    assert c.model_name == "alger-ia/dziribert"
    assert (c.learning_rate, c.batch_size, c.epochs, c.max_length) == (2e-5, 16, 4, 128)
    assert (c.weight_decay, c.warmup_ratio) == (0.01, 0.1)


def test_tiny_bert_fine_tunes_offline(tmp_path):
    tokenizer = tiny_tokenizer(tmp_path)
    config = BertConfig(learning_rate=1e-3, epochs=2, max_length=48, batch_size=16)
    result = train_bert(
        DEFAULT_DATA,
        tmp_path / "run",
        config=config,
        tokenizer=tokenizer,
        model=tiny_model(len(tokenizer)),
    )
    assert 0.0 <= result["val_macro_f1"] <= 1.0
    assert result["best_checkpoint"] is not None
    saved = json.loads((tmp_path / "run" / "metrics.json").read_text(encoding="utf-8"))
    assert saved["config"]["epochs"] == 2

    prediction = BertPredictor(tmp_path / "run" / "best", max_length=48).predict("sa7a khoya")
    assert prediction["label"] in LABELS
    assert sum(prediction["scores"].values()) == pytest.approx(1.0, abs=1e-5)
