import pytest

from darija_sentiment import compare as compare_mod
from darija_sentiment.cli import main
from darija_sentiment.compare import compare, render_comparison
from darija_sentiment.data import DEFAULT_DATA


def test_baseline_multi_seed():
    summary = compare(DEFAULT_DATA, models=("baseline",), seeds=(1, 2, 3))
    row = summary["results"]["baseline"]
    assert len(row["scores"]) == 3
    assert min(row["scores"]) <= row["mean"] <= max(row["scores"])
    assert row["std"] >= 0


def test_bert_skipped_without_transformers(monkeypatch):
    monkeypatch.setattr(compare_mod, "bert_available", lambda: False)
    summary = compare(DEFAULT_DATA, seeds=(1, 2))
    assert list(summary["results"]) == ["baseline"]
    assert "bert skipped" in summary["notes"][0]
    assert "- bert skipped" in render_comparison(summary)


def test_bert_runs_with_tiny_model(tmp_path):
    pytest.importorskip("transformers")
    from darija_sentiment.bert import BertConfig
    from tests.test_bert import tiny_model, tiny_tokenizer

    def kwargs():
        tok = tiny_tokenizer(tmp_path)
        return {
            "config": BertConfig(learning_rate=1e-3, epochs=1, max_length=48),
            "tokenizer": tok,
            "model": tiny_model(len(tok)),
        }

    summary = compare(DEFAULT_DATA, models=("bert",), seeds=(1, 2), bert_kwargs=kwargs)
    assert len(summary["results"]["bert"]["scores"]) == 2


def test_compare_cli_writes_markdown(tmp_path, monkeypatch):
    monkeypatch.setattr(compare_mod, "bert_available", lambda: False)
    out = tmp_path / "comparison.md"
    assert main(["compare", "--seeds", "1", "2", "--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "| baseline |" in text and "±" in text
