import pandas as pd
import pytest

from darija_sentiment.agreement import agreement
from darija_sentiment.cli import main


def frame(ids, labels):
    return pd.DataFrame({"id": ids, "label": labels})


def test_perfect_agreement():
    a = frame([1, 2, 3], ["positive", "negative", "neutral"])
    result = agreement(a, a)
    assert result["kappa"] == pytest.approx(1.0)
    assert result["n_shared"] == 3


def test_only_shared_ids_are_compared():
    a = frame([1, 2, 3, 4], ["positive", "negative", "neutral", "positive"])
    b = frame([2, 3, 4, 9], ["negative", "neutral", "negative", "positive"])
    result = agreement(a, b)
    assert result["n_shared"] == 3
    assert result["raw_agreement"] == pytest.approx(2 / 3)
    assert int(result["table"].values.sum()) == 3


def test_no_shared_ids_is_an_error():
    with pytest.raises(ValueError):
        agreement(frame([1], ["positive"]), frame([2], ["positive"]))


def test_cli_warns_on_low_kappa(tmp_path, capsys):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    frame([1, 2, 3, 4], ["positive", "negative", "neutral", "positive"]).to_csv(a, index=False)
    frame([1, 2, 3, 4], ["negative", "positive", "positive", "neutral"]).to_csv(b, index=False)
    assert main(["agreement", str(a), str(b)]) == 0
    out = capsys.readouterr().out
    assert "shared ids: 4" in out
    assert "warning: kappa below 0.6" in out
