import json

from sklearn.model_selection import GridSearchCV

from darija_sentiment.cli import main
from darija_sentiment.data import DEFAULT_DATA, load_dataset, split_three
from darija_sentiment.tune import PARAM_GRID, load_params, tune


def test_tune_uses_train_split_only(tmp_path, monkeypatch):
    seen = []
    original_fit = GridSearchCV.fit

    def spy_fit(self, x, y=None, **kw):
        seen.append(list(x))
        return original_fit(self, x, y, **kw)

    monkeypatch.setattr(GridSearchCV, "fit", spy_fit)
    result = tune(DEFAULT_DATA, tmp_path / "p.json", folds=3)

    splits = split_three(load_dataset(DEFAULT_DATA))
    assert sorted(seen[0]) == sorted(splits["train"]["clean"])
    assert not set(seen[0]) & (set(splits["test"]["clean"]) - set(splits["train"]["clean"]))
    assert result["c"] in PARAM_GRID["clf__C"]
    assert result["n_train"] == len(splits["train"])


def test_params_are_reused_by_train(tmp_path):
    params = tmp_path / "p.json"
    assert main(["tune", "--out", str(params), "--folds", "3"]) == 0
    saved = json.loads(params.read_text(encoding="utf-8"))
    assert load_params(params) == {"c": saved["c"], "class_weight": saved["class_weight"]}
    assert main(["train", "--model", str(tmp_path / "m.joblib"), "--params", str(params)]) == 0
