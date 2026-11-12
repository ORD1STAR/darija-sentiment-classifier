"""Cross-validated search over C and class weights, on the train split only."""

import json
from pathlib import Path

from sklearn.model_selection import GridSearchCV, StratifiedKFold

from darija_sentiment.baseline import build_pipeline
from darija_sentiment.data import load_dataset, split_three

PARAM_GRID = {
    "clf__C": [0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
    "clf__class_weight": [None, "balanced"],
}


def tune(data_path: str | Path, out_path: str | Path, seed: int = 42, folds: int = 5) -> dict:
    """Run stratified k-fold CV on the train split and save the best params as JSON."""
    train_df = split_three(load_dataset(data_path), seed=seed)["train"]
    folds = min(folds, int(train_df["label"].value_counts().min()))
    search = GridSearchCV(
        build_pipeline(),
        PARAM_GRID,
        scoring="f1_macro",
        cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed),
        n_jobs=1,
    )
    search.fit(train_df["clean"], train_df["label"])
    result = {
        "c": search.best_params_["clf__C"],
        "class_weight": search.best_params_["clf__class_weight"],
        "cv_macro_f1": float(search.best_score_),
        "folds": folds,
        "n_train": len(train_df),
        "seed": seed,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def load_params(path: str | Path) -> dict:
    """Return the pipeline keyword arguments stored by `tune`."""
    saved = json.loads(Path(path).read_text(encoding="utf-8"))
    return {"c": saved["c"], "class_weight": saved["class_weight"]}
