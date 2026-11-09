"""TF-IDF + Logistic Regression baseline.

Character n-grams handle the non-standard spelling of Darija far better than
word tokens, so they are the main feature block.
"""

from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

from darija_sentiment.data import load_dataset, split_three


def build_pipeline(c: float = 4.0) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), lowercase=True),
            ),
            (
                "clf",
                LogisticRegression(C=c, class_weight="balanced", max_iter=2000),
            ),
        ]
    )


def train(data_path: str | Path, model_path: str | Path, seed: int = 42) -> dict:
    """Train the baseline on the train split, save it and return validation metrics.

    The test split is left untouched for the final evaluation.
    """
    splits = split_three(load_dataset(data_path), seed=seed)
    train_df, val_df = splits["train"], splits["val"]
    model = build_pipeline()
    model.fit(train_df["clean"], train_df["label"])
    preds = model.predict(val_df["clean"])
    metrics = {
        "n_train": len(train_df),
        "n_val": len(val_df),
        "val_accuracy": float(accuracy_score(val_df["label"], preds)),
        "val_macro_f1": float(f1_score(val_df["label"], preds, average="macro")),
    }
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return metrics
