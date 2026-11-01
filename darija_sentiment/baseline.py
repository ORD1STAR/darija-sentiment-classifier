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

from darija_sentiment.data import load_dataset, split_dataset


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
    """Train the baseline, save it and return test metrics."""
    df = load_dataset(data_path)
    train_df, test_df = split_dataset(df, seed=seed)
    model = build_pipeline()
    model.fit(train_df["clean"], train_df["label"])
    preds = model.predict(test_df["clean"])
    metrics = {
        "n_train": len(train_df),
        "n_test": len(test_df),
        "accuracy": float(accuracy_score(test_df["label"], preds)),
        "macro_f1": float(f1_score(test_df["label"], preds, average="macro")),
    }
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return metrics
