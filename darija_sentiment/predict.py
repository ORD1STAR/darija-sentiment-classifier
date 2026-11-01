"""Inference helpers."""

from pathlib import Path

import joblib

from darija_sentiment.clean import clean_text


class Predictor:
    def __init__(self, model_path: str | Path):
        self.model = joblib.load(model_path)

    def predict(self, text: str) -> dict:
        """Return the predicted label and the probability of each class."""
        probs = self.model.predict_proba([clean_text(text)])[0]
        scores = {label: float(p) for label, p in zip(self.model.classes_, probs, strict=True)}
        return {"label": max(scores, key=scores.get), "scores": scores}
