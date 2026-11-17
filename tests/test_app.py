import sys
from pathlib import Path

import pytest

gr = pytest.importorskip("gradio")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
import app  # noqa: E402


def test_examples_cover_each_script():
    from darija_sentiment.script import detect_script

    scripts = [detect_script(text) for (text,) in app.EXAMPLES]
    assert scripts.count("arabic") == 2
    assert scripts.count("latin") == 4  # 2 Arabizi + 2 Darija/French code-switching


def test_load_predictor_trains_when_model_missing(tmp_path):
    model = tmp_path / "missing" / "m.joblib"
    predictor = app.load_predictor(model)
    assert model.exists()
    assert predictor.predict("sa7a")["label"] in {"positive", "negative", "neutral"}


def test_build_demo(tmp_path):
    demo = app.build_demo(app.load_predictor(tmp_path / "m.joblib"))
    assert isinstance(demo, gr.Blocks)
