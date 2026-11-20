"""Gradio demo: type a Darija comment, get the probability of each sentiment."""

import os
import sys
from pathlib import Path

import gradio as gr

HERE = Path(__file__).resolve().parent
# repo layout: app/app.py next to ../darija_sentiment; Space bundle: everything flat
ROOT = HERE if (HERE / "darija_sentiment").is_dir() else HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from darija_sentiment.predict import Predictor  # noqa: E402

MODEL_PATH = Path(os.environ.get("DARIJA_MODEL", ROOT / "artifacts" / "baseline.joblib"))

EXAMPLES = [
    ["بارك الله فيك على هذا العمل الرائع"],
    ["الكونيكسيون ماكاش، خدمة خايبة"],
    ["machallah 3lik khoya, video zina bezzaf"],
    ["khdma ta3 walou, dhaya3tou wa9ti"],
    ["l'équipe nationale rahi top, ya3tik saha"],
    ["wa9tach ykhrej l'épisode jdid?"],
]

ABOUT = """
**What it does.** Classifies a comment written in Algerian Darija (Arabic letters,
Latin letters / Arabizi, or mixed with French) as positive, negative or neutral.

**Limits.** Trained on a small labeled set of public YouTube and Facebook comments.
Sarcasm, other dialects and very short comments are often misread.

**Data.** Public comments only; usernames, phone numbers, links and emails are
removed before anything is stored.
"""


def load_predictor(model_path: Path = MODEL_PATH) -> Predictor:
    """Load the saved model, training the baseline first if it is missing."""
    if not model_path.exists():
        from darija_sentiment.baseline import train
        from darija_sentiment.data import DEFAULT_DATA

        train(DEFAULT_DATA, model_path)
    return Predictor(model_path)


def build_demo(predictor: Predictor) -> gr.Blocks:
    def classify(text: str) -> dict[str, float]:
        if not text or not text.strip():
            return {}
        return predictor.predict(text)["scores"]

    # plain font stacks: the default theme requests font files that are not served
    sans = [gr.themes.Font(name) for name in ("Segoe UI", "Arial", "sans-serif")]
    mono = [gr.themes.Font(name) for name in ("Consolas", "monospace")]
    theme = gr.themes.Default(font=sans, font_mono=mono)
    with gr.Blocks(title="Darija Sentiment", theme=theme) as demo:
        gr.Markdown("# Darija Sentiment Classifier")
        with gr.Row():
            with gr.Column():
                text = gr.Textbox(label="Comment", lines=3, elem_id="comment")
                button = gr.Button("Classify", variant="primary", elem_id="classify")
            with gr.Column():
                output = gr.Label(label="Sentiment", num_top_classes=3, elem_id="result")
        gr.Examples(EXAMPLES, inputs=text)
        gr.Markdown(ABOUT)
        button.click(classify, inputs=text, outputs=output)
        text.submit(classify, inputs=text, outputs=output)
    return demo


def main() -> None:
    port = int(os.environ.get("PORT", "7860"))
    build_demo(load_predictor()).launch(server_name="127.0.0.1", server_port=port)


if __name__ == "__main__":
    main()
