"""Hugging Face publishing kit: Space bundle, model card, dataset card, clean dataset export."""

import json
import shutil
from pathlib import Path

import pandas as pd

from darija_sentiment.data import load_dataset

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = Path(__file__).resolve().parent

SPACE_HEADER = """---
title: Darija Sentiment
emoji: 💬
colorFrom: green
colorTo: red
sdk: gradio
app_file: app.py
pinned: false
---
"""


def export_space(out_dir: str | Path, model_path: str | Path) -> Path:
    """Build a self-contained Gradio Space folder: app.py, package, model, requirements."""
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "artifacts").mkdir(parents=True)
    shutil.copy2(ROOT / "app" / "app.py", out_dir / "app.py")
    shutil.copy2(ROOT / "app" / "requirements.txt", out_dir / "requirements.txt")
    shutil.copytree(
        PACKAGE, out_dir / "darija_sentiment", ignore=shutil.ignore_patterns("__pycache__")
    )
    shutil.copy2(model_path, out_dir / "artifacts" / "baseline.joblib")
    (out_dir / "README.md").write_text(
        SPACE_HEADER + "\nDemo of the Darija sentiment classifier.\n", encoding="utf-8"
    )
    return out_dir


def export_dataset(data_path: str | Path, out_path: str | Path) -> int:
    """Publishable dataset: cleaned text and label only, no ids, sources or notes."""
    df = load_dataset(data_path)
    out = pd.DataFrame({"text": df["clean"], "label": df["label"]})
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, lineterminator="\n")
    return len(out)


def model_card(metrics: dict, repo_id: str = "your-name/darija-sentiment-dziribert") -> str:
    per_class = "\n".join(
        f"| {label} | {row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} |"
        for label, row in metrics["per_class"].items()
    )
    return f"""---
language: [ar]
license: mit
pipeline_tag: text-classification
tags: [darija, algerian-arabic, arabizi, sentiment]
---

# {repo_id}

Sentiment classifier (positive / negative / neutral) for Algerian Darija written
in Arabic script, Latin script (Arabizi) or mixed with French.

## Results ({metrics.get("split", "test")} split, {metrics["n"]} comments)

| Metric | Value |
|--------|-------|
| Macro F1 | {metrics["macro_f1"]:.3f} |
| Accuracy | {metrics["accuracy"]:.3f} |

| Label | Precision | Recall | F1 |
|-------|-----------|--------|----|
{per_class}

## Limits

- Trained on a small set of public YouTube and Facebook comments.
- Sarcasm, other Maghrebi dialects and one-word comments are often misread.
- Not meant for moderation decisions about individuals.
"""


def dataset_card(n_rows: int, label_counts: dict[str, int], guide: str) -> str:
    counts = "\n".join(f"| {label} | {n} |" for label, n in sorted(label_counts.items()))
    return f"""---
language: [ar]
license: cc-by-4.0
task_categories: [text-classification]
---

# Algerian Darija sentiment dataset

{n_rows} public comments in Algerian Darija, each labeled positive, negative or
neutral. Only the cleaned text and the label are published.

| Label | Count |
|-------|-------|
{counts}

## Collection

Public comments from YouTube (Data API v3) and a smaller hand-copied Facebook
sample. Usernames, phone numbers, links and emails were removed before storage.
Spam, ads and duplicates were dropped.

{guide.replace("# Labeling guide", "## Labeling guide", 1)}
"""


def export_cards(
    data_path: str | Path, metrics_path: str | Path, out_dir: str | Path
) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    n = export_dataset(data_path, out_dir / "dataset.csv")
    counts = load_dataset(data_path)["label"].value_counts().to_dict()
    guide = (ROOT / "docs" / "labeling_guide.md").read_text(encoding="utf-8")
    paths = {
        "model_card": out_dir / "MODEL_CARD.md",
        "dataset_card": out_dir / "DATASET_CARD.md",
        "dataset": out_dir / "dataset.csv",
    }
    paths["model_card"].write_text(model_card(metrics), encoding="utf-8", newline="\n")
    paths["dataset_card"].write_text(dataset_card(n, counts, guide), encoding="utf-8", newline="\n")
    return paths
