"""Run each model with several seeds and write a mean ± std comparison table."""

import importlib.util
import statistics
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory

from sklearn.metrics import f1_score

from darija_sentiment.baseline import build_pipeline
from darija_sentiment.data import load_dataset, split_three

MODELS = ("baseline", "bert")


def bert_available() -> bool:
    return all(importlib.util.find_spec(m) is not None for m in ("torch", "transformers"))


def run_baseline(data_path: str | Path, seed: int, split: str) -> float:
    parts = split_three(load_dataset(data_path), seed=seed)
    model = build_pipeline().fit(parts["train"]["clean"], parts["train"]["label"])
    preds = model.predict(parts[split]["clean"])
    return float(f1_score(parts[split]["label"], preds, average="macro"))


def run_bert(
    data_path: str | Path, seed: int, split: str, bert_kwargs: Callable[[], dict] | None = None
) -> float:
    from darija_sentiment.bert import BertConfig, BertPredictor, train_bert

    extra = bert_kwargs() if bert_kwargs else {}
    config = extra.pop("config", BertConfig())
    config.seed = seed
    parts = split_three(load_dataset(data_path), seed=seed)
    with TemporaryDirectory() as tmp:
        train_bert(data_path, tmp, config=config, **extra)
        predictor = BertPredictor(Path(tmp) / "best", max_length=config.max_length)
        preds = [predictor.predict(t)["label"] for t in parts[split]["text"]]
    return float(f1_score(parts[split]["label"], preds, average="macro"))


def compare(
    data_path: str | Path,
    models: tuple[str, ...] = MODELS,
    seeds: tuple[int, ...] = (13, 42, 2024),
    split: str = "test",
    bert_kwargs: Callable[[], dict] | None = None,
) -> dict:
    results, notes = {}, []
    for name in models:
        if name == "bert" and not bert_available():
            notes.append("bert skipped: torch/transformers are not installed")
            continue
        scores = []
        for seed in seeds:
            if name == "baseline":
                scores.append(run_baseline(data_path, seed, split))
            elif name == "bert":
                scores.append(run_bert(data_path, seed, split, bert_kwargs))
            else:
                raise ValueError(f"unknown model: {name}")
        results[name] = {
            "scores": scores,
            "mean": statistics.mean(scores),
            "std": statistics.stdev(scores) if len(scores) > 1 else 0.0,
        }
    return {"split": split, "seeds": list(seeds), "results": results, "notes": notes}


def render_comparison(summary: dict) -> str:
    seeds = ", ".join(map(str, summary["seeds"]))
    lines = [
        "# Model comparison",
        "",
        f"Macro F1 on the **{summary['split']}** split, seeds {seeds}.",
        "",
        "| Model | Macro F1 (mean ± std) | Per seed |",
        "|-------|----------------------|----------|",
    ]
    for name, row in summary["results"].items():
        per_seed = ", ".join(f"{s:.3f}" for s in row["scores"])
        lines.append(f"| {name} | {row['mean']:.3f} ± {row['std']:.3f} | {per_seed} |")
    if summary["notes"]:
        lines += ["", *(f"- {note}" for note in summary["notes"])]
    lines.append("")
    return "\n".join(lines)


def write_comparison(summary: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_comparison(summary), encoding="utf-8", newline="\n")
    return path
