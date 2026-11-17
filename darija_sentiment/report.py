"""Render reports/results.md and a confusion matrix PNG from a metrics.json file."""

import json
from pathlib import Path


def render_markdown(metrics: dict, image_name: str = "confusion_matrix.png") -> str:
    lines = [
        "# Results",
        "",
        f"Split: **{metrics.get('split', '?')}** ({metrics['n']} comments)",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Macro F1 | {metrics['macro_f1']:.3f} |",
        f"| Accuracy | {metrics['accuracy']:.3f} |",
        "",
        "## Per class",
        "",
        "| Label | Precision | Recall | F1 | Support |",
        "|-------|-----------|--------|----|---------|",
    ]
    for label, row in metrics["per_class"].items():
        lines.append(
            f"| {label} | {row['precision']:.3f} | {row['recall']:.3f} "
            f"| {row['f1']:.3f} | {row['support']} |"
        )
    lines += ["", f"![Confusion matrix]({image_name})", ""]
    for name, groups in metrics.get("breakdowns", {}).items():
        lines += [
            f"## By {name}",
            "",
            "| Group | n | Macro F1 | Accuracy |",
            "|-------|---|----------|----------|",
        ]
        for group, row in groups.items():
            flag = " (low support)" if row.get("low_support") else ""
            lines.append(
                f"| {group}{flag} | {row['n']} | {row['macro_f1']:.3f} | {row['accuracy']:.3f} |"
            )
        lines.append("")
    return "\n".join(lines)


def plot_confusion_matrix(metrics: dict, path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = metrics["confusion_matrix"]
    labels, matrix = cm["labels"], cm["matrix"]
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels=labels)
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Gold")
    top = max(max(row) for row in matrix) or 1
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            color = "white" if value > top / 2 else "black"
            ax.text(j, i, str(value), ha="center", va="center", color=color)
    ax.set_title(f"Macro F1 {metrics['macro_f1']:.3f}")
    fig.tight_layout()
    fig.savefig(path, dpi=120, metadata={"Software": None})
    plt.close(fig)


def write_report(metrics_path: str | Path, out_dir: str | Path) -> Path:
    metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(metrics, out_dir / "confusion_matrix.png")
    report = out_dir / "results.md"
    report.write_text(render_markdown(metrics), encoding="utf-8", newline="\n")
    return report
