"""Export wrong predictions for manual review, plus a short markdown summary."""

from pathlib import Path

import joblib
import pandas as pd

from darija_sentiment.data import load_dataset, split_three
from darija_sentiment.evaluate import predict_split

ERROR_COLUMNS = ["id", "text", "script", "topic", "gold", "pred", "confidence"]


def wrong_predictions(table: pd.DataFrame) -> pd.DataFrame:
    """Rows where pred != gold, most confident mistakes first."""
    wrong = table[table["label"] != table["pred"]].rename(columns={"label": "gold"})
    wrong = wrong.sort_values("confidence", ascending=False, kind="stable")
    return wrong[[c for c in ERROR_COLUMNS if c in wrong.columns]].reset_index(drop=True)


def summarize(errors: pd.DataFrame, n_total: int) -> str:
    lines = [
        "# Error analysis",
        "",
        f"{len(errors)} wrong predictions out of {n_total}.",
        "",
        "## By gold -> predicted",
        "",
        "| gold | pred | count |",
        "|------|------|-------|",
    ]
    pairs = errors.groupby(["gold", "pred"]).size().sort_values(ascending=False)
    lines += [f"| {g} | {p} | {n} |" for (g, p), n in pairs.items()]
    if "script" in errors.columns:
        lines += ["", "## By script", "", "| script | count |", "|--------|-------|"]
        by_script = errors["script"].value_counts()
        lines += [f"| {s} | {n} |" for s, n in by_script.items()]
    lines += [
        "",
        "## Review categories",
        "",
        "Tag each row of `errors.csv` as one of: sarcasm, mixed language,",
        "ambiguous label, wrong label. Fix wrong labels and retrain once.",
        "",
    ]
    return "\n".join(lines)


def export_errors(
    model_path: str | Path,
    data_path: str | Path,
    out_dir: str | Path,
    split: str = "val",
    seed: int = 42,
) -> pd.DataFrame:
    model = joblib.load(model_path)
    table = predict_split(model, split_three(load_dataset(data_path), seed=seed)[split])
    errors = wrong_predictions(table)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    errors.to_csv(out_dir / "errors.csv", index=False, lineterminator="\n")
    (out_dir / "errors_summary.md").write_text(summarize(errors, len(table)), encoding="utf-8")
    return errors
