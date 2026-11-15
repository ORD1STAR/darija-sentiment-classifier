"""Turn raw collected JSONL into an anonymized, deduped CSV ready for labeling."""

import json
import re
from pathlib import Path

import pandas as pd

from darija_sentiment.clean import scrub_personal_data
from darija_sentiment.dedupe import dedupe_and_filter
from darija_sentiment.script import detect_script
from darija_sentiment.validate import SCHEMA

_MENTION_RE = re.compile(r"@[\w.-]+")
_SPACES_RE = re.compile(r"\s+")


def anonymize(text: str) -> str:
    """Remove links, emails, phone numbers and @mentions; keep the wording otherwise."""
    text = _MENTION_RE.sub(" ", scrub_personal_data(text or ""))
    return _SPACES_RE.sub(" ", text).strip()


def read_raw(paths: list[str | Path]) -> pd.DataFrame:
    """Read raw JSONL files without modifying them."""
    records = []
    for path in paths:
        with Path(path).open(encoding="utf-8") as f:
            records.extend(json.loads(line) for line in f if line.strip())
    return pd.DataFrame(records)


def build_dataset(
    raw_paths: list[str | Path],
    out_path: str | Path,
    source: str = "youtube",
    topics: dict[str, str] | None = None,
) -> dict[str, int]:
    """Write a labeling-schema CSV with an empty `label` column. Returns the removal report."""
    raw = read_raw(raw_paths)
    if raw.empty:
        raise ValueError("no raw records found")
    if "comment_id" in raw.columns:
        raw = raw.drop_duplicates(subset="comment_id", keep="last")
    topics = topics or {}
    frame = pd.DataFrame(
        {
            "text": raw["text"].map(anonymize),
            "topic": raw.get("video_id", pd.Series([""] * len(raw))).map(
                lambda v: topics.get(v, "")
            ),
        }
    )
    kept, report = dedupe_and_filter(frame)
    out = pd.DataFrame(
        {
            "id": range(1, len(kept) + 1),
            "text": kept["text"],
            "script": kept["text"].map(detect_script),
            "source": source,
            "topic": kept["topic"],
            "label": "",
            "notes": "",
        }
    )[list(SCHEMA)]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, lineterminator="\n")
    return {"raw": int(len(raw)), **report}
