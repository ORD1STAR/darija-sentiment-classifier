"""Collect public YouTube comments with the Data API v3 into raw JSONL.

Only the comment text and public metadata are kept; author names and channel
ids are never stored. Raw files are append-only.
"""

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

API_URL = "https://www.googleapis.com/youtube/v3/commentThreads"
API_KEY_ENV = "YOUTUBE_API_KEY"


class QuotaExceeded(Exception):
    """Daily API quota is used up; collection should stop for today."""


class ApiError(Exception):
    def __init__(self, status: int, reason: str):
        super().__init__(f"HTTP {status}: {reason}")
        self.status = status
        self.reason = reason


def http_get_json(url: str) -> dict:
    """GET a URL and decode JSON, mapping API errors to exceptions."""
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        try:
            body = json.loads(err.read().decode("utf-8"))
            reason = body["error"]["errors"][0]["reason"]
        except (ValueError, KeyError, IndexError):
            reason = "unknown"
        raise ApiError(err.code, reason) from None


class YouTubeCollector:
    def __init__(
        self,
        api_key: str | None = None,
        get_json: Callable[[str], dict] = http_get_json,
        max_pages: int = 20,
    ):
        self.api_key = api_key or os.environ.get(API_KEY_ENV)
        if not self.api_key:
            raise RuntimeError(f"set the {API_KEY_ENV} environment variable")
        self.get_json = get_json
        self.max_pages = max_pages

    def _url(self, video_id: str, page_token: str | None) -> str:
        params = {
            "part": "snippet",
            "videoId": video_id,
            "maxResults": 100,
            "textFormat": "plainText",
            "order": "relevance",
            "key": self.api_key,
        }
        if page_token:
            params["pageToken"] = page_token
        return f"{API_URL}?{urllib.parse.urlencode(params)}"

    def video_comments(self, video_id: str) -> list[dict]:
        """All top-level comments of one video (up to `max_pages` pages)."""
        rows, token = [], None
        for _ in range(self.max_pages):
            try:
                data = self.get_json(self._url(video_id, token))
            except ApiError as err:
                if err.reason in ("quotaExceeded", "dailyLimitExceeded"):
                    raise QuotaExceeded(err.reason) from None
                raise
            for item in data.get("items", []):
                snippet = item["snippet"]["topLevelComment"]["snippet"]
                rows.append(
                    {
                        "comment_id": item["id"],
                        "video_id": video_id,
                        "text": snippet.get("textOriginal") or snippet.get("textDisplay", ""),
                        "published_at": snippet.get("publishedAt"),
                        "like_count": snippet.get("likeCount", 0),
                    }
                )
            token = data.get("nextPageToken")
            if not token:
                break
        return rows

    def collect(self, video_ids: list[str], out_path: str | Path) -> dict:
        """Append comments of each video to `out_path`; one failing video never stops the run."""
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        stats = {"videos_ok": 0, "videos_failed": 0, "comments": 0, "quota_exceeded": False}
        collected_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        with out_path.open("a", encoding="utf-8", newline="\n") as f:
            for video_id in video_ids:
                try:
                    rows = self.video_comments(video_id)
                except QuotaExceeded:
                    stats["quota_exceeded"] = True
                    break
                except ApiError:
                    # commentsDisabled, videoNotFound, ...
                    stats["videos_failed"] += 1
                    continue
                for row in rows:
                    record = {**row, "collected_at": collected_at}
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                stats["videos_ok"] += 1
                stats["comments"] += len(rows)
        return stats
