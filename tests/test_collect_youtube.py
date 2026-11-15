import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from darija_sentiment.collect_youtube import ApiError, YouTubeCollector

FIXTURES = Path(__file__).parent / "fixtures" / "youtube"


def load(name):
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def error_from(name):
    body = load(name)["error"]
    return ApiError(body["code"], body["errors"][0]["reason"])


class FakeApi:
    """Replays recorded responses keyed by (video id, page token)."""

    def __init__(self, responses):
        self.responses = responses
        self.urls = []

    def __call__(self, url):
        self.urls.append(url)
        q = parse_qs(urlparse(url).query)
        key = (q["videoId"][0], q.get("pageToken", [None])[0])
        response = self.responses[key]
        if isinstance(response, Exception):
            raise response
        return response


def test_requires_api_key(monkeypatch):
    monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="YOUTUBE_API_KEY"):
        YouTubeCollector()


def test_reads_key_from_env(monkeypatch):
    monkeypatch.setenv("YOUTUBE_API_KEY", "k-env")
    api = FakeApi({("v1", None): {"items": []}})
    YouTubeCollector(get_json=api).video_comments("v1")
    assert "key=k-env" in api.urls[0]


def test_pagination_and_no_author_data(tmp_path):
    api = FakeApi({("v1", None): load("page1"), ("v1", "P2"): load("page2")})
    out = tmp_path / "raw.jsonl"
    stats = YouTubeCollector("k", get_json=api).collect(["v1"], out)
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert stats["comments"] == 3 and len(rows) == 3
    assert [r["comment_id"] for r in rows] == ["c1", "c2", "c3"]
    assert rows[0]["text"] == "sa7a ftorkom 😍"
    assert all("author" not in json.dumps(r).lower() for r in rows)


def test_failed_video_is_skipped_and_quota_stops_run(tmp_path):
    api = FakeApi(
        {
            ("off", None): error_from("comments_disabled"),
            ("v2", None): load("page2"),
            ("v3", None): error_from("quota_error"),
        }
    )
    collector = YouTubeCollector("k", get_json=api)
    stats = collector.collect(["off", "v2", "v3", "v4"], tmp_path / "r.jsonl")
    assert stats == {"videos_ok": 1, "videos_failed": 1, "comments": 1, "quota_exceeded": True}
    assert not any("videoId=v4" in u for u in api.urls)


def test_raw_file_is_appended_not_overwritten(tmp_path):
    out = tmp_path / "r.jsonl"
    api = FakeApi({("v2", None): load("page2")})
    collector = YouTubeCollector("k", get_json=api)
    collector.collect(["v2"], out)
    collector.collect(["v2"], out)
    assert len(out.read_text(encoding="utf-8").splitlines()) == 2
