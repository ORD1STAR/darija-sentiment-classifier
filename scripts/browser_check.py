"""Headless browser check for the Gradio demo.

Starts app/app.py on a free port, opens it in headless Chromium, checks that the
page loads without console errors, runs one prediction and checks the result.
Exit code 0 on success.
"""

import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
LABELS = ("positive", "negative", "neutral")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for(url: str, timeout: float = 90.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2):
                return
        except OSError:
            time.sleep(0.5)
    raise TimeoutError(f"{url} did not answer within {timeout}s")


def run_check(url: str, screenshot: Path | None = None) -> list[str]:
    problems = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        errors = []
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(url, wait_until="networkidle")
        if "Darija Sentiment" not in page.content():
            problems.append("title text not found on the page")
        page.locator("#comment textarea").fill("machallah 3lik khoya, video zina bezzaf")
        page.locator("#classify").click()
        result = page.locator("#result")
        result.locator("text=/positive|negative|neutral/").first.wait_for(timeout=30000)
        text = result.inner_text().lower()
        if not any(label in text for label in LABELS):
            problems.append(f"no label in result: {text!r}")
        if screenshot:
            screenshot.parent.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(screenshot), full_page=True)
        browser.close()
    problems += [f"console error: {e}" for e in errors]
    return problems


def main() -> int:
    port = free_port()
    env = {**os.environ, "PORT": str(port), "GRADIO_ANALYTICS_ENABLED": "False"}
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "app" / "app.py")],
        cwd=ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        url = f"http://127.0.0.1:{port}/"
        wait_for(url)
        problems = run_check(url, ROOT / ".screenshots" / "app.png")
    finally:
        proc.terminate()
        proc.wait(timeout=15)
    for problem in problems:
        print(f"FAIL {problem}")
    print("browser check passed" if not problems else "browser check failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
