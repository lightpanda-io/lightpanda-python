# /// script
# requires-python = ">=3.10"
# dependencies = ["lightpanda"]
#
# # classify() is not on PyPI yet: use this checkout until it is.
# [tool.uv.sources]
# lightpanda = { path = "../..", editable = true }
# ///
"""Crawl the URL list and label each page with Jev, for training a local model.

Each row of the output JSONL holds what Jev sees (url, title, status and the
semantic tree, as the browser's classify sends it), the page's HTML for
feature extraction, and JUDGES `classify()` calls on that one load. Each call
also asks every preset in a second wording (REWORDED), so a preset gets two
votes per call. Jev rarely changes its answer to the same question, but a
page where two wordings disagree is one whose label is in doubt. train.py
labels each preset by a majority of the votes and drops the rest.

Jev may favor questions by their position in the call, so the first call
asks the 18 questions in a random order and the second in the reverse one:
each is asked once early and once late. Each row records the orders.

Rows are appended as pages finish, and URLs already in the output are
skipped, so an interrupted crawl resumes where it stopped. A row with fewer
calls than JUDGES, or from before the shuffled orders, is dropped and crawled again, so all of a page's votes
come from the same load.

Needs TYPESAFE_API_KEY. Run:
    uv run research/page_classifier/collect.py --limit 50
"""

import argparse
import asyncio
import json
import random
import re
import sys
from pathlib import Path
from time import perf_counter

from lightpanda import AsyncBrowser, LightpandaError

HERE = Path(__file__).parent
DATA = HERE / "data"

# The browser sends Jev at most this much of the tree (classify.zig's
# max_state_bytes).
MAX_TREE_BYTES = 8192
# Enough HTML for the scripts and banners the features look for.
MAX_HTML_BYTES = 256 * 1024
# Jev labels each page this many times, for train.py's majority vote.
JUDGES = 2

# Each preset asked again in other words, keyed by the preset it shadows.
REWORDED = {
    "is_blocked": "Has the site refused to serve this visitor, with a bot check, security challenge or access-denied notice instead of the page?",
    "is_captcha": "Must the visitor solve a puzzle, tick an 'I am not a robot' box or otherwise prove they are human before seeing the page?",
    "is_consent_wall": "Does a privacy or cookie notice asking the visitor to accept or reject tracking cover or block the page?",
    "is_empty_catalog": "Is this a search, category or filtered listing that reports zero matching items?",
    "is_error_page": "Is this a 404, 'page not found', 'something went wrong' or similar failure page rather than the page that was asked for?",
    "is_login_wall": "Must the visitor sign in or create an account before they can see what this page is for?",
    "is_paywall": "Is the article or content locked or truncated until the visitor pays or subscribes?",
    "is_unsupported_browser": "Does the page tell the visitor to enable JavaScript, switch browsers or upgrade their browser before it will work?",
    "is_loading": "Is the page's main area still empty except for a spinner, skeleton boxes or a 'loading' message?",
}

# The tree tool prints each node's backendNodeId first; classify's tree has none.
NODE_ID = re.compile(r"^( *)\d+ ", re.MULTILINE)


def camel(name: str) -> str:
    head, *rest = name.split("_")
    return head + "".join(part.title() for part in rest)


def snake(name: str) -> str:
    return name if name.startswith("re_") else re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def complete(row: dict) -> bool:
    """Whether the row has every call, asked in the current shuffled form."""
    return len(row["jev"]) >= JUDGES and "jev_orders" in row


def cap(text: str, limit: int) -> str:
    return text.encode()[:limit].decode(errors="ignore")


async def visit(page, url: str, timeout: int) -> dict:
    loaded = await page.goto(url=url, timeout=timeout)
    tree = NODE_ID.sub(r"\1", await page.tree())
    html = await page.html()
    # A preset is asked by its camelCase name set to true, a rewording as text.
    questions = {camel(key): True for key in REWORDED} | {f"re_{key}": q for key, q in REWORDED.items()}
    order = random.sample(list(questions), len(questions))
    orders = [order if i % 2 == 0 else order[::-1] for i in range(JUDGES)]
    scores, seconds = [], []
    for keys in orders:
        start = perf_counter()
        answers = await page.classify(questions={key: questions[key] for key in keys})
        scores.append({snake(key): answers.get(key) for key in keys})
        seconds.append(round(perf_counter() - start, 3))
    return {
        "url": loaded.url or url,
        "title": loaded.title,
        "status": loaded.http_status,
        "tree": cap(tree, MAX_TREE_BYTES),
        "html": cap(html, MAX_HTML_BYTES),
        "jev": scores,
        "jev_orders": [[snake(key) for key in keys] for keys in orders],
        "jev_seconds": seconds,
    }


async def crawl(browser, slots, out, source: str, url: str, timeout: int, counts: dict):
    async with slots:
        try:
            async with await browser.new_session() as page:
                row = await asyncio.wait_for(visit(page, url, timeout), timeout / 1000 * 4)
        except (LightpandaError, asyncio.TimeoutError) as err:
            counts["failed"] += 1
            row = {"error": str(err) or type(err).__name__}
        else:
            counts["ok"] += 1
        row = {"seed": url, "source": source, **row}
        out.write(json.dumps(row) + "\n")
        out.flush()
        done = counts["ok"] + counts["failed"]
        if sys.stderr.isatty():
            print(f"\r{done}/{counts['total']}  ok {counts['ok']}  failed {counts['failed']}", end="", file=sys.stderr)


async def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--urls", type=Path, default=DATA / "urls.tsv")
    parser.add_argument("--out", type=Path, default=DATA / "pages.jsonl")
    parser.add_argument("--limit", type=int, help="crawl at most this many new urls")
    parser.add_argument("--crawlers", type=int, default=6, help="pages loading at once")
    parser.add_argument("--timeout", type=int, default=15000, help="page load timeout, ms")
    args = parser.parse_args()

    seen = set()
    if args.out.exists():
        rows = [json.loads(line) for line in args.out.open()]
        rows = [r for r in rows if "error" in r or complete(r)]
        args.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
        seen = {r["seed"] for r in rows}
    todo = []
    for line in args.urls.read_text().splitlines():
        source, url = line.split("\t")
        if url not in seen:
            todo.append((source, url))
    todo = todo[: args.limit]

    counts = {"ok": 0, "failed": 0, "total": len(todo)}
    slots = asyncio.Semaphore(args.crawlers)
    start = perf_counter()
    with args.out.open("a") as out:
        async with AsyncBrowser(max_concurrency=args.crawlers * 2) as browser:
            await asyncio.gather(
                *(crawl(browser, slots, out, source, url, args.timeout, counts) for source, url in todo)
            )
    print(f"\n{counts['ok']} labelled, {counts['failed']} failed in {perf_counter() - start:.0f}s -> {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
