# /// script
# requires-python = ">=3.10"
# dependencies = ["lightpanda"]
#
# # classify() is not on PyPI yet: use this checkout until it is.
# [tool.uv.sources]
# lightpanda = { path = "../..", editable = true }
# ///
"""Crawl the URL list and label each page with Jev, for training a local model.

Each row of the output JSONL holds what the browser's local model reads of
the page, which classify returns beside its answers when asked with
`includeInput`: the url,
title, status and semantic tree exactly as classify sends them to Jev, and
the page's HTML as the html tool renders it, both cut to the model's sizes.
Then come JUDGES `classify()` calls on that one load. train.py labels each
preset by a majority of the calls, which with two means both must agree, and
leaves it undecided otherwise. Each call also asks every preset in a second
wording (REWORDED). Those answers do not vote, but a page where the two
wordings disagree is one whose label is in doubt, for review.py.

Jev may favor questions by their position in the call, so the first call
asks all the questions in a random order and the second in the reverse one:
each is asked once early and once late. Each row records the orders.

Rows are appended as pages finish, and URLs already in the output are
skipped, so an interrupted crawl resumes where it stopped. A row with fewer
calls than JUDGES, from before the shuffled orders or from before
`includeInput` (marked by `input`), is dropped and crawled again, so all of a
page's votes come from the same load. A preset added to REWORDED later is asked of the
stored pages by extend.py, not by crawling them again.

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

# Jev labels each page this many times, for train.py's majority vote. Two
# is the fewest that ask each question both early and late in a call; on
# the first crawl a third would have settled at most 0.7% of the labels,
# the ones where only one call was confident.
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
    "is_age_gate": "Must the visitor say they are over 18, or give their date of birth, before the site shows its content?",
    "is_region_blocked": "Does the site refuse to show its content because of where the visitor is, such as their country?",
    "is_rate_limited": "Does the site refuse to show this page because the visitor made too many requests in a short time?",
}

def camel(name: str) -> str:
    head, *rest = name.split("_")
    return head + "".join(part.title() for part in rest)


def snake(name: str) -> str:
    return name if name.startswith("re_") else re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def complete(row: dict) -> bool:
    """Whether the row has every call, asked in the current shuffled form,
    on the local model's own input."""
    return len(row["jev"]) >= JUDGES and "jev_orders" in row and row.get("input") == "classify"


def questions_for(presets) -> dict:
    """Each preset asked by its camelCase name set to true, and again in its
    rewording, keyed `re_<preset>`."""
    return {camel(key): True for key in presets} | {f"re_{key}": REWORDED[key] for key in presets}


async def judge(page, questions: dict) -> tuple[dict, list, list, list]:
    """Asks the loaded page `questions` JUDGES times, in a random order and
    then reversed. Returns what the browser's local model reads of the page
    (url, title, status, tree and html, from the first call), each call's
    answers by snake_case key, the orders and each call's seconds."""
    order = random.sample(list(questions), len(questions))
    orders = [order if i % 2 == 0 else order[::-1] for i in range(JUDGES)]
    local, scores, seconds = None, [], []
    for i, keys in enumerate(orders):
        start = perf_counter()
        result = await page.call("classify", questions={key: questions[key] for key in keys},
                                 includeInput=True if i == 0 else None)
        if i == 0:
            local, result = result["input"], result["answers"]
        scores.append({snake(key): result.get(key) for key in keys})
        seconds.append(round(perf_counter() - start, 3))
    return local, scores, [[snake(key) for key in keys] for keys in orders], seconds


async def visit(page, url: str, timeout: int) -> dict:
    await page.goto(url=url, timeout=timeout)
    local, scores, orders, seconds = await judge(page, questions_for(REWORDED))
    return {
        "url": local["url"] or url,
        "title": local["title"] or None,
        "status": local["status"] or None,
        "tree": local["tree"],
        "html": local["html"],
        "input": "classify",
        "jev": scores,
        "jev_orders": orders,
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
