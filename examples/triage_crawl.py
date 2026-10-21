# /// script
# requires-python = ">=3.10"
# dependencies = ["lightpanda"]
#
# # judge() is not on PyPI yet: use this checkout until it is.
# [tool.uv.sources]
# lightpanda = { path = "..", editable = true }
# ///
"""Triage a crawl of real sites in parallel, and act on what Jev sees.

Each page gets one ``page.judge()`` call: every wall check plus our own
questions. The verdict decides what happens next: back off when the site
says we are going too fast, accept a cookie wall and look again, wait for a
page that is still loading, or skip the page and say why.

With --model, only the presets are asked, and the browser answers the ones
its local model is sure of (a model.lpcm exported by ../page-classifier)
and asks Jev only the rest, so run it with and without to compare. Add
--local to leave Jev out altogether: a page the model is unsure of is
reported as one that would need Jev.

Needs TYPESAFE_API_KEY, except with --local.
Run:  uv run examples/triage_crawl.py [--model PATH [--local]]
"""

import argparse
import asyncio
import os
import statistics
import sys
from collections import Counter
from time import perf_counter

from lightpanda import AsyncBrowser, LightpandaError, ToolError

# Pages loading at once, each in its own session of one browser.
CRAWLERS = 6

URLS = [
    # content
    "https://books.toscrape.com/catalogue/a-light-in-the-attic_1000/index.html",
    "https://quotes.toscrape.com/js/",
    "https://en.wikipedia.org/wiki/Headless_browser",
    "https://en.wikipedia.org/wiki/Web_scraping",
    "https://docs.python.org/3/library/asyncio.html",
    "https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API",
    "https://github.com/lightpanda-io/browser",
    "https://lobste.rs/",
    "https://www.spiegel.de/",
    "https://news.ycombinator.com/",
    "https://news.ycombinator.com/newest",
    # a cookie wall in front of the content
    "https://www.elmundo.es/",
    "https://www.20minutes.fr/",
    # walls
    "https://news.ycombinator.com/login",
    "https://github.com/settings/profile",
    "https://www.linkedin.com/feed/",
    "https://www.g2.com/",
    "https://www.wsj.com/",
    "https://www.economist.com/leaders/2024/01/04/what-to-expect-in-2024",
    "https://pypi.org/project/lightpanda/",
    "https://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do-in-python",
    "https://www.zeit.de/index",
    "https://web.whatsapp.com/",
    # dead ends
    "https://www.npmjs.com/search?q=xqzzzvqkplorbt",
    "https://github.com/search?q=xqzzzvqkplorbt&type=repositories",
    "https://en.wikipedia.org/w/index.php?search=xqzzzvqk+plorbt&fulltext=1",
    "https://www.python.org/no-such-page",
    "https://github.com/lightpanda-io/does-not-exist",
    "https://arxiv.org/abs/1706.03762",
    "https://www.ebay.com/sch/i.html?_nkw=xqzzzvqkplorbt",
]

QUESTIONS = {
    "kind": {
        "question": "What kind of page is this?",
        "options": ["product", "listing", "article", "docs", "search", "other"],
    },
}

# Accept buttons start with one of these, in the languages we crawl.
ACCEPT = "/^(accept|aceptar|accepter|akzeptieren|agree|allow)/i"

# A call this quick never reached Jev: the local model answered it all.
LOCAL_SECONDS = 0.1

TTY = sys.stdout.isatty()
call_times = []
presets_only = False  # --model: Jev answers only what the model is unsure of
local = False  # --local: and never Jev


class Unsure(Exception):
    """The local model is unsure of the page, and only Jev could tell."""


def paint(text, color):
    codes = {"green": "32", "red": "31", "yellow": "33", "dim": "2"}
    return f"\033[{codes[color]}m{text}\033[0m" if TTY else text


async def judge(page):
    """The page's verdict, and who gave it: "jev" or "local"."""
    start = perf_counter()
    try:
        verdict = await page.judge(questions=None if presets_only else QUESTIONS)
    except ToolError as err:
        # Without a key, the browser can only answer what its model is sure of.
        if local and "TYPESAFE_API_KEY" in str(err):
            raise Unsure from err
        raise
    took = perf_counter() - start
    call_times.append(took)
    return verdict, "local" if local or presets_only and took < LOCAL_SECONDS else "jev"


def rate_limited(verdict):
    return verdict.is_rate_limited


def describe(verdict):
    """The page's kind when it is worth reading, else why not."""
    if rate_limited(verdict):
        return "rate limited"
    if verdict.ok:
        return verdict.answers["kind"]["choice"] if "kind" in verdict.answers else "ok"
    return verdict.reason.removeprefix("is_").replace("_", " ")  # is_login_wall -> login wall


async def visit(page, url):
    """Load `url` and act on what Jev sees, printing a line per verdict.
    Returns the last verdict and the actions taken."""
    name = url.split("://", 1)[1][:44]
    start = perf_counter()
    actions = []

    def show(verdict, by, action=None, detail=""):
        if action:
            actions.append(action)
            status, note = paint("retry", "yellow"), paint(f"  -> {action}{detail}", "yellow")
        elif verdict.ok and not rate_limited(verdict):
            status, note = paint("keep ", "green"), ""
        else:
            status, note = paint("skip ", "red"), ""
        took = paint(f"{perf_counter() - start:.2f}s", "dim")
        source = paint(f"{by:<5}  ", "dim") if presets_only and not local else ""
        print(f"{status}  {name:<44}  {describe(verdict):<22}  {source}{took}{note}")

    await page.goto(url=url)
    verdict, by = await judge(page)

    for attempt in (1, 2):
        if not rate_limited(verdict):
            break
        show(verdict, by, "back off", f" {3 * attempt}s")
        await asyncio.sleep(3 * attempt)
        await page.goto(url=url)
        verdict, by = await judge(page)

    if verdict.is_consent_wall:
        found = await page.find_element(role="button", name=ACCEPT)
        if found:
            show(verdict, by, "accept cookies")
            await found[0].click()
            await page.wait_for_state(state="networkidle")  # banners fade out
            verdict, by = await judge(page)

    if verdict.is_loading:
        show(verdict, by, "wait")
        await page.wait_for_state(state="networkidle")
        verdict, by = await judge(page)

    show(verdict, by)
    return verdict, actions


async def triage(browser, slots, url):
    """Visit `url` once a slot is free. Returns (outcome, actions, seconds)."""
    async with slots:
        start = perf_counter()
        try:
            async with await browser.new_session() as page:
                verdict, actions = await visit(page, url)
        except Unsure:
            print(f"{paint('jev? ', 'yellow')}  {url.split('://', 1)[1][:44]:<44}  {'unsure, would ask Jev':<22}")
            return "unsure", [], perf_counter() - start
        except LightpandaError as err:
            print(f"{paint('fail ', 'red')}  {url.split('://', 1)[1][:44]:<44}  {err}")
            return "failed", [], perf_counter() - start
        took = perf_counter() - start
    outcome = "kept" if verdict.ok and not rate_limited(verdict) else describe(verdict)
    return outcome, actions, took


async def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", help="ask the presets only, and Jev only those this local model is unsure of")
    parser.add_argument("--local", action="store_true", help="with --model: never ask Jev")
    opts = parser.parse_args()
    if opts.local and not opts.model:
        parser.error("--local needs --model")
    global presets_only, local
    model, local = opts.model, opts.local
    presets_only = bool(model)
    if local:
        os.environ.pop("TYPESAFE_API_KEY", None)  # the browser inherits our environment

    slots = asyncio.Semaphore(CRAWLERS)
    start = perf_counter()
    async with AsyncBrowser(args=["--classify-model", model] if model else []) as browser:
        results = await asyncio.gather(*(triage(browser, slots, url) for url in URLS))
    elapsed = perf_counter() - start

    outcomes = Counter(outcome for outcome, _, _ in results)
    kept = outcomes.pop("kept", 0)
    unsure = outcomes.pop("unsure", 0)
    acted = Counter(action for _, actions, _ in results for action in actions)
    page_time = sum(took for _, _, took in results)

    print()
    print(
        f"{len(results)} pages in {elapsed:.1f}s with {CRAWLERS} crawlers "
        f"({page_time:.1f}s of page time): {kept} kept, {len(results) - kept - unsure} skipped"
        + (f", {unsure} would ask Jev" if unsure else "")
    )
    if presets_only and not local:
        quick = [t for t in call_times if t < LOCAL_SECONDS]
        slow = [t for t in call_times if t >= LOCAL_SECONDS]
        print(f"classify: {len(call_times)} calls, {len(quick)} answered locally, {len(slow)} through Jev", end="")
        print(f" (median {statistics.median(slow):.2f}s)" if slow else "")
    elif call_times:
        print(
            f"{'Local' if local else 'Jev'}: {len(call_times)} calls, median {statistics.median(call_times):.2f}s, "
            f"p90 {statistics.quantiles(call_times, n=10)[-1]:.2f}s"
        )
    print("skipped: " + ", ".join(f"{reason} {n}" for reason, n in outcomes.most_common()))
    if acted:
        print("acted:   " + ", ".join(f"{action} {n}" for action, n in acted.most_common()))


asyncio.run(main())
