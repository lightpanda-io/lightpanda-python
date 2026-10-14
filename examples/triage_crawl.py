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
questions. The verdict decides what happens next: accept a cookie wall and
look again, back off when the site says we are going too fast, wait for a
page that is still loading, or skip the page and say why.

Needs TYPESAFE_API_KEY. Run:  uv run examples/triage_crawl.py
"""

import asyncio
import statistics
import sys
from collections import Counter
from time import perf_counter

from lightpanda import AsyncBrowser, LightpandaError

# A browser loads one page at a time, so a crawl runs several.
BROWSERS = 6

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
    "https://www.indeed.com/",
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
    "rate_limited": "Does the page say we sent too many requests and should slow down or retry later?",
    "kind": {
        "question": "What kind of page is this?",
        "options": ["product", "listing", "article", "docs", "search", "other"],
    },
}

# Accept buttons start with one of these, in the languages we crawl.
ACCEPT = "/^(accept|aceptar|accepter|akzeptieren|agree|allow)/i"

TTY = sys.stdout.isatty()


def paint(text, color):
    codes = {"green": "32", "red": "31", "yellow": "33", "dim": "2"}
    return f"\033[{codes[color]}m{text}\033[0m" if TTY else text


def label(reason):
    """is_empty_catalog -> empty catalog"""
    return reason.removeprefix("is_").replace("_", " ")


async def judge(page, jev_times):
    start = perf_counter()
    verdict = await page.judge(questions=QUESTIONS)
    jev_times.append(perf_counter() - start)
    return verdict


async def accept_cookies(page):
    """Click the wall's accept button. False when there is none to click."""
    found = await page.find_element(role="button", name=ACCEPT)
    if not found:
        return False
    await page.click(backend_node_id=found[0]["backendNodeId"])
    # Banners fade out after the click.
    await page.wait_for_state(state="networkidle")
    return True


async def visit(page, url, jev_times):
    """Load `url`, judge it, act on the verdict. Returns (verdict, actions)."""
    actions = []
    await page.goto(url=url)
    verdict = await judge(page, jev_times)
    for _ in range(3):
        if verdict.answers["rate_limited"] >= 0.5 and actions.count("backed off") < 2:
            actions.append("backed off")
            await asyncio.sleep(3 * actions.count("backed off"))
            await page.goto(url=url)
        elif verdict.is_consent_wall and "accepted cookies" not in actions and await accept_cookies(page):
            actions.append("accepted cookies")
        elif verdict.is_loading and "waited" not in actions:
            actions.append("waited")
            await page.wait_for_state(state="networkidle")
        else:
            break
        verdict = await judge(page, jev_times)
    return verdict, actions


def outcome(verdict):
    """Either "kept" or why the page was skipped."""
    if verdict.answers["rate_limited"] >= 0.5:
        return "rate limited"
    return "kept" if verdict.ok else label(verdict.reason)


async def crawl(browser, queue, results, jev_times):
    while not queue.empty():
        url = queue.get_nowait()
        name = url.split("://", 1)[1][:44]
        start = perf_counter()
        async with await browser.new_session() as page:
            try:
                verdict, actions = await visit(page, url, jev_times)
            except LightpandaError as err:
                print(f"{paint('fail', 'red')}  {name:<44}  {err}")
                results.append(("failed", [], perf_counter() - start))
                continue
        took = perf_counter() - start
        acted = paint(f"  <- {', '.join(actions)}", "yellow") if actions else ""
        result = outcome(verdict)
        if result == "kept":
            line = f"{paint('keep', 'green')}  {name:<44}  {verdict.answers['kind']['choice']:<22}"
        else:
            line = f"{paint('skip', 'red')}  {name:<44}  {result:<22}"
        print(f"{line}  {paint(f'{took:.2f}s', 'dim')}{acted}")
        results.append((result, actions, took))


async def main():
    queue = asyncio.Queue()
    for url in URLS:
        queue.put_nowait(url)
    results, jev_times = [], []

    start = perf_counter()
    browsers = [AsyncBrowser() for _ in range(BROWSERS)]
    try:
        await asyncio.gather(*(b.start() for b in browsers))
        await asyncio.gather(*(crawl(b, queue, results, jev_times) for b in browsers))
    finally:
        await asyncio.gather(*(b.close() for b in browsers))
    elapsed = perf_counter() - start

    outcomes = Counter(result for result, _, _ in results)
    kept = outcomes.pop("kept", 0)
    acted = Counter(action for _, actions, _ in results for action in actions)
    page_time = sum(took for _, _, took in results)
    jev_times.sort()

    print()
    print(
        f"{len(results)} pages in {elapsed:.1f}s with {BROWSERS} browsers "
        f"({page_time:.1f}s of page time): {kept} kept, {len(results) - kept} skipped"
    )
    print(
        f"Jev: {len(jev_times)} calls, median {statistics.median(jev_times):.2f}s, "
        f"p90 {jev_times[int(len(jev_times) * 0.9)]:.2f}s"
    )
    print("skipped: " + ", ".join(f"{reason} {n}" for reason, n in outcomes.most_common()))
    if acted:
        print("acted:   " + ", ".join(f"{action} {n}" for action, n in acted.most_common()))


asyncio.run(main())
