# /// script
# requires-python = ">=3.10"
# dependencies = ["lightpanda"]
#
# # judge() is not on PyPI yet: use this checkout until it is.
# [tool.uv.sources]
# lightpanda = { path = "..", editable = true }
# ///
"""Triage a crawl: ``page.judge()`` says whether each page is a wall, a CAPTCHA
or an empty search, and answers your own questions in the same call.

Needs TYPESAFE_API_KEY. Run:  uv run examples/classify_pages.py
"""

from time import perf_counter

from lightpanda import Browser

URLS = [
    "https://books.toscrape.com/catalogue/a-light-in-the-attic_1000/index.html",
    "https://quotes.toscrape.com/js/",
    "https://en.wikipedia.org/wiki/Headless_browser",
    "https://news.ycombinator.com/login",
    "https://en.wikipedia.org/w/index.php?search=xqzzzvqk+plorbt&fulltext=1",
    "https://www.g2.com/",
]

QUESTIONS = {
    # yes/no
    "has_price": "Does the page show a price?",
    # choice
    "kind": {
        "question": "What kind of page is this?",
        "options": ["product", "listing", "article", "other"],
    },
    # score
    "content": {
        "question": "How much useful content would a scraper get from this page?",
        "levels": ["none", "little", "plenty"],
    },
}


def timed(call, **kwargs):
    """The call's result and the seconds it took."""
    start = perf_counter()
    return call(**kwargs), perf_counter() - start


with Browser() as browser:
    for url in URLS:
        name = url.split("://", 1)[1][:40]
        with browser.new_session() as page:
            _, load = timed(page.goto, url=url)
            # One call: every preset, plus our own questions.
            verdict, jev = timed(page.judge, questions=QUESTIONS)
            times = f"[load {load:.2f}s, jev {jev:.2f}s]"
            if not verdict.ok:
                print(f"skip  {name:<40}  {verdict.reason:<38}  {times}")
                continue

            answers = verdict.answers
            kind = answers["kind"]["choice"]
            price = answers["has_price"]
            content = answers["content"]
            print(
                f"keep  {name:<40}  {kind:<8} price {price:.2f}  "
                f"content {content['level']} ({content['score']:.2f})  {times}"
            )
