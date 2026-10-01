# /// script
# requires-python = ">=3.10"
# dependencies = ["lightpanda"]
#
# # judge() is not on PyPI yet: use this checkout until it is.
# [tool.uv.sources]
# lightpanda = { path = "..", editable = true }
# ///
"""Triage a crawl: ask each page what it is before scraping it.

A scraper that gets HTTP 200 can still be looking at a cookie wall, a bot
check or a search with no results. ``classify`` asks TypeSafe's page model
about the rendered page, so one call per URL says whether the page is worth
extracting from and what kind of page it is:

- ``True`` asks a preset: isBlocked, isCaptcha, isConsentWall, isEmptyCatalog
- a string asks a yes/no question and returns its probability
- ``{"question", "options"}`` asks a choice
- a plain list of categories returns the one that fits

When all you need is the gate, ``page.judge()`` asks every preset in one call
and answers with bools: ``ok`` is true when none of them fired.

Needs TYPESAFE_API_KEY in the environment.

Run:  uv run examples/classify_pages.py
"""

import os
import sys

from lightpanda import Browser

URLS = [
    "https://quotes.toscrape.com/js/",
    "https://books.toscrape.com/catalogue/a-light-in-the-attic_1000/index.html",
    "https://en.wikipedia.org/wiki/Headless_browser",
    "https://news.ycombinator.com/login",
    "https://en.wikipedia.org/w/index.php?search=xqzzzvqk+plorbt&fulltext=1&ns0=1",
    "https://www.g2.com/",
    "https://www.yahoo.com/",
]

QUESTIONS = {
    "isBlocked": True,
    "isCaptcha": True,
    "isConsentWall": True,
    "isEmptyCatalog": True,
    "has_price": "Does the page show a price?",
    "kind": {
        "question": "What kind of page is this?",
        "options": ["homepage", "listing", "product", "article", "search results", "login"],
    },
}

PRESETS = ["isBlocked", "isCaptcha", "isConsentWall", "isEmptyCatalog"]


def main() -> None:
    if not os.environ.get("TYPESAFE_API_KEY"):
        sys.exit("Set TYPESAFE_API_KEY to use classify.")

    with Browser() as browser:
        if "classify" not in browser.tools:
            sys.exit("This lightpanda binary has no classify tool; point LIGHTPANDA_BIN at a newer one.")

        print(f"{'page':<44}  {'kind':<16} {'price':>5}  flags")
        for url in URLS:
            with browser.new_session() as page:
                page.goto(url=url)
                answers = page.classify(questions=QUESTIONS)
                kind = answers["kind"]
                flags = ", ".join(f"{p} {answers[p]:.2f}" for p in PRESETS if answers[p] >= 0.5)
                name = url.split("://", 1)[1][:44]
                print(
                    f"{name:<44}  {kind['choice']:<16} {answers['has_price']:>5.2f}  {flags or '-'}"
                )

        # A list of categories is the shortest form, and `selector` narrows the
        # question to one part of the page.
        with browser.new_session() as page:
            page.goto(url="https://en.wikipedia.org/wiki/Headless_browser")
            topic = page.classify(questions=["software", "biology", "history", "sports"], selector="#bodyContent")
            print(f"\nWikipedia article body is about: {topic}")

        # judge() is the gate on its own: every preset, as bools.
        print()
        for url in URLS:
            with browser.new_session() as page:
                page.goto(url=url)
                verdict = page.judge()
                name = url.split("://", 1)[1][:44]
                print(f"keep  {name}" if verdict.ok else f"skip  {name:<44}  {verdict.reason:<16}  {verdict}")


if __name__ == "__main__":
    main()
