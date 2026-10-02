# /// script
# requires-python = ">=3.10"
# dependencies = ["lightpanda"]
#
# # judge() is not on PyPI yet: use this checkout until it is.
# [tool.uv.sources]
# lightpanda = { path = "..", editable = true }
# ///
"""Triage a crawl: skip pages that are walls, CAPTCHAs or empty searches with
``page.judge()``, then ask the rest your own questions with ``page.classify``.

Needs TYPESAFE_API_KEY. Run:  uv run examples/classify_pages.py
"""

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

with Browser() as browser:
    for url in URLS:
        name = url.split("://", 1)[1][:40]
        with browser.new_session() as page:
            page.goto(url=url)

            verdict = page.judge()
            if not verdict.ok:
                print(f"skip  {name:<40}  {verdict.reason}")
                continue

            answers = page.classify(questions=QUESTIONS)
            kind = answers["kind"]["choice"]
            price = answers["has_price"]
            content = answers["content"]
            print(
                f"keep  {name:<40}  {kind:<8} price {price:.2f}  "
                f"content {content['level']} ({content['score']:.2f})"
            )
