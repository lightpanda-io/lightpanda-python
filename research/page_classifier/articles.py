# /// script
# requires-python = ">=3.10"
# dependencies = ["lightpanda"]
#
# [tool.uv.sources]
# lightpanda = { path = "../..", editable = true }
# ///
"""Add article links from the paywalled news sites to the URL list.

A paywall sits on the articles, not on the homepage that links to them, so
the homepages in seeds.py's PAYWALLS catch almost none. This opens each one,
keeps the links that look like articles (a headline-length anchor on the
same site, with a dated or long slugged path) and appends a few of them per
site to urls.tsv as `article` rows. collect.py then crawls only the new ones.

Run:  uv run research/page_classifier/articles.py --per-site 6
"""

import argparse
import asyncio
import random
import re
from pathlib import Path
from urllib.parse import urlsplit

from lightpanda import AsyncBrowser, LightpandaError
from seeds import DATA, PAYWALLS

# A year or date in the path, or a slug of several hyphenated words.
ARTICLE_PATH = re.compile(r"/20\d\d/|/\d{4}-\d\d-\d\d|[a-z0-9]+(?:-[a-z0-9]+){3,}", re.IGNORECASE)
NOT_ARTICLES = re.compile(r"/(tag|tags|topic|topics|author|authors|section|video|videos|live|podcast|newsletter|subscribe|account|login|search)/", re.IGNORECASE)


def base(host: str) -> str:
    """Last two labels, or three under a two-letter second level (smh.com.au)."""
    labels = host.lower().split(".")
    return ".".join(labels[-3:] if len(labels) > 2 and len(labels[-2]) <= 3 and len(labels[-1]) == 2 else labels[-2:])


def articles(home: str, links: list[dict]) -> list[str]:
    site = base(urlsplit(home).netloc)
    found = []
    for link in links:
        href = link["href"].split("#", 1)[0]
        parts = urlsplit(href)
        if parts.scheme not in ("http", "https") or base(parts.netloc) != site:
            continue
        if len(link.get("text") or "") < 25 or NOT_ARTICLES.search(parts.path) or not ARTICLE_PATH.search(parts.path):
            continue
        if href not in found:
            found.append(href)
    return found


async def visit(browser, slots, home: str) -> list[str]:
    async with slots:
        try:
            async with await browser.new_session() as page:
                links = await page.links(url=home, timeout=20000)
        except LightpandaError as err:
            print(f"  {home}: {err}")
            return []
        if not isinstance(links, list):  # the browser explained a failed load instead
            print(f"  {home}: {str(links)[:80]}")
            return []
        return articles(home, links)


async def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--per-site", type=int, default=6)
    parser.add_argument("--crawlers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--urls", type=Path, default=DATA / "urls.tsv")
    parser.add_argument("--sites", nargs="+", default=PAYWALLS, help="homepages to visit (default: all of PAYWALLS)")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    slots = asyncio.Semaphore(args.crawlers)
    async with AsyncBrowser() as browser:
        found = await asyncio.gather(*(visit(browser, slots, home) for home in args.sites))

    listed = {line.split("\t", 1)[1] for line in args.urls.read_text().splitlines()}
    new = []
    for home, links in zip(args.sites, found):
        picked = rng.sample(links, min(args.per_site, len(links)))
        print(f"{len(links):>4} articles, {len(picked)} picked  {home}")
        new += [url for url in picked if url not in listed]
    with args.urls.open("a") as f:
        f.writelines(f"article\t{url}\n" for url in new)
    print(f"{len(new)} article urls -> {args.urls}")


if __name__ == "__main__":
    asyncio.run(main())
