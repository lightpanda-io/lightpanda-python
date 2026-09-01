# /// script
# requires-python = ">=3.10"
# dependencies = ["lightpanda", "pandas", "matplotlib"]
# ///
"""Scrape a JavaScript-rendered site and analyse it with pandas.

https://quotes.toscrape.com/js/ renders every quote client-side with jQuery:
``requests`` gets the empty page shell (0 quotes). Lightpanda runs the page's
JavaScript, and ``extract`` returns structured records straight from the
rendered DOM — nested tag lists included — so there is no HTML parsing step
between the browser and the DataFrame.

Run:  uv run examples/quotes_analysis.py
"""

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from lightpanda import Browser

START_URL = "https://quotes.toscrape.com/js/"

# One record per `.quote`, with the fields resolved relative to it; `next` is
# the pagination link's absolute URL (null on the last page).
SCHEMA = {
    "quotes": [
        {
            "selector": ".quote",
            "fields": {"text": ".text", "author": ".author", "tags": [".tag"]},
        }
    ],
    "next": {"selector": "li.next a", "attr": "href"},
}


def scrape(start_url: str = START_URL) -> pd.DataFrame:
    """Follow the site's own "Next" links and collect every quote."""
    rows = []
    with Browser() as browser, browser.new_session() as page:
        url = start_url
        while url:
            page.goto(url=url)
            data = page.extract(schema=SCHEMA)
            rows.extend(data["quotes"])
            url = data["next"]
            print(f"  {len(rows):3d} quotes so far", file=sys.stderr)
    return pd.DataFrame(rows)


def analyse(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    top_tags = df.explode("tags")["tags"].value_counts().head(10)
    top_authors = df["author"].value_counts().head(10)

    print(f"\n{len(df)} quotes from {df['author'].nunique()} authors")
    print("\nTop tags:\n" + top_tags.to_string())
    print("\nMost quoted authors:\n" + top_authors.to_string())
    print("\nQuote length (characters):\n" + df["text"].str.len().describe().round(1).to_string())
    return top_tags, top_authors


def plot(top_tags: pd.Series, top_authors: pd.Series, out: Path) -> None:
    matplotlib.use("Agg")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), facecolor="#fcfcfb")
    for ax, series, title in (
        (axes[0], top_tags, "Most common tags"),
        (axes[1], top_authors, "Most quoted authors"),
    ):
        series = series[::-1]  # largest at the top
        ax.barh(series.index, series.values, color="#2a78d6", height=0.6)
        ax.bar_label(ax.containers[0], padding=4, color="#52514e", fontsize=9)
        ax.set_title(title, loc="left", color="#0b0b0b", fontsize=12)
        ax.set_facecolor("#fcfcfb")
        ax.set_xlim(0, series.max() * 1.15)
        ax.tick_params(colors="#52514e", length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#d9d8d3")
        ax.xaxis.grid(True, color="#e8e7e2", linewidth=0.8)
        ax.set_axisbelow(True)
    fig.suptitle("quotes.toscrape.com/js — rendered by Lightpanda, analysed with pandas",
                 x=0.01, ha="left", fontsize=10, color="#52514e")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"\nChart written to {out}")


if __name__ == "__main__":
    start = sys.argv[1] if len(sys.argv) > 1 else START_URL
    print(f"Scraping {start} ...", file=sys.stderr)
    df = scrape(start)
    print(df.head().to_string())
    plot(*analyse(df), Path(__file__).with_name("quotes.png"))
