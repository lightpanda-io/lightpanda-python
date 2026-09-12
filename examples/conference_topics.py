# /// script
# requires-python = ">=3.10"
# dependencies = ["lightpanda", "model2vec", "scikit-learn", "pandas", "matplotlib"]
# ///
"""Map the research topics of an ML conference from its JavaScript-only site.

https://neurips.cc/virtual/2025/papers.html is an empty shell without
JavaScript: the HTML holds ~1 KB of navigation and a ``<noscript>`` apology,
and the page's scripts fetch a 27 MB index of the 5858 accepted papers and
render the cards client-side. ``requests`` + BeautifulSoup find 0 papers.

Lightpanda runs the page. From Python we switch it to the "detail" layout
(cards then include the abstract), walk the site's own topic filter, and
``extract`` every rendered card: title, authors, topic, abstract, URL. That is
the dataset. The ML part runs on it locally, on CPU, in seconds:

* embed title + abstract with a static sentence-embedding model (model2vec),
* cluster the embeddings (k-means) and name each cluster by its TF-IDF terms,
* compare the clusters with the conference's own topic labels
  (adjusted Rand index), and
* lay the papers out in 2D (t-SNE) as ``conference_topics.png``.

Run:  uv run examples/conference_topics.py [--conferences neurips-2025 iclr-2026]
                                            [--clusters 8] [--query "..."] [--csv papers.csv]
"""

import argparse
import asyncio
import functools
import json
import re
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lightpanda import AsyncBrowser

# One object per rendered card; selectors resolve inside the card.
CARDS = {
    "cards": [
        {
            "selector": ".myCard",
            "fields": {
                "title": ".card-title",
                "url": {"selector": "a.text-muted", "attr": "href"},
                "authors": ["h6.card-subtitle a"],
                "topic": ".card-topic a",
                "abstract": ".card-text",
            },
        }
    ],
    "shown": "#show_panel",  # "showing 12 of 12 papers"
}

NAMES = {"neurips": "NeurIPS", "iclr": "ICLR", "icml": "ICML"}
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def site_url(conference: str) -> str:
    """'neurips-2025' -> the papers page of that conference's virtual site."""
    name, year = conference.rsplit("-", 1)
    return f"https://{name}.cc/virtual/{year}/papers.html"


def shown_of(text: str) -> tuple[int, int]:
    """Parse the page's 'showing N of M papers' counter."""
    shown, total = re.search(r"showing (\d+) of (\d+)", text).groups()
    return int(shown), int(total)


async def select_topic(page, topic: str) -> None:
    """Pick `topic` in the page's own <select> and return once its cards have been re-rendered."""
    # The page re-renders asynchronously (it refreshes bookmarks first), so wait for the card
    # list to stop mutating and to show the requested topic rather than a previous render.
    await page.evaluate(script=f"""
        await new Promise(done => {{
            let timer = null;
            const settled = () => {{
                const first = document.querySelector(".myCard .card-topic a");
                if (first && first.innerText.toLowerCase().includes({json.dumps(topic.lower())})) {{
                    observer.disconnect();
                    done();
                }}
            }};
            const observer = new MutationObserver(() => {{
                clearTimeout(timer);
                timer = setTimeout(settled, 300);
            }});
            observer.observe(document.querySelector(".cards"), {{childList: true, subtree: true}});
            const select = document.querySelector("#topic-filter");
            select.value = {json.dumps(topic)};
            select.dispatchEvent(new Event("change"));
        }});""", timeout=60_000)


async def scrape_conference(browser: AsyncBrowser, conference: str, limit_topics: int | None) -> list[dict]:
    """Every paper of one conference that carries a topic label, with its abstract."""
    rows: list[dict] = []
    async with browser.session() as page:
        await page.goto(url=site_url(conference), timeout=120_000)
        # The cards appear once the page's JavaScript has fetched and rendered the index.
        await page.wait_for_selector(selector=".myCard", timeout=120_000)
        _, total = shown_of((await page.extract(schema={"shown": "#show_panel"}))["shown"])
        # Switch to the site's "detail" layout: cards now include the abstract and topic.
        await page.click(selector="#option4")
        await page.wait_for_selector(selector=".pp-mode-detail", timeout=60_000)
        options = await page.extract(
            schema={"topics": [{"selector": "#topic-filter option", "attr": "value"}]})
        topics = [t for t in options["topics"] if t and t != "All"][:limit_topics]
        for topic in topics:
            await select_topic(page, topic)
            data = await page.extract(schema=CARDS)
            shown, matched = shown_of(data["shown"])
            # The filter is a substring match, so "Applications" also lists every "Applications->..."
            # paper; those come back under their own topic. Warn only when papers are really lost.
            if shown < matched and not any(t.startswith(topic + "->") for t in topics):
                print(f"  {conference}: {topic!r} has {matched} papers, the page renders at most {shown}",
                      file=sys.stderr)
            rows.extend({"conference": conference, **card} for card in data["cards"])
            print(f"  {conference}: {len(rows):5d} cards after {topic!r}", file=sys.stderr)
    print(f"{conference}: {len(rows)} cards from {len(topics)} topic filters ({total} accepted)",
          file=sys.stderr)
    return rows


async def scrape(conferences: list[str], limit_topics: int | None) -> pd.DataFrame:
    """One browser session per conference, run concurrently."""
    # The 27 MB paper index takes longer than the browser's default 15 s per-request timeout.
    async with AsyncBrowser(args=["--http-timeout", "120000"], max_concurrency=4) as browser:
        results = await asyncio.gather(*(scrape_conference(browser, c, limit_topics) for c in conferences))
    df = pd.DataFrame([row for rows in results for row in rows]).drop_duplicates("url")
    df = df[df["topic"].notna()].reset_index(drop=True)
    df["abstract"] = df["abstract"].fillna("")
    df["area"] = df["topic"].str.split("->").str[0]  # "Deep Learning->Generative Models" -> "Deep Learning"
    return df


@functools.cache
def model():
    """potion-base-8M: a 30 MB static sentence-embedding model, no torch, no GPU."""
    from huggingface_hub.utils import disable_progress_bars
    from model2vec import StaticModel

    disable_progress_bars()  # the one-time download is quiet, like the rest of the script
    return StaticModel.from_pretrained("minishlab/potion-base-8M")


def embed(texts: list[str]) -> np.ndarray:
    """Unit-length embeddings, one row per text."""
    vectors = model().encode(texts, show_progress_bar=False)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def cluster(df: pd.DataFrame, X: np.ndarray, k: int) -> list[str]:
    """k-means on the embeddings; fills df['cluster'] and returns a three-term name per cluster."""
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

    df["cluster"] = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)
    # Name each cluster by the terms its papers use more than the conference as a whole does.
    latex = {"mathcal", "mathbb", "sqrt", "frac", "log", "cdot", "leq", "geq", "tilde", "widetilde",
             "epsilon", "eta"}
    tfidf = TfidfVectorizer(stop_words=list(ENGLISH_STOP_WORDS | latex), ngram_range=(1, 2),
                            min_df=20, max_df=0.3, sublinear_tf=True)
    weights = tfidf.fit_transform(df["text"])
    terms = tfidf.get_feature_names_out()
    overall = np.asarray(weights.mean(axis=0)).ravel()
    names = []
    for c in range(k):
        lift = np.asarray(weights[(df["cluster"] == c).to_numpy()].mean(axis=0)).ravel() - overall
        names.append(", ".join(terms[i] for i in lift.argsort()[::-1][:3]))
    return names


def report(df: pd.DataFrame, names: list[str]) -> None:
    from sklearn.metrics import adjusted_rand_score

    print(f"\n{len(df)} papers, {df['topic'].nunique()} topics in {df['area'].nunique()} areas, "
          f"{len(names)} clusters")
    print(f"Adjusted Rand index of the clusters vs the official topics: "
          f"{adjusted_rand_score(df['topic'], df['cluster']):.2f}, "
          f"vs the top-level areas: {adjusted_rand_score(df['area'], df['cluster']):.2f}")
    rows = []
    for c, name in enumerate(names):
        members = df[df["cluster"] == c]
        top = members["area"].value_counts(normalize=True).head(2)
        rows.append({"cluster": c + 1, "papers": len(members), "terms": name,
                     "main areas": ", ".join(f"{a} {s:.0%}" for a, s in top.items())})
    print("\n" + pd.DataFrame(rows).to_string(index=False))


def search(df: pd.DataFrame, X: np.ndarray, query: str, n: int = 5) -> None:
    best = (X @ embed([query])[0]).argsort()[::-1][:n]  # cosine similarity: rows are unit length
    print(f"\nClosest papers to {query!r}:")
    for i in best:
        print(f"  {df.iloc[i]['title']}\n    {df.iloc[i]['topic']} - {df.iloc[i]['url']}")


def plot(df: pd.DataFrame, X: np.ndarray, names: list[str], title: str, out: Path) -> None:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    print("\nProjecting to 2D with t-SNE ...", file=sys.stderr)
    xy = TSNE(n_components=2, init="pca", perplexity=30, random_state=0).fit_transform(
        PCA(n_components=50, random_state=0).fit_transform(X))

    matplotlib.use("Agg")
    fig, ax = plt.subplots(figsize=(11, 8), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    colors = [PALETTE[c % len(PALETTE)] for c in df["cluster"]]
    ax.scatter(xy[:, 0], xy[:, 1], s=7, c=colors, alpha=0.65, linewidths=0)
    handles = []
    for c, name in enumerate(names):
        members = df["cluster"].to_numpy() == c
        cx, cy = np.median(xy[members], axis=0)
        color = PALETTE[c % len(PALETTE)]
        ax.text(cx, cy, str(c + 1), ha="center", va="center", fontsize=11, fontweight="bold", color="#0b0b0b",
                bbox={"boxstyle": "circle,pad=0.25", "facecolor": "#fcfcfb", "edgecolor": color,
                      "linewidth": 1.5})
        handles.append(plt.Line2D([], [], marker="o", linestyle="", markersize=7, color=color,
                                  label=f"{c + 1}  {name}  ({members.sum()})"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False,
              fontsize=9, labelcolor="#52514e", handletextpad=0.4, columnspacing=1.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, loc="left", fontsize=12, color="#0b0b0b")
    fig.text(0.01, 0.006, "One dot per accepted paper, placed by t-SNE of its title + abstract embedding "
             "(model2vec potion-base-8M), coloured by k-means cluster.\nData: the cards of the conference's "
             "virtual site, rendered by Lightpanda.", fontsize=8, color="#52514e", ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, dpi=120)
    print(f"Map written to {out}", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--conferences", nargs="+", default=["neurips-2025"], metavar="NAME-YEAR",
                        help="virtual sites to read, e.g. neurips-2025 iclr-2026 icml-2026 "
                             "(default: neurips-2025)")
    parser.add_argument("--clusters", type=int, default=8, metavar="K",
                        help="number of k-means clusters (default: 8)")
    parser.add_argument("--query", metavar="TEXT", help="also print the five papers closest to this text")
    parser.add_argument("--csv", type=Path, metavar="PATH", help="write the scraped papers to a CSV file")
    parser.add_argument("--from-csv", type=Path, metavar="PATH", help="skip the browser, analyse a saved CSV")
    parser.add_argument("--limit-topics", type=int, metavar="N",
                        help="read only the first N topics (quick run)")
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # keep stdout and the stderr progress in order

    if args.from_csv:
        df = pd.read_csv(args.from_csv).fillna({"abstract": ""})
        df["area"] = df["topic"].str.split("->").str[0]
        args.conferences = list(df["conference"].unique())
    else:
        print(f"Reading {', '.join(args.conferences)} ...", file=sys.stderr)
        df = asyncio.run(scrape(args.conferences, args.limit_topics))
    if args.csv:
        df.drop(columns=["area"]).to_csv(args.csv, index=False)
        print(f"Papers written to {args.csv}", file=sys.stderr)

    df["text"] = df["title"] + ". " + df["abstract"]
    X = embed(df["text"].tolist())
    names = cluster(df, X, args.clusters)
    report(df, names)
    if args.query:
        search(df, X, args.query)
    label = " + ".join(f"{NAMES.get(name, name.capitalize())} {year}"
                       for name, year in (c.rsplit("-", 1) for c in args.conferences))
    plot(df, X, names, f"{label}: {len(df)} accepted papers by abstract similarity",
         Path(__file__).with_name("conference_topics.png"))
