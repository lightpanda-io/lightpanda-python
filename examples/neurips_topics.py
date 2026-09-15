# /// script
# requires-python = ">=3.10"
# dependencies = ["lightpanda", "pandas", "numpy", "requests", "fastembed", "scikit-learn",
#                 "matplotlib"]
# ///
"""What NeurIPS started working on, discovered from a site that serves no HTML.

https://neurips.cc/virtual/2025/papers.html renders nothing without JavaScript.
``requests`` gets 800 KB of scripts around an empty card list and a
``<noscript>`` apology; BeautifulSoup finds 0 papers. The page fetches a 27 MB
index, keeps every paper in a JavaScript array, and renders at most 400 cards
from it, so reading the DOM would still miss most of the conference.

Lightpanda runs the page and reads that array instead, one browser per year and
five years at once: 19,219 papers with abstracts, in about a minute.

Those are embedded once, then each year is clustered on its own by Chinese
Whispers: link every pair of papers alike enough, and let each paper repeatedly
adopt the weighted-majority topic of whatever it is linked to. No number of
clusters is chosen anywhere. Pooling the years would average a topic against the
years it did not exist in, so instead every year contributes its own topics, and
duplicates are merged afterwards.

The centre of each topic then labels every paper in every year, which makes a
topic's share comparable across time. Each one is named by the phrases its
papers use far more than the conference does.

What comes out is the shape of the field: "3d gaussian splatting" appears from
nothing in 2024, "kv cache, long context" rises tenfold, "federated" and
"adversarial robustness" fall away. Nothing supplied those topics.

Passing a query searches the same embeddings by meaning instead.

Run:  uv run examples/neurips_topics.py
      uv run examples/neurips_topics.py "making models reason step by step"
"""
import argparse
import asyncio
import functools
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from lightpanda import AsyncBrowser

YEARS = [2021, 2022, 2023, 2024, 2025]

# The page keeps every paper here; only 400 of them ever reach the DOM.
PULL = "return allPapers.map(p => [p.title, p.abstract || '', p.url, p.id])"
# 8192 input tokens, comfortably more than any abstract here, so nothing truncates.
GEMINI = "gemini-embedding-2"
EMBED_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI}:batchEmbedContents"


async def scrape_year(year: int) -> list[dict]:
    """Every paper of one year, read from the page's own data rather than its cards."""
    # One browser process per year: concurrent sessions sharing a process fail on this
    # workload, each parsing a 27 MB index. --http-timeout is a deadline for the whole
    # response rather than an idle timeout, so that index trips the 15 s default.
    async with AsyncBrowser(args=["--http-timeout", "180000"], max_concurrency=1) as browser:
        async with browser.session() as page:
            await page.goto(url=f"https://neurips.cc/virtual/{year}/papers.html", timeout=180_000)
            # `allPapers` is filled just before the first cards render.
            await page.wait_for_script(
                script="typeof allPapers !== 'undefined' && allPapers.length > 0", timeout=120_000)
            rows = await page.evaluate(script=PULL, timeout=180_000)
    print(f"  {year}: {len(rows)} papers", file=sys.stderr)
    # Site-relative paths, and the oldest years carry none at all: those are linked by id.
    return [{"year": year, "title": t, "abstract": a,
             "url": f"https://neurips.cc{u}" if (u or "").startswith("/")
                    else f"https://neurips.cc/virtual/{year}/poster/{i}"}
            for t, a, u, i in rows]


async def scrape(years: list[int]) -> pd.DataFrame:
    """All the requested years at once, one browser each."""
    results = await asyncio.gather(*(scrape_year(y) for y in years))
    return pd.DataFrame([row for rows in results for row in rows])


def corpus(df: pd.DataFrame) -> pd.Series:
    return df["title"] + ". " + df["abstract"]


def embed_gemini(texts: list[str], key: str) -> np.ndarray:
    """Gemini embeddings, 100 texts per request, a few requests in flight at once."""
    import concurrent.futures as cf

    import requests

    session, batches = requests.Session(), [texts[i:i + 100] for i in range(0, len(texts), 100)]

    def one(batch: list[str]) -> list[list[float]]:
        body = {"requests": [{"model": f"models/{GEMINI}", "outputDimensionality": 768,
                              "content": {"parts": [{"text": t}]}} for t in batch]}
        for attempt in range(5):
            r = session.post(EMBED_URL, headers={"x-goog-api-key": key}, json=body, timeout=180)
            if r.status_code == 429:  # rate limited: back off and try again
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            return [e["values"] for e in r.json()["embeddings"]]
        r.raise_for_status()
        return []

    done, out = 0, []
    with cf.ThreadPoolExecutor(max_workers=6) as pool:
        for chunk in pool.map(one, batches):
            out += chunk
            done += len(chunk)
            print(f"  embedded {done}/{len(texts)}", end="\r", file=sys.stderr)
    return np.array(out)


@functools.cache
def local_model():
    from fastembed import TextEmbedding

    return TextEmbedding(model_name="BAAI/bge-small-en-v1.5")


def embed_local(texts: list[str]) -> np.ndarray:
    """bge-small: 130 MB of ONNX, no torch, no key, no network, but slow over a whole corpus.

    Its limit is 512 tokens, which fastembed enforces by truncating.
    """
    try:  # all cores, where fastembed's multiprocessing is available
        return np.array(list(local_model().embed(texts, batch_size=256, parallel=0)))
    except Exception as exc:
        print(f"  (parallel encoding unavailable: {type(exc).__name__}; using one core)",
              file=sys.stderr)
        return np.array(list(local_model().embed(texts, batch_size=256)))


def unit(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


def embed(texts: list[str], key: str | None) -> np.ndarray:
    return unit(embed_gemini(texts, key) if key else embed_local(texts))


def default_cache() -> Path:
    base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return base / "neurips_topics" / "embeddings.npz"


def embed_corpus(df: pd.DataFrame, key: str | None, cache: Path) -> np.ndarray:
    """Embed every paper, reusing whatever a previous run already paid for.

    Keyed by paper URL, so adding a year only embeds that year. float16 is ample for a
    cosine ranking and halves a file that runs to tens of megabytes.
    """
    model = GEMINI if key else "bge-small-en-v1.5"
    store: dict[str, np.ndarray] = {}
    if cache.exists():
        with np.load(cache, allow_pickle=False) as z:
            if z["model"].item() == model:
                store = dict(zip(z["keys"].tolist(), z["vectors"]))
            else:
                print(f"  cache holds {z['model'].item()} embeddings; re-embedding", file=sys.stderr)

    keys, texts = df["url"].tolist(), corpus(df).tolist()
    todo = [i for i, k in enumerate(keys) if k not in store]
    if todo:
        if not key:
            print(f"  no GOOGLE_API_KEY, so {len(todo)} papers are encoded locally. This takes a "
                  f"while once; {cache} makes every later run instant.", file=sys.stderr)
        for i, vector in zip(todo, embed([texts[i] for i in todo], key)):
            store[keys[i]] = vector.astype(np.float16)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, model=model, keys=np.array(list(store)),
                            vectors=np.stack(list(store.values())))
        print(f"  cached {len(store)} embeddings in {cache}", file=sys.stderr)
    else:
        print(f"  {len(keys)} embeddings came straight from {cache}", file=sys.stderr)
    return np.stack([store[k] for k in keys]).astype(np.float32)


def graph(X: np.ndarray, threshold: float) -> dict[int, list[tuple[int, float]]]:
    """Every pair of papers more alike than `threshold`, as an adjacency list.

    Compared in chunks, so the 185 million possible pairs never exist at once. The bar is
    high enough that what survives is small: about 200,000 edges out of those millions.
    """
    adjacency: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for i in range(0, len(X), 2000):
        block = X[i:i + 2000] @ X.T
        for row in range(len(block)):
            block[row, :i + row + 1] = -1  # upper triangle: count each pair once
        for r, c in zip(*np.where(block > threshold)):
            a, b, weight = int(r) + i, int(c), float(block[r, c])
            adjacency[a].append((b, weight))
            adjacency[b].append((a, weight))
    return adjacency


def whisper(adjacency: dict[int, list[tuple[int, float]]], n: int, rounds: int = 8) -> np.ndarray:
    """Chinese Whispers: take the weighted-majority label of your neighbours, repeatedly.

    Unlike k-means this needs no k, and unlike a nearest-topic assignment it leaves papers
    with no close neighbour in a cluster of their own rather than forcing them somewhere.
    """
    rng = np.random.default_rng(0)
    label = np.arange(n)
    for _ in range(rounds):
        for i in rng.permutation(n):
            if not (close := adjacency.get(i)):
                continue
            tally: dict[int, float] = defaultdict(float)
            for j, weight in close:
                tally[label[j]] += weight
            label[i] = max(tally, key=tally.get)
    return label


def centroids(df: pd.DataFrame, X: np.ndarray, threshold: float, min_cluster: int = 15,
              merge: float = 0.95) -> np.ndarray:
    """Discover topics inside each year, then pool them into one set of definitions.

    Clustering a year on its own finds what that year was about. Pooled over five years the
    same topic would be averaged against the years it did not exist in, and the ones that
    only appeared recently get lost.
    """
    found: list[tuple[int, np.ndarray]] = []
    for year in sorted(df["year"].unique()):
        rows = (df["year"] == year).to_numpy()
        V = X[rows]
        label = whisper(graph(V, threshold), len(V))
        sizes = pd.Series(label).value_counts()
        for cluster in sizes[sizes >= min_cluster].index:
            members = label == cluster
            found.append((int(members.sum()), unit(V[members].mean(axis=0))))
        print(f"  {year}: {len(sizes[sizes >= min_cluster])} topics", file=sys.stderr)

    # The same topic is found again every year it persists, so keep the biggest of each.
    found.sort(key=lambda f: -f[0])
    kept: list[np.ndarray] = []
    for _, vector in found:
        if all(float(vector @ other) < merge for other in kept):
            kept.append(vector)
    print(f"  {len(found)} topics over all years, {len(kept)} after merging duplicates",
          file=sys.stderr)
    return np.stack(kept)


def topics(df: pd.DataFrame, X: np.ndarray, threshold: float, min_size: int) -> pd.DataFrame:
    """Label every paper with its nearest topic, and measure each topic's share of every year.

    One set of definitions covers all five years, so a topic's share is comparable across
    them. Every paper gets a label, which is only safe because the topics came from the
    corpus: with a handful of hand-written ones, most papers would land in whichever was
    least wrong.
    """
    from sklearn.feature_extraction.text import CountVectorizer

    print(f"  clustering {len(df):,} papers ...", file=sys.stderr)
    C = centroids(df, X, threshold)
    nearest = (X @ C.T).argmax(axis=1)

    vec = CountVectorizer(ngram_range=(1, 3), min_df=10, max_df=0.3, stop_words="english",
                          binary=True)
    D = vec.fit_transform(corpus(df).str.lower())
    terms = np.array(vec.get_feature_names_out())
    overall = np.asarray(D.mean(axis=0)).ravel()

    def name(members: np.ndarray) -> str:
        """The phrases this topic uses far more than the conference as a whole."""
        inside = np.asarray(D[members].mean(axis=0)).ravel()
        lift = (inside + 0.003) / (overall + 0.003)
        kept: list[str] = []
        for i in np.argsort(lift)[::-1]:
            if inside[i] < 0.25 or any(terms[i] in k or k in terms[i] for k in kept):
                continue
            kept.append(str(terms[i]))
            if len(kept) == 3:
                break
        return ", ".join(kept)

    years = sorted(df["year"].unique())
    rows = {}
    for topic in range(len(C)):
        members = nearest == topic
        if members.sum() < min_size:
            continue
        rows[name(members)] = [members[(df["year"] == y).to_numpy()].mean() * 100 for y in years]
    named = pd.DataFrame(rows, index=pd.Index(years, name="year"))
    print(f"  {named.shape[1]} topics with {min_size}+ papers", file=sys.stderr)
    return named


def movers(pct: pd.DataFrame, n: int = 4) -> pd.DataFrame:
    """The clusters that grew and shrank most, by ratio between the first and last year."""
    growth = ((pct.iloc[-1] + 0.05) / (pct.iloc[0] + 0.05)).sort_values()
    return pct[list(growth.index[::-1][:n]) + list(growth.index[:n])]


def words_of(query: str) -> re.Pattern:
    """The query's own content words, for spotting hits that share none of them."""
    common = {"the", "a", "an", "of", "for", "to", "in", "on", "with", "and", "or", "by", "that",
              "how", "what", "using", "use", "via", "from", "at", "is", "are", "be", "as", "it"}
    words = [w for w in re.findall(r"[a-z0-9]+", query.lower()) if w not in common and len(w) > 2]
    return re.compile("|".join(re.escape(w) for w in words) or r"(?!)", re.I)


def show(df: pd.DataFrame, X: np.ndarray, key: str | None, query: str, top: int) -> None:
    """Rank every paper against the query and print the best of them."""
    scores = X @ embed([query], key)[0]
    shares = corpus(df).str.contains(words_of(query), regex=True)

    print(f"\n{len(df):,} papers · {query!r}\n")
    for i in scores.argsort()[::-1][:top]:
        row = df.iloc[i]
        borrowed = "" if shares.iloc[i] else "   (shares no word with your query)"
        print(f"  {scores[i]:.2f}  {row['year']}  {row['title']}{borrowed}")
        print(f"              {row['url']}")

    print("\nclosest in each year:")
    for year, group in df.assign(score=scores).groupby("year"):
        row = group.nlargest(1, "score").iloc[0]
        print(f"  {row['score']:.2f}  {year}  {row['title']}")


def report(pct: pd.DataFrame, counts: pd.Series) -> None:
    top = movers(pct)
    print(f"\nPapers per year:\n{counts.to_string()}")
    print("\nShare of each year's papers, by discovered topic (%):")
    print(top.round(2).T.to_string())


def plot(pct: pd.DataFrame, out: Path, papers: int) -> None:
    import matplotlib

    matplotlib.use("Agg")  # before pyplot, so no GUI backend is pulled in
    import matplotlib.pyplot as plt

    top = movers(pct)
    fig, ax = plt.subplots(figsize=(11, 7), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    years = list(top.index)
    # Eight series ending within a whisker of each other cannot carry end labels: they would
    # sit nowhere near their own line. A legend says which is which without guessing.
    handles = []
    for i, column in enumerate(top.columns):
        ax.plot(years, top[column], color=PALETTE[i], linewidth=2, marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        first, last = top[column].iloc[0], top[column].iloc[-1]
        handles.append(plt.Line2D([], [], marker="o", linestyle="", markersize=7,
                                  color=PALETTE[i], label=f"{column}   {first:.2f} → {last:.2f}%"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=2,
              frameon=False, fontsize=9, labelcolor=MUTED, handletextpad=0.4, columnspacing=2)
    ax.set_xticks(years)
    ax.set_xlim(years[0] - 0.08, years[-1] + 0.08)
    ax.set_ylim(0, max(top.max()) * 1.10)
    ax.set_ylabel("% of that year's accepted papers", color=MUTED, fontsize=9)
    ax.tick_params(colors=MUTED, length=0, labelsize=9.5)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#d9d8d3")
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.text(0.012, 0.955, "What NeurIPS started working on", fontsize=15, color=INK, va="top")
    fig.text(0.012, 0.906, "Topics found by clustering each year's paper embeddings, named by "
             "the phrases they over-use", fontsize=9.5, color=MUTED, va="top")
    fig.text(0.008, 0.012, f"{papers:,} papers read from neurips.cc/virtual, a site that renders "
             "nothing without JavaScript. Scraped with Lightpanda.",
             fontsize=8, color=MUTED, va="bottom")
    fig.tight_layout(rect=(0, 0.04, 1, 0.885))
    fig.savefig(out, dpi=120)
    print(f"\nChart written to {out}")


PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7", "#e34948", "#e87ba4", "#008300"]
INK, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e8e7e2"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("query", nargs="*", help="what to look for; omit for an interactive prompt")
    parser.add_argument("--years", nargs="+", type=int, default=YEARS, metavar="YEAR",
                        help=f"NeurIPS years to read (default: {YEARS[0]}-{YEARS[-1]})")
    parser.add_argument("--top", type=int, default=8, metavar="N", help="results to show (default: 8)")
    parser.add_argument("--threshold", type=float, default=0.875, metavar="S",
                        help="similarity two papers need to be linked (default: 0.875)")
    parser.add_argument("--min-size", type=int, default=60, metavar="N",
                        help="smallest cluster to report (default: 60)")
    parser.add_argument("--cache", type=Path, default=default_cache(), metavar="PATH",
                        help="where to keep embeddings between runs")
    parser.add_argument("--csv", type=Path, metavar="PATH", help="write the scraped papers to a CSV file")
    parser.add_argument("--from-csv", type=Path, metavar="PATH",
                        help="skip the browser, search a saved CSV")
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # keep stdout and the stderr progress in order

    if args.from_csv:
        df = pd.read_csv(args.from_csv).fillna({"abstract": ""})
    else:
        print(f"Reading NeurIPS {', '.join(map(str, args.years))} ...", file=sys.stderr)
        df = asyncio.run(scrape(args.years))
    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"Papers written to {args.csv}", file=sys.stderr)

    key = os.environ.get("GOOGLE_API_KEY")
    print(f"Embedding {len(df):,} papers with {GEMINI if key else 'bge-small-en-v1.5'} ...",
          file=sys.stderr)
    X = embed_corpus(df, key, args.cache)

    if args.query:  # the embeddings are paid for, so searching them is free
        show(df, X, key, " ".join(args.query), args.top)
    else:
        pct = topics(df, X, args.threshold, args.min_size)
        report(pct, df.groupby("year").size())
        plot(pct, Path(__file__).with_name("neurips_topics.png"), len(df))
