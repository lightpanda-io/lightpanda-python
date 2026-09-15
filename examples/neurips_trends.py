# /// script
# requires-python = ">=3.10"
# dependencies = ["lightpanda", "pandas", "matplotlib", "numpy", "requests", "fastembed",
#                 "scikit-learn"]
# ///
"""What NeurIPS started talking about: five years of abstracts from a JS-only site.

https://neurips.cc/virtual/2025/papers.html renders nothing without JavaScript.
``requests`` gets 800 KB of scripts around an empty card list and a
``<noscript>`` apology; BeautifulSoup finds 0 papers. The page fetches a 27 MB
index, keeps every paper in a JavaScript array, and renders at most 400 cards
from it, so reading the DOM would still miss most of the conference.

Lightpanda runs the page and reads that array instead, one browser per year and
five years at once: 19,219 papers with abstracts.

The terms are not supplied. Every 1-to-3 word phrase is counted and ranked by
how much its share grew or shrank, and the phrases that peak in a single year
name what happened that year. The chart stays on those counts because a count
cannot be wrong about what it measures.

The papers are then embedded and ranked against the term that grew most, which
surfaces the ones that read like it while using none of its words. Embeddings
come from Gemini when GOOGLE_API_KEY is set and from bge-small-en-v1.5 locally
otherwise, cached by paper URL. This runs by default when the key or the cache
makes it cheap; ``--semantic`` forces it and ``--no-semantic`` skips it.

Run:  uv run examples/neurips_trends.py
      uv run examples/neurips_trends.py --semantic --csv papers.csv
"""
import argparse
import asyncio
import functools
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from lightpanda import AsyncBrowser

YEARS = [2021, 2022, 2023, 2024, 2025]

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7", "#e34948", "#e87ba4", "#008300"]
YEARLIKE = re.compile(r"\b(19|20)\d\d\b|^\d+$")  # "2021" is not a topic
INK, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e8e7e2"

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


def discover(df: pd.DataFrame, n: int = 4, floor: float = 0.5) -> pd.DataFrame:
    """The phrases whose share grew and shrank most.

    Ranked by ratio, not absolute change: a topic multiplies from nothing, while writing
    style drifts smoothly across every paper. Near-duplicate names for one topic are
    collapsed by how often they co-occur.
    """
    from sklearn.feature_extraction.text import CountVectorizer

    vec = CountVectorizer(ngram_range=(1, 3), min_df=30, max_df=0.4, stop_words="english",
                          binary=True)
    X = vec.fit_transform(corpus(df).str.lower())
    terms = np.array(vec.get_feature_names_out())
    # Topics reach titles and prose does not, so this keeps "3d gaussian splatting" and drops
    # "advancements". It also drops bare model names that only ever appear in abstracts.
    in_title = (np.asarray(vec.transform(df["title"].str.lower()).sum(axis=0)).ravel()
                / np.maximum(np.asarray(X.sum(axis=0)).ravel(), 1))
    years = sorted(df["year"].unique())
    share = np.vstack([np.asarray(X[(df["year"] == y).to_numpy()].mean(axis=0)).ravel()
                       for y in years]) * 100
    seen = np.asarray(X.sum(axis=0)).ravel()

    def same_topic(i: int, j: int) -> bool:
        both = X[:, i].multiply(X[:, j]).sum()
        return (terms[i] in terms[j] or terms[j] in terms[i]
                or both / (seen[i] + seen[j] - both) > 0.25)

    def take(order: np.ndarray, row: int) -> list[int]:
        kept: list[int] = []
        for i in order:
            if YEARLIKE.search(terms[i]) or share[row, i] < floor or in_title[i] < 0.04:
                continue
            if any(same_topic(i, k) for k in kept):
                continue
            kept.append(int(i))
            if len(kept) == n:
                return kept
        return kept

    ratio = (share[-1] + 0.15) / (share[0] + 0.15)
    risers = take(np.argsort(ratio)[::-1], -1)
    picked = risers + take(np.argsort(ratio), 0)
    print(f"\nDiscovered from {X.shape[1]} candidate phrases, ranked by growth.", file=sys.stderr)

    # A phrase that peaks in one year is usually a real event in that year.
    print("\nWhat peaked each year:")
    peaks = share.argmax(axis=0)
    for row, year in enumerate(years):
        lift = (share[row] + 0.2) / (np.delete(share, row, axis=0).mean(axis=0) + 0.2)
        lift[peaks != row] = 0
        names = [terms[i] for i in take(np.argsort(lift)[::-1], row)]
        print(f"  {year}: " + ", ".join(names))

    out = pd.DataFrame({terms[i]: share[:, i] for i in picked}, index=pd.Index(years, name="year"))
    # What --semantic chases: the phrase that grew most, described by its close company.
    top = risers[0]
    inside = np.asarray(X[X[:, top].toarray().ravel() > 0].mean(axis=0)).ravel()
    company = (inside + 0.002) / (np.asarray(X.mean(axis=0)).ravel() + 0.002)
    company[[i for i in range(len(terms)) if in_title[i] < 0.04]] = 0
    neighbours = [terms[i] for i in np.argsort(company)[::-1][:8]]
    # "Uses the vocabulary" means any of the discovered variants, not just the one token:
    # a paper spelling out "large language models" is not avoiding the words.
    vocabulary = "|".join(re.escape(t) for t in dict.fromkeys([terms[top], *neighbours]))
    out.attrs["focus"] = (terms[top], vocabulary, ", ".join(neighbours))
    return out


def report(df: pd.DataFrame, pct: pd.DataFrame) -> None:
    counts = df.groupby("year").size()
    print("\nPapers per year:\n" + counts.to_string())
    print("\nShare of papers whose title or abstract mentions each term (%):")
    print(pct.round(1).T.to_string())
    first, last = pct.index[0], pct.index[-1]
    change = ((pct.loc[last] - pct.loc[first]) / pct.loc[first].clip(lower=0.05)).sort_values()
    print(f"\nBiggest movers, {first} to {last}:")
    for name in list(change.index[:2]) + list(change.index[-2:][::-1]):
        a, b = pct.loc[first, name], pct.loc[last, name]
        print(f"  {name:24s} {a:5.1f}% -> {b:5.1f}%   "
              f"({int(round(b / 100 * counts[last]))} of {counts[last]} papers in {last})")


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

    done = 0
    with cf.ThreadPoolExecutor(max_workers=6) as pool:
        out = []
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
        print(f"  (parallel encoding unavailable: {type(exc).__name__}; using one core)", file=sys.stderr)
        return np.array(list(local_model().embed(texts, batch_size=256)))


def unit(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def default_cache() -> Path:
    base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return base / "neurips_trends" / "embeddings.npz"


def cached_already(df: pd.DataFrame, cache: Path) -> bool:
    """True when the cache can answer for every paper without embedding anything new."""
    if not cache.exists():
        return False
    with np.load(cache, allow_pickle=False) as z:
        return z["model"].item() == "bge-small-en-v1.5" and df["url"].isin(z["keys"]).all()


def embed_all(df: pd.DataFrame, key: str | None, cache: Path) -> np.ndarray:
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
            print(f"  {len(todo)} papers to embed locally -- this takes a while the first time; "
                  f"the cache at {cache} makes later runs instant.", file=sys.stderr)
        fresh = embed_gemini([texts[i] for i in todo], key) if key else embed_local([texts[i] for i in todo])
        for i, vector in zip(todo, unit(fresh)):
            store[keys[i]] = vector.astype(np.float16)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, model=model, keys=np.array(list(store)),
                            vectors=np.stack(list(store.values())))
        print(f"  cached {len(store)} embeddings in {cache}", file=sys.stderr)
    else:
        print(f"  all {len(keys)} embeddings came from {cache}", file=sys.stderr)
    return np.stack([store[k] for k in keys]).astype(np.float32)


def semantic_report(df: pd.DataFrame, key: str | None, cache: Path,
                    focus: tuple[str, str, str]) -> None:
    """Rank every paper by meaning.

    Only a ranking: a share per year would need a threshold, and embeddings order documents
    far more reliably than they decide whether one clears a bar.
    """
    term, pattern, concept = focus
    model = GEMINI if key else "bge-small-en-v1.5, locally"
    print(f"\nRanking all {len(df)} papers against {term!r} with {model}.", file=sys.stderr)
    print(f"  query built from the corpus: {concept}", file=sys.stderr)
    X = embed_all(df, key, cache)
    q = unit(embed_gemini([concept], key) if key else embed_local([concept]))[0]
    ranked = df.assign(similarity=X @ q,
                       phrase=corpus(df).str.contains(pattern, regex=True, case=False))

    print(f"\nThe paper each year that reads most like {term!r}:")
    for year, group in ranked.groupby("year"):
        row = group.nlargest(1, "similarity").iloc[0]
        says = "uses the words" if row["phrase"] else "never uses them"
        print(f"  {year}  {row['similarity']:.2f}  {row['title']}\n            {says} · {row['url']}")

    missed = ranked[~ranked["phrase"]].nlargest(6, "similarity")
    print(f"\nRanked highest among papers that never use any of those words:")
    for row in (missed.iloc[i] for i in range(len(missed))):
        print(f"  [{row['year']}] {row['similarity']:.2f}  {row['title']}\n            {row['url']}")


def spread(values: list[float], gap: float) -> list[float]:
    """Nudge end-of-line labels apart so close series stay readable."""
    y = list(values)
    for a, b in zip(order := sorted(range(len(y)), key=lambda i: y[i]), order[1:]):
        y[b] = max(y[b], y[a] + gap)
    return y


def plot(pct: pd.DataFrame, out: Path, papers: int) -> None:
    import matplotlib

    matplotlib.use("Agg")  # before pyplot, so no GUI backend is pulled in
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    years = list(pct.index)
    ends, names = list(pct.iloc[-1]), list(pct.columns)
    top = max(max(pct.max()), max(ends))
    labels = spread(ends, gap=top * 0.052)  # keep end labels legible at any y-range
    for i, name in enumerate(pct.columns):
        ax.plot(years, pct[name], color=PALETTE[i], linewidth=2, marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
    for i, name in enumerate(names):
        ax.text(years[-1] + 0.08, labels[i], f" {name}  {ends[i]:.1f}%",
                color=PALETTE[i % len(PALETTE)], fontsize=9.5, va="center")
    ax.set_xticks(years)
    ax.set_xlim(years[0] - 0.1, years[-1] + 2.5)
    ax.set_ylim(0, top * 1.12)
    ax.set_ylabel("% of accepted papers", color=MUTED, fontsize=9)
    ax.tick_params(colors=MUTED, length=0, labelsize=9.5)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#d9d8d3")
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.text(0.012, 0.955, "What NeurIPS talks about", fontsize=15, color=INK, va="top")
    fig.text(0.012, 0.906, "Share of accepted papers whose title or abstract mentions each term",
             fontsize=9.5, color=MUTED, va="top")
    fig.text(0.008, 0.012, f"{papers:,} papers read from neurips.cc/virtual, a site that renders "
             "nothing without JavaScript. Scraped with Lightpanda.",
             fontsize=8, color=MUTED, va="bottom")
    fig.tight_layout(rect=(0, 0.03, 1, 0.885))
    fig.savefig(out, dpi=120)
    print(f"\nChart written to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--years", nargs="+", type=int, default=YEARS, metavar="YEAR",
                        help=f"NeurIPS years to read (default: {YEARS[0]}-{YEARS[-1]})")
    parser.add_argument("--semantic", action=argparse.BooleanOptionalAction, default=None,
                        help="rank every paper by meaning (default: when it is cheap to do)")
    parser.add_argument("--cache", type=Path, default=default_cache(), metavar="PATH",
                        help="where to keep embeddings between runs")
    parser.add_argument("--csv", type=Path, metavar="PATH", help="write the scraped papers to a CSV file")
    parser.add_argument("--from-csv", type=Path, metavar="PATH",
                        help="skip the browser, analyse a saved CSV")
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

    pct = discover(df)
    report(df, pct)

    key = os.environ.get("GOOGLE_API_KEY")
    if args.semantic is not False:
        # On by default when it costs little: an API key, or a cache that already covers these
        # papers. Encoding 19k abstracts locally from cold takes far too long to do unasked.
        if args.semantic or key or cached_already(df, args.cache):
            semantic_report(df, key, args.cache, pct.attrs["focus"])
        else:
            print("\nSkipping the ranking: set GOOGLE_API_KEY, or pass --semantic to encode "
                  "locally (slow the first time, cached after).", file=sys.stderr)
    plot(pct, Path(__file__).with_name("neurips_trends.png"), len(df))
