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
from it -- so even a scraper that could read the DOM would see a fraction.

Lightpanda runs the page and hands that array straight to Python, one page load
per year, five years at once. 19,219 papers with abstracts.

Nothing then tells it what to look for. It counts every 1-to-3 word phrase and
ranks them by how much their share grew, which is enough to find that every
phrase that grew is about language models in some form: "llms" goes from 0 to
1,012 of 5858 papers, while "deep neural" falls from 6.3% and crosses it on the
way down. ``--curated`` swaps in six hand-written terms instead, which makes a
prettier chart and is kept as a warning about picking terms you already expect.

``--semantic`` adds the interesting half: it embeds a sample of each year and
estimates the same share by meaning instead of by phrase, chasing whichever
term grew most and describing it with the phrases that keep it company. The two
curves disagree most in the early years, and the gap is the point -- papers
that were about language models before the field settled on the words for it.

It uses Gemini when GOOGLE_API_KEY is set and bge-small-en-v1.5 locally (ONNX,
no torch, no network) otherwise. The model matters: ranking the true phrase
matches to the top scores 0.96 AUC with Gemini, 0.91 with bge-small and 0.79
with a static model, which is the difference between a curve and a flat line.

Run:  uv run examples/neurips_trends.py
      uv run examples/neurips_trends.py --semantic --csv papers.csv
"""

import argparse
import asyncio
import functools
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from lightpanda import AsyncBrowser

YEARS = [2021, 2022, 2023, 2024, 2025]

# Literal phrases, matched against title + abstract: what the authors wrote, not
# what a model thinks they meant.
TERMS = {
    "large language models": r"\blarge language model|\bLLMs?\b|\bGPT-|\bChatGPT\b",
    "diffusion models": r"\bdiffusion model|\bdenoising diffusion|\bscore-based generat",
    "in-context learning": r"\bin-context learning\b|\bchain[- ]of[- ]thought\b|\bprompting\b",
    "RLHF & alignment": r"\bRLHF\b|human feedback|\bpreference optimi|\bDPO\b",
    "graph neural networks": r"\bgraph neural network|\bGNNs?\b",
    "GANs": r"\bGANs?\b|\bgenerative adversarial",
}
# What "large language models" means, for the semantic pass.
CONCEPT = "large language models, LLMs, GPT, instruction tuning, in-context learning"

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7", "#e34948", "#e87ba4", "#008300"]
YEARLIKE = re.compile(r"\b(19|20)\d\d\b|^\d+$")  # "2021" is not a topic
INK, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e8e7e2"

# The page keeps every paper here; only 400 of them ever reach the DOM.
PULL = "return allPapers.map(p => [p.title, p.abstract || '', p.url, p.id])"
EMBED_URL = ("https://generativelanguage.googleapis.com/v1beta/models/"
             "gemini-embedding-001:batchEmbedContents")


async def scrape_year(year: int) -> list[dict]:
    """Every paper of one year, read from the page's own data rather than its cards."""
    # One browser process per year. Several sessions inside a single process contend badly on
    # this workload -- parsing a 27 MB index in each -- and most of them fail; separate processes
    # run all five years in about the time one of them takes.
    # --http-timeout is a deadline for the whole response, not an idle timeout, so the index
    # trips the 15 s default even at full speed.
    async with AsyncBrowser(args=["--http-timeout", "180000"], max_concurrency=1) as browser:
        async with browser.session() as page:
            await page.goto(url=f"https://neurips.cc/virtual/{year}/papers.html", timeout=180_000)
            # `allPapers` is filled just before the first cards render, so this waits for the
            # index without waiting for DOM work whose output we are not going to read.
            await page.wait_for_script(
                script="typeof allPapers !== 'undefined' && allPapers.length > 0", timeout=120_000)
            rows = await page.evaluate(script=PULL, timeout=180_000)
    print(f"  {year}: {len(rows)} papers", file=sys.stderr)
    # The array holds site-relative paths, and the oldest years carry none at all -- their
    # cards are linked by id instead, which is the same URL the site would build.
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
    """Let the corpus nominate the terms: the phrases whose share grew and shrank most.

    Ranking by *ratio* rather than by absolute change is what separates topics from prose:
    a new topic multiplies from nothing, while writing style drifts smoothly across every
    paper. Near-duplicate names for one topic are collapsed by how often they co-occur.
    """
    from sklearn.feature_extraction.text import CountVectorizer

    vec = CountVectorizer(ngram_range=(1, 3), min_df=30, max_df=0.4, stop_words="english",
                          binary=True)
    X = vec.fit_transform(corpus(df).str.lower())
    terms = np.array(vec.get_feature_names_out())
    # Authors put topics in titles and never put prose there, so the share of a phrase's papers
    # that carry it in the title separates "3d gaussian splatting" from "advancements". It is a
    # property of the corpus, not a hand-written stoplist -- the cost is bare model names like
    # "qwen2", which appear in abstracts as baselines and rarely in a title.
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
    # What --semantic should chase: the phrase that grew most, described by the phrases that
    # keep it company rather than by anything written here.
    top = risers[0]
    inside = np.asarray(X[X[:, top].toarray().ravel() > 0].mean(axis=0)).ravel()
    company = (inside + 0.002) / (np.asarray(X.mean(axis=0)).ravel() + 0.002)
    company[[i for i in range(len(terms)) if in_title[i] < 0.04]] = 0
    neighbours = [terms[i] for i in np.argsort(company)[::-1][:8]]
    out.attrs["focus"] = (terms[top], re.escape(terms[top]), ", ".join(neighbours))
    return out


def phrase_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Share of each year's papers, in percent, whose text matches each phrase."""
    text = corpus(df)
    hits = pd.DataFrame({name: text.str.contains(pattern, regex=True, case=False)
                         for name, pattern in TERMS.items()})
    out = hits.groupby(df["year"]).mean().mul(100)
    out.attrs["focus"] = ("large language models", TERMS["large language models"], CONCEPT)
    return out


def embed_gemini(texts: list[str], key: str) -> np.ndarray:
    """Unit-length Gemini embeddings, 100 texts per request."""
    import requests

    out: list[list[float]] = []
    for i in range(0, len(texts), 100):
        batch = [{"model": "models/gemini-embedding-001", "taskType": "SEMANTIC_SIMILARITY",
                  "outputDimensionality": 768, "content": {"parts": [{"text": t[:1500]}]}}
                 for t in texts[i:i + 100]]
        r = requests.post(EMBED_URL, headers={"x-goog-api-key": key},
                          json={"requests": batch}, timeout=180)
        r.raise_for_status()
        out += [e["values"] for e in r.json()["embeddings"]]
        print(f"  embedded {len(out)}/{len(texts)}", end="\r", file=sys.stderr)
    return unit(np.array(out))


@functools.cache
def local_model():
    from fastembed import TextEmbedding

    return TextEmbedding(model_name="BAAI/bge-small-en-v1.5")


def embed_local(texts: list[str]) -> np.ndarray:
    """Unit-length bge-small embeddings: 130 MB of ONNX, no torch, no key, no network."""
    return unit(np.array(list(local_model().embed([t[:1500] for t in texts]))))


def unit(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def semantic_trend(df: pd.DataFrame, key: str | None, per_year: int,
                   focus: tuple[str, str, str]) -> tuple[pd.Series, pd.DataFrame]:
    """Estimate the same share by meaning, on a sample, and show what the phrase missed."""
    sample = pd.concat([g.sample(min(per_year, len(g)), random_state=0)
                        for _, g in df.groupby("year")])
    # Ranking the real phrase matches to the top scores 0.96 AUC with Gemini and 0.91 with
    # bge-small locally; a static model manages 0.79, which is not enough to see the trend.
    term, pattern, concept = focus
    which = "gemini-embedding-001" if key else "bge-small-en-v1.5, locally (slower)"
    print(f"\nEmbedding {len(sample)} papers with {which} to find {term!r} by meaning "
          f"({concept}) ...", file=sys.stderr)
    texts = corpus(sample).tolist()
    X, q = ((embed_gemini(texts, key), embed_gemini([concept], key)[0]) if key
            else (embed_local(texts), embed_local([concept])[0]))
    sample = sample.assign(similarity=X @ q,
                           phrase=corpus(sample).str.contains(pattern, regex=True, case=False))
    # One cutoff for all years, set so the overall rate matches the phrase rate. The shape across
    # years, and which papers swap in, are what the comparison is about.
    cutoff = sample["similarity"].quantile(1 - sample["phrase"].mean())
    sample = sample.assign(semantic=sample["similarity"] > cutoff)
    return sample.groupby("year")["semantic"].mean().mul(100), sample


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


def report_semantic(sem: pd.Series, sample: pd.DataFrame, pct: pd.DataFrame, term: str) -> None:
    print(f"\n{term!r}, counted two ways (%):")
    both = pd.DataFrame({"phrase": pct[term], "semantic (sampled)": sem}).round(1)
    print(both.to_string())
    missed = sample[sample["semantic"] & ~sample["phrase"]].sort_values("similarity", ascending=False)
    early = missed[missed["year"] <= sample["year"].min() + 1]
    if len(early):
        print(f"\nRanked as {term!r} work without using the phrase ({len(missed)} in the "
              f"sample, {len(early)} of them before the term caught on):")
        for row in (early.iloc[i] for i in range(min(3, len(early)))):
            print(f"  [{row['year']}] {row['title']}"
                  + (f"\n        {row['url']}" if isinstance(row["url"], str) and row["url"] else ""))


def spread(values: list[float], gap: float) -> list[float]:
    """Nudge end-of-line labels apart so close series stay readable."""
    y = list(values)
    for a, b in zip(order := sorted(range(len(y)), key=lambda i: y[i]), order[1:]):
        y[b] = max(y[b], y[a] + gap)
    return y


def plot(pct: pd.DataFrame, sem: pd.Series | None, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")  # before pyplot, so no GUI backend is pulled in
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    years = list(pct.index)
    ends, names = list(pct.iloc[-1]), list(pct.columns)
    if sem is not None:
        ends.append(sem.iloc[-1])
        names.append(f"{pct.attrs['focus'][0]}, by meaning")
    top = max(max(pct.max()), max(ends))
    labels = spread(ends, gap=top * 0.052)  # keep end labels legible at any y-range
    for i, name in enumerate(pct.columns):
        ax.plot(years, pct[name], color=PALETTE[i], linewidth=2, marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
    if sem is not None:
        ax.plot(list(sem.index), sem, color=PALETTE[0], linewidth=2, linestyle=(0, (4, 2)),
                marker="o", markersize=4, markerfacecolor=SURFACE)
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
    fig.text(0.012, 0.906, "Share of accepted papers whose title or abstract mentions each term"
             + (", dashed: counted by meaning instead" if sem is not None else ""),
             fontsize=9.5, color=MUTED, va="top")
    fig.text(0.008, 0.012, "19,219 papers read from neurips.cc/virtual, a site that renders nothing "
             "without JavaScript. Scraped with Lightpanda.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.885))
    fig.savefig(out, dpi=120)
    print(f"\nChart written to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--years", nargs="+", type=int, default=YEARS, metavar="YEAR",
                        help=f"NeurIPS years to read (default: {YEARS[0]}-{YEARS[-1]})")
    parser.add_argument("--curated", action="store_true",
                        help="use the built-in hand-picked terms instead of discovering them")
    parser.add_argument("--semantic", action="store_true",
                        help="also count by meaning; uses GOOGLE_API_KEY if set, else a local model")
    parser.add_argument("--sample", type=int, default=400, metavar="N",
                        help="papers per year to embed for --semantic (default: 500)")
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

    pct = phrase_trend(df) if args.curated else discover(df)
    report(df, pct)

    sem = None
    if args.semantic:
        focus = pct.attrs["focus"]
        sem, sample = semantic_trend(df, os.environ.get("GOOGLE_API_KEY"), args.sample, focus)
        report_semantic(sem, sample, pct, focus[0])
    plot(pct, sem, Path(__file__).with_name("neurips_trends.png"))
