# /// script
# requires-python = ">=3.10"
# dependencies = ["lightpanda", "pandas", "matplotlib", "selenium", "requests", "beautifulsoup4", "psutil"]
# ///
"""The same scrape three ways: requests, Selenium + headless Chrome, Lightpanda.

Each leg loads the ten pages of https://quotes.toscrape.com/js/ (quotes are
rendered by JavaScript) and collects (text, author, tags) records. We report
how many quotes each tool actually saw, wall time including browser startup,
and peak resident memory summed over every child process (Chrome's whole
process tree for Selenium, the single sidecar for Lightpanda).

Run:  uv run examples/compare.py [--repeat N]
With --repeat, the legs are interleaved N times and the table reports the
median (and min–max range) per tool. Selenium Manager downloads a chromedriver
on its first run — use --repeat or run twice for a fair timing. Needs Chrome
installed; the Selenium leg is skipped otherwise.
"""

import argparse
import threading
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import psutil

URLS = [f"https://quotes.toscrape.com/js/page/{n}/" for n in range(1, 11)]


def with_requests() -> int:
    import requests
    from bs4 import BeautifulSoup

    records, http = [], requests.Session()
    for url in URLS:
        soup = BeautifulSoup(http.get(url, timeout=30).text, "html.parser")
        for q in soup.select("div.quote"):  # never matches: the DOM is built by JS
            records.append((q.select_one(".text").text, q.select_one(".author").text,
                            [t.text for t in q.select(".tag")]))
    return len(records)


def with_selenium() -> int:
    from selenium import webdriver
    from selenium.webdriver.common.by import By

    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    driver = webdriver.Chrome(options=options)
    try:
        records = []
        for url in URLS:
            driver.get(url)
            for q in driver.find_elements(By.CSS_SELECTOR, ".quote"):
                records.append((q.find_element(By.CSS_SELECTOR, ".text").text,
                                q.find_element(By.CSS_SELECTOR, ".author").text,
                                [t.text for t in q.find_elements(By.CSS_SELECTOR, ".tag")]))
        return len(records)
    finally:
        driver.quit()


def with_lightpanda() -> int:
    from lightpanda import Browser

    schema = {"quotes": [{"selector": ".quote",
                          "fields": {"text": ".text", "author": ".author", "tags": [".tag"]}}]}
    records = []
    with Browser() as browser, browser.new_session() as page:
        for url in URLS:
            page.goto(url=url)
            records.extend(page.extract(schema=schema)["quotes"])
    return len(records)


def measure(fn) -> dict:
    """Run fn, sampling the RSS of all child processes every 50 ms."""
    me, peak, stop = psutil.Process(), 0, threading.Event()

    def sample():
        nonlocal peak
        while not stop.is_set():
            rss = 0
            for child in me.children(recursive=True):
                try:
                    rss += child.memory_info().rss
                except psutil.Error:
                    pass
            peak = max(peak, rss)
            stop.wait(0.05)

    sampler = threading.Thread(target=sample, daemon=True)
    start = time.perf_counter()
    sampler.start()
    try:
        quotes = fn()
    finally:
        stop.set()
        sampler.join()
    return {"quotes": quotes, "seconds": round(time.perf_counter() - start, 2),
            "peak_mb": round(peak / 2**20, 1)}


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    """Per tool: quotes found, median of each measure, and the seconds range."""
    by_tool = runs.groupby("tool", sort=False)
    df = by_tool[["quotes", "seconds", "peak_mb"]].median()
    df["quotes"] = df["quotes"].astype(int)
    df["seconds_min"] = by_tool["seconds"].min()
    df["seconds_max"] = by_tool["seconds"].max()
    df["runs"] = by_tool.size()
    return df


def plot(df: pd.DataFrame, out: Path) -> None:
    matplotlib.use("Agg")
    colors = {"requests": "#2a78d6", "selenium": "#eb6834", "lightpanda": "#1baf7a"}
    repeats = int(df["runs"].max())
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), facecolor="#fcfcfb")
    for ax, col, title in ((axes[0], "seconds", "Wall time (s), 10 pages incl. startup"),
                           (axes[1], "peak_mb", "Peak memory of child processes (MB)")):
        labels = [f"{tool}\n{quotes} quotes" for tool, quotes in df["quotes"].items()]
        ax.bar(labels, df[col], color=[colors[t] for t in df.index], width=0.55)
        tops = df[col]
        if col == "seconds" and repeats > 1:  # min–max whiskers around the median
            ax.errorbar(labels, df[col], fmt="none", ecolor="#0b0b0b", capsize=4, linewidth=1,
                        yerr=[df[col] - df["seconds_min"], df["seconds_max"] - df[col]])
            tops = df["seconds_max"]
        for x, (value, top) in enumerate(zip(df[col], tops)):  # label above bar or whisker
            ax.text(x, top + df[col].max() * 0.03, f"{value:g}", ha="center", va="bottom",
                    color="#52514e", fontsize=9)
        ax.set_title(title, loc="left", fontsize=11, color="#0b0b0b")
        ax.set_facecolor("#fcfcfb")
        ax.set_ylim(0, df[col].max() * 1.18)
        ax.tick_params(colors="#52514e", length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#d9d8d3")
        ax.yaxis.grid(True, color="#e8e7e2", linewidth=0.8)
        ax.set_axisbelow(True)
    runs = f"median of {repeats} runs, whiskers = min–max" if repeats > 1 else "single run"
    fig.suptitle(f"Ten pages of quotes.toscrape.com/js (rendered by JavaScript) — {runs}",
                 x=0.01, ha="left", fontsize=10, color="#52514e")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"\nChart written to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repeat", type=int, default=1, metavar="N",
                        help="run every leg N times, interleaved; report medians")
    repeat = parser.parse_args().repeat

    legs = (("requests", with_requests), ("selenium", with_selenium),
            ("lightpanda", with_lightpanda))
    runs = []
    for i in range(repeat):
        for name, fn in legs:
            print(f"[{i + 1}/{repeat}] {name:11s}...", end=" ", flush=True)
            try:
                result = measure(fn)
            except Exception as exc:  # e.g. no Chrome for the Selenium leg
                print(f"skipped ({type(exc).__name__}: {str(exc).splitlines()[0]})")
                continue
            print(result)
            runs.append({"tool": name, **result})

    df = summarize(pd.DataFrame(runs))
    print("\n" + df.to_string())
    plot(df, Path(__file__).with_name("compare.png"))
