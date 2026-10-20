# /// script
# requires-python = ">=3.10"
# dependencies = ["scikit-learn>=1.5", "scipy", "numpy", "lightpanda"]
#
# [tool.uv.sources]
# lightpanda = { path = "../..", editable = true }
# ///
"""Find more pages of the rare presets without asking Jev about every page.

collect.py asks Jev about every page it loads, and most pages are ordinary
content: the rare presets (paywalls, login walls, loading shells, empty
searches, unsupported browsers, rate limits) stay a few dozen examples each.
This loads many more pages but labels only the ones worth it:

  urls    write data/candidates.tsv: more Tranco homepages, a nonsense
          search and a members-only path on some of them, and article links
          from the paywalled news sites (articles.py's)
  load    load each candidate and keep what classify sends Jev
          (`includeInput`) -> data/candidates.jsonl. This is NOT free: an
          empty questions object means every preset to classify, so each
          page costs one Jev call (its answers are kept as `jev_load`).
          Getting the input without Jev needs a browser change
  pick    train the linear model on the labelled pages and score the
          candidates; take the top of each rare preset, then the pages the
          model is least sure of, at most --per-site per site, up to
          --budget pages -> data/picked.jsonl
  label   ask Jev about the picked pages, JUDGES calls each in opposite
          orders as collect.py does, straight to TypeSafe's API on the stored
          input (the browser sends the same state, and its answers on it come
          back the same), and append them to pages.jsonl as collect.py rows

Needs TYPESAFE_API_KEY for `label`. Run:
    uv run research/page_classifier/mine.py urls && \\
    uv run research/page_classifier/mine.py load && \\
    uv run research/page_classifier/mine.py pick --budget 1500 && \\
    uv run research/page_classifier/mine.py label
"""

import argparse
import asyncio
import hashlib
import json
import os
import random
import sys
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter, sleep

import numpy as np

from collect import DATA, JUDGES, REWORDED, camel, complete, snake
from features import PRESETS
from lightpanda import AsyncBrowser, LightpandaError
from seeds import NONSENSE, PAYWALLS, tranco

CANDIDATES = DATA / "candidates.jsonl"
PICKED = DATA / "picked.jsonl"

# Presets with too few positives, in the order pick fills them.
RARE = ["is_unsupported_browser", "is_rate_limited", "is_loading", "is_empty_catalog", "is_age_gate",
        "is_region_blocked", "is_login_wall", "is_paywall"]

# Paths a site keeps for members: a visitor who is not signed in gets a
# sign-in form, or a redirect to one.
MEMBER_PATHS = ["/account/orders", "/settings", "/dashboard", "/my-account/orders", "/profile/edit",
                "/account/settings", "/user/settings", "/checkout", "/messages", "/wishlist"]
# Search forms most sites answer at one of these.
SEARCH_PATHS = ["/?s={q}", "/search?q={q}", "/search?query={q}", "/search/?q={q}", "/catalogsearch/result/?q={q}"]

# Each preset's question as the browser asks it (Preset.description in the
# browser's src/browser/classify.zig): the API is asked what classify asks.
PRESET_QUESTIONS = {
    "is_blocked": "Is the page behind a CAPTCHA, Cloudflare challenge, or bot block?",
    "is_captcha": "Is the primary view obstructed by a CAPTCHA or human verification challenge?",
    "is_consent_wall": "Is the primary view obstructed by a cookie consent banner?",
    "is_empty_catalog": "Does the page say its search or filter found nothing (no results, no matches, 0 items)?",
    "is_error_page": "Does the page show an error, such as page not found or server error, instead of its content? A search or listing with no results is not an error.",
    "is_login_wall": "Does the page show a sign-in form or prompt instead of its content?",
    "is_paywall": "Is the content hidden or cut off behind a subscription or payment prompt?",
    "is_unsupported_browser": "Does the page say this browser is unsupported or out of date, or that JavaScript must be enabled, instead of showing its content?",
    "is_loading": "Is the main content still loading, shown only as spinners, skeleton placeholders or loading text?",
    "is_age_gate": "Is the primary view obstructed by an age check, asking the visitor to confirm they are an adult or to enter their date of birth before continuing?",
    "is_region_blocked": "Does the page say its content is not available in the visitor's country or region, instead of showing it?",
    "is_rate_limited": "Does the page say too many requests were sent and to slow down or try again later, instead of showing its content?",
}
assert list(PRESET_QUESTIONS) == PRESETS


def seen_seeds() -> set[str]:
    seen = set()
    for path in (DATA / "pages.jsonl", CANDIDATES):
        if path.exists():
            seen |= {json.loads(line)["seed"] for line in path.open()}
    return seen


def cmd_urls(args):
    import articles

    rng = random.Random(args.seed)
    known = tranco(3000)  # seeds.py's crawl
    rows = []
    for domain in tranco(3000 + args.skip + args.domains)[3000 + args.skip:]:
        rows.append(("mine-tranco", f"https://{domain}/"))
        if rng.random() < args.search:
            rows.append(("mine-search", f"https://{domain}{rng.choice(SEARCH_PATHS).format(q=NONSENSE)}"))
        if rng.random() < args.members:
            rows.append(("mine-members", f"https://{domain}{rng.choice(MEMBER_PATHS)}"))
    # Searches and members' pages on the sites already crawled are new pages too.
    for domain in rng.sample(known, args.known):
        rows.append(("mine-search", f"https://{domain}{rng.choice(SEARCH_PATHS).format(q=NONSENSE)}"))
        rows.append(("mine-members", f"https://{domain}{rng.choice(MEMBER_PATHS)}"))

    async def harvest():
        slots = asyncio.Semaphore(8)
        async with AsyncBrowser() as browser:
            return await asyncio.gather(*(articles.visit(browser, slots, home) for home in PAYWALLS))
    for links in asyncio.run(harvest()):
        rows += [("mine-article", url) for url in rng.sample(links, min(args.articles, len(links)))]

    seen = seen_seeds()
    rows = [r for r in dict.fromkeys(rows) if r[1] not in seen]
    rng.shuffle(rows)
    args.urls.write_text("".join(f"{s}\t{u}\n" for s, u in rows))
    print(f"{len(rows)} candidate urls -> {args.urls}: {dict(Counter(s for s, _ in rows))}")


async def load_one(browser, slots, out, source, url, timeout, counts):
    async with slots:
        try:
            async with await browser.new_session() as page:
                async def visit():
                    await page.goto(url=url, timeout=timeout)
                    return await page.call("classify", questions={}, includeInput=True)
                result = await asyncio.wait_for(visit(), timeout / 1000 * 2)
            local = result["input"]
            row = {"url": local["url"] or url, "title": local["title"] or None, "status": local["status"] or None,
                   "tree": local["tree"], "html": local["html"], "input": "classify", "jev_load": result["answers"]}
            counts["ok"] += 1
        except (LightpandaError, asyncio.TimeoutError, KeyError, TypeError) as err:
            row = {"error": str(err) or type(err).__name__}
            counts["failed"] += 1
        out.write(json.dumps({"seed": url, "source": source, **row}) + "\n")
        out.flush()
        done = counts["ok"] + counts["failed"]
        if done % 200 == 0:
            print(f"  {done}/{counts['total']}  ok {counts['ok']}  failed {counts['failed']}", flush=True)


def cmd_load(args):
    seen = seen_seeds()
    todo = [tuple(line.split("\t")) for line in args.urls.read_text().splitlines()]
    todo = [(s, u) for s, u in todo if u not in seen][: args.limit or None]
    counts = {"ok": 0, "failed": 0, "total": len(todo)}
    start = perf_counter()

    async def run():
        slots = asyncio.Semaphore(args.crawlers)
        with CANDIDATES.open("a") as out:
            async with AsyncBrowser(max_concurrency=args.crawlers * 2) as browser:
                await asyncio.gather(*(load_one(browser, slots, out, s, u, args.timeout, counts) for s, u in todo))
    asyncio.run(run())
    print(f"{counts['ok']} loaded, {counts['failed']} failed in {perf_counter() - start:.0f}s -> {CANDIDATES}")


def page_key(row: dict) -> tuple[str, str]:
    from train import site
    return site(row), hashlib.sha1(row["tree"].encode()).hexdigest()


def cmd_pick(args):
    from train import Linear, load

    rows = load(DATA / "pages.jsonl", 0.7, DATA / "gold.jsonl")
    known = {page_key(r) for r in rows}
    candidates, keys = [], set()
    for line in CANDIDATES.open():
        c = json.loads(line)
        if "error" in c or not c["tree"].strip():
            continue
        k = page_key(c)
        if k in known or k in keys:  # a redirect to a page already labelled, or loaded twice
            continue
        keys.add(k)
        candidates.append(c)
    start = perf_counter()
    scores = Linear(soft=True).fit(rows).predict(candidates)
    print(f"linear model on {len(rows)} pages scored {len(candidates)} candidates in {perf_counter() - start:.0f}s")

    from train import site
    picked, per_site = {}, Counter()

    def take(i, why):
        s = site(candidates[i])
        if i in picked or per_site[s] >= args.per_site:
            return False
        picked[i] = why
        per_site[s] += 1
        return True

    share = int(args.budget * args.rare / len(args.presets))
    for preset in args.presets:
        j = PRESETS.index(preset)
        taken = 0
        for i in np.argsort(-scores[:, j]):
            if scores[i, j] < args.floor or taken >= share:
                break
            taken += take(i, preset)
        print(f"  {preset:<24}{taken:>5} picked")
    # The rest: the pages the model is least sure of, all presets together.
    doubt = (scores * (1 - scores)).sum(axis=1)
    for i in np.argsort(-doubt):
        if len(picked) >= args.budget:
            break
        take(i, "unsure")
    with PICKED.open("w") as out:
        for i, why in picked.items():
            out.write(json.dumps({**{k: v for k, v in candidates[i].items() if not k.startswith("_")}, "picked": why, "picked_scores": np.round(scores[i], 3).tolist()}) + "\n")
    print(f"{len(picked)} pages -> {PICKED}: {dict(Counter(picked.values()))}")


def ask(key: str, row: dict, order: list[str]) -> dict:
    """One classify call on the stored page, the questions in `order`: the
    answers by snake_case key."""
    state = {"url": row["url"], "title": row.get("title")}
    if row.get("status"):
        state["status"] = row["status"]
    state["content"] = row["tree"]
    questions = {(k if k.startswith("re_") else camel(k)):
                 {"type": "noul", "instructions": REWORDED[k[3:]] if k.startswith("re_") else PRESET_QUESTIONS[k]}
                 for k in order}
    url = os.environ.get("TYPESAFE_BASE_URL", "https://api.typesafe.ai") + "/v1/systemone"
    body = json.dumps({"state": state, "model": "jev-latest", "questions": questions}).encode()
    request = urllib.request.Request(url, body, {"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                reply = json.load(response)
            return {snake(k): a.get("noul") for k, a in reply["answers"].items()}
        except urllib.error.HTTPError as err:
            if err.code not in (429, 500, 502, 503, 504) or attempt == 4:
                raise RuntimeError(f"HTTP {err.code}: {err.read()[:200]!r}") from err
        except (urllib.error.URLError, TimeoutError):
            if attempt == 4:
                raise
        sleep(2 ** attempt)


def judge(key: str, row: dict) -> dict:
    keys = [*PRESETS, *(f"re_{p}" for p in PRESETS)]
    order = random.sample(keys, len(keys))
    orders = [order if i % 2 == 0 else order[::-1] for i in range(JUDGES)]
    scores, seconds = [], []
    for keys_in_order in orders:
        start = perf_counter()
        scores.append(ask(key, row, keys_in_order))
        seconds.append(round(perf_counter() - start, 3))
    out = {k: v for k, v in row.items() if not k.startswith(("picked", "_", "jev_load"))}
    return out | {"jev": scores, "jev_orders": orders, "jev_seconds": seconds, "jev_direct": True}


def cmd_label(args):
    key = os.environ.get("TYPESAFE_API_KEY") or sys.exit("TYPESAFE_API_KEY is not set")
    pages = DATA / "pages.jsonl"
    done = {json.loads(line)["seed"] for line in pages.open()}
    todo = [r for r in map(json.loads, PICKED.open()) if r["seed"] not in done][: args.limit or None]
    print(f"asking Jev about {len(todo)} pages, {JUDGES} calls each")
    failed = 0
    with pages.open("a") as out, ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(judge, key, row): row for row in todo}
        for n, future in enumerate(as_completed(futures), 1):
            try:
                row = future.result()
                assert complete(row)
                out.write(json.dumps(row) + "\n")
                out.flush()
            except Exception as err:  # noqa: BLE001 - reported; the next run asks again
                failed += 1
                print(f"  {futures[future]['url'][:70]}: {err}", file=sys.stderr)
            if n % 100 == 0:
                print(f"  {n}/{len(todo)}, {failed} failed", flush=True)
    print(f"{len(todo) - failed} labelled, {failed} failed -> {pages}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["urls", "load", "pick", "label"])
    parser.add_argument("--urls", type=Path, default=DATA / "candidates.tsv")
    parser.add_argument("--domains", type=int, default=7000, help="urls: Tranco domains past seeds.py's 3000")
    parser.add_argument("--skip", type=int, default=0, help="urls: Tranco domains to skip first, those an earlier round took")
    parser.add_argument("--search", type=float, default=0.2, help="urls: share of new domains also searched for nonsense")
    parser.add_argument("--members", type=float, default=0.2, help="urls: share of new domains also given a members' path")
    parser.add_argument("--known", type=int, default=1000, help="urls: crawled domains also searched and given a members' path")
    parser.add_argument("--articles", type=int, default=15, help="urls: articles per paywalled site")
    parser.add_argument("--per-site", type=int, default=3, help="pick: most pages from one site")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--crawlers", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=15000, help="load: page load timeout, ms")
    parser.add_argument("--budget", type=int, default=1500, help="pick: pages to ask Jev about")
    parser.add_argument("--rare", type=float, default=0.6, help="pick: share of the budget for the rare presets")
    parser.add_argument("--presets", nargs="+", default=RARE, choices=PRESETS, help="pick: the rare presets to look for")
    parser.add_argument("--floor", type=float, default=0.2, help="pick: lowest score a rare pick may have")
    parser.add_argument("--workers", type=int, default=8, help="label: calls in flight")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    {"urls": cmd_urls, "load": cmd_load, "pick": cmd_pick, "label": cmd_label}[args.command](args)


if __name__ == "__main__":
    main()
