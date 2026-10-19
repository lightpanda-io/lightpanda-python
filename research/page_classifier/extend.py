# /// script
# requires-python = ">=3.10"
# dependencies = ["lightpanda"]
#
# [tool.uv.sources]
# lightpanda = { path = "../..", editable = true }
# ///
"""Ask presets added after a crawl on the pages it already labelled.

collect.py asks every preset in its REWORDED list. When the browser gains
presets after a crawl, this reloads each stored page and asks only the
presets (and their rewordings) its row lacks, in JUDGES calls with opposite
orders as collect.py does, instead of crawling everything again and losing
the loads the gold labels were checked on.

The new answers come from a fresh load, which can differ from the stored
one: a homepage that rotates, a challenge that passed, a rate limit that
lifted. They are merged into the row only when the status is the same and
the new tree shares at least --same of its lines with the stored one.
Otherwise the row only records the attempt, and train.py leaves those
presets out for that page as undecided.

Answers are appended to data/added.jsonl as pages finish, so an interrupted
pass resumes; at the end they are merged into pages.jsonl.

Needs TYPESAFE_API_KEY and a browser with the new presets. Run:
    uv run research/page_classifier/extend.py --limit 50
"""

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path
from time import perf_counter

from collect import DATA, JUDGES, MAX_TREE_BYTES, NODE_ID, REWORDED, camel, cap, complete, snake
from lightpanda import AsyncBrowser, LightpandaError, ProcessError


def missing(row: dict) -> list[str]:
    """The presets the row has no answers for and has not been asked since."""
    tried = {p for added in row.get("added", []) for p in added["presets"]}
    return [p for p in REWORDED if p not in row["jev"][0] and p not in tried]


def similarity(a: str, b: str) -> float:
    """The share of lines two trees have in common (Jaccard over line sets)."""
    a, b = set(a.splitlines()), set(b.splitlines())
    return len(a & b) / max(len(a | b), 1)


async def ask(page, row: dict, presets: list[str], timeout: int) -> dict:
    loaded = await page.goto(url=row["seed"], timeout=timeout)
    tree = cap(NODE_ID.sub(r"\1", await page.tree()), MAX_TREE_BYTES)
    same = round(similarity(row["tree"], tree), 3)
    added = {"seed": row["seed"], "presets": presets, "status": loaded.http_status, "similarity": same}
    questions = {camel(p): True for p in presets} | {f"re_{p}": REWORDED[p] for p in presets}
    order = random.sample(list(questions), len(questions))
    orders = [order if i % 2 == 0 else order[::-1] for i in range(JUDGES)]
    scores, seconds = [], []
    for keys in orders:
        start = perf_counter()
        answers = await page.classify(questions={key: questions[key] for key in keys})
        scores.append({snake(key): answers.get(key) for key in keys})
        seconds.append(round(perf_counter() - start, 3))
    return added | {"jev": scores, "orders": [[snake(k) for k in keys] for keys in orders], "seconds": seconds}


async def visit(browser, slots, out, row: dict, presets: list[str], timeout: int, counts: dict):
    async with slots:
        try:
            async with await browser.new_session() as page:
                added = await asyncio.wait_for(ask(page, row, presets, timeout), timeout / 1000 * 4)
        except ProcessError:
            counts["lost"].append((row, presets))  # the browser died: ask again in a new one
            return
        except (LightpandaError, asyncio.TimeoutError) as err:
            counts["failed"] += 1
            added = {"seed": row["seed"], "presets": presets, "error": str(err) or type(err).__name__}
        else:
            counts["ok"] += 1
        out.write(json.dumps(added) + "\n")
        out.flush()
        if sys.stderr.isatty():
            done = counts["ok"] + counts["failed"]
            print(f"\r{done}/{counts['total']}  ok {counts['ok']}  failed {counts['failed']}", end="", file=sys.stderr)


def merge(pages: Path, added_path: Path, same: float) -> None:
    """Fold the answers into pages.jsonl: into the row's calls when the load
    matches the stored one, else as a record of the attempt only."""
    by_seed = {a["seed"]: a for a in map(json.loads, added_path.open())}
    merged = differed = 0
    rows = []
    for line in pages.open():
        row = json.loads(line)
        added = by_seed.get(row["seed"])
        if added and "error" not in row and "error" not in added:  # a failed load is asked again next time
            match = added["status"] == row["status"] and added["similarity"] >= same
            if match:
                for call, scores in zip(row["jev"], added["jev"]):
                    call.update(scores)
                merged += 1
            else:
                differed += 1
            added = {k: v for k, v in added.items() if k != "seed" and (match or k != "jev")}
            row.setdefault("added", []).append(added | {"merged": match})
        rows.append(row)
    tmp = pages.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    tmp.replace(pages)
    added_path.unlink()
    print(f"{merged} pages got the new answers, {differed} loaded differently and stay undecided -> {pages}")


async def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pages", type=Path, default=DATA / "pages.jsonl")
    parser.add_argument("--added", type=Path, default=DATA / "added.jsonl")
    parser.add_argument("--limit", type=int, help="ask at most this many pages")
    parser.add_argument("--crawlers", type=int, default=6, help="pages loading at once")
    parser.add_argument("--timeout", type=int, default=15000, help="page load timeout, ms")
    parser.add_argument("--same", type=float, default=0.5, help="share of tree lines a reload must keep")
    parser.add_argument("--verbose", action="store_true", help="let the browser's logs through to stderr")
    args = parser.parse_args()

    done = {json.loads(line)["seed"] for line in args.added.open()} if args.added.exists() else set()
    rows = [json.loads(line) for line in args.pages.open()]
    todo = [(r, missing(r)) for r in rows if "error" not in r and complete(r) and r["seed"] not in done]
    todo = [(r, presets) for r, presets in todo if presets][: args.limit]

    # A page can crash the browser (debug builds assert), which fails every
    # page after it: those go round again in a new browser, until a round
    # gets nothing done.
    counts = {"ok": 0, "failed": 0, "total": len(todo)}
    slots = asyncio.Semaphore(args.crawlers)
    start = perf_counter()
    with args.added.open("a") as out:
        while todo:
            counts["lost"], before = [], counts["ok"] + counts["failed"]
            async with AsyncBrowser(max_concurrency=args.crawlers * 2, verbose=args.verbose) as browser:
                await asyncio.gather(*(visit(browser, slots, out, r, presets, args.timeout, counts) for r, presets in todo))
            todo = counts["lost"]
            if todo:
                print(f"\nthe browser died, {len(todo)} pages left for a new one", file=sys.stderr)
            if counts["ok"] + counts["failed"] == before:
                break
    print(f"\n{counts['ok']} asked, {counts['failed']} failed, {len(todo)} lost to crashes in {perf_counter() - start:.0f}s")
    merge(args.pages, args.added, args.same)


if __name__ == "__main__":
    asyncio.run(main())
