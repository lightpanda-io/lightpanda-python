# /// script
# requires-python = ">=3.10"
# dependencies = ["scikit-learn>=1.5", "scipy", "numpy", "joblib"]
# ///
"""Score every page with the rules and the linear model, trained without its fold.

Reads the folds from data/folds.jsonl (folds.py) and the pages, HTML and all,
from data/pages.jsonl, which the linear model's structure features need.
Writes one line per page, in folds.jsonl's order, to data/oof/rules.jsonl
and data/oof/linear.jsonl: `{"url": ..., "scores": [12 floats]}`.
compare.py lines them up with the transformers' scores.

Run:  uv run research/page_classifier/crossval.py
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

import features
from features import PRESETS
from train import DATA, Linear, load


def write(path: Path, rows: list[dict], scores) -> None:
    path.parent.mkdir(exist_ok=True)
    with path.open("w") as out:
        for row, s in zip(rows, scores):
            out.write(json.dumps({"url": row["url"], "scores": np.round(s, 5).tolist()}) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pages", type=Path, default=DATA / "pages.jsonl")
    parser.add_argument("--gold", type=Path, default=DATA / "gold.jsonl")
    parser.add_argument("--folds", type=Path, default=DATA / "folds.jsonl")
    parser.add_argument("--confidence", type=float, default=0.7)
    parser.add_argument("--soft", action="store_true", help="also learn the undecided labels at Jev's mean score; writes linear-soft")
    args = parser.parse_args()

    rows = load(args.pages, args.confidence, args.gold)
    split = [json.loads(line) for line in args.folds.open()]
    assert [r["url"] for r in rows] == [s["url"] for s in split], "folds.jsonl is stale: rerun folds.py"
    fold = np.array([s["fold"] for s in split])

    rules = [[features.rules(r)[p] for p in PRESETS] for r in rows]
    write(DATA / "oof" / "rules.jsonl", rows, rules)

    linear = np.zeros((len(rows), len(PRESETS)))
    for f in sorted(set(fold)):
        start = perf_counter()
        train = [r for r, k in zip(rows, fold) if k != f]
        test = np.flatnonzero(fold == f)
        linear[test] = Linear(soft=args.soft).fit(train).predict([rows[i] for i in test])
        print(f"fold {f}: {len(train)} train, {len(test)} scored in {perf_counter() - start:.0f}s")
    write(DATA / "oof" / ("linear-soft.jsonl" if args.soft else "linear.jsonl"), rows, linear)
    print(f"{len(rows)} pages -> {DATA / 'oof'}")


if __name__ == "__main__":
    import crossval

    crossval.main()
