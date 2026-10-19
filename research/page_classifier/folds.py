# /// script
# requires-python = ">=3.10"
# dependencies = ["scikit-learn>=1.5", "scipy", "numpy", "joblib"]
# ///
"""Split the labelled pages into the folds every model is cross-validated on.

Each page goes to one of --folds folds, grouped by site so no site is on both
sides of a split. The pages are written with what a transformer needs and no
more (url, title, status, tree, labels), so the file is small enough to copy
to a GPU machine; crossval.py and transformer.py read their folds from it, so
their out-of-fold scores cover the same pages split the same way.

Writes data/folds.jsonl.

Run:  uv run research/page_classifier/folds.py
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from sklearn.model_selection import GroupKFold

from train import DATA, load, site


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pages", type=Path, default=DATA / "pages.jsonl")
    parser.add_argument("--gold", type=Path, default=DATA / "gold.jsonl")
    parser.add_argument("--confidence", type=float, default=0.7)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--out", type=Path, default=DATA / "folds.jsonl")
    args = parser.parse_args()

    rows = load(args.pages, args.confidence, args.gold)
    groups = [site(r) for r in rows]
    fold_of = [0] * len(rows)
    for fold, (_, test) in enumerate(GroupKFold(n_splits=args.folds).split(rows, groups=groups)):
        for i in test:
            fold_of[i] = fold

    with args.out.open("w") as out:
        for row, group, fold in zip(rows, groups, fold_of):
            out.write(json.dumps({
                "url": row["url"], "site": group, "fold": fold, "source": row["source"],
                "title": row["title"], "status": row["status"], "tree": row["tree"],
                "label": row["label"], "vote": row["vote"], "jev_calls": row["jev_calls"],
            }, ensure_ascii=False) + "\n")
    sizes = Counter(fold_of)
    print(f"{len(rows)} pages from {len(set(groups))} sites in {args.folds} folds "
          f"({', '.join(str(sizes[f]) for f in range(args.folds))}) -> {args.out}")


if __name__ == "__main__":
    main()
