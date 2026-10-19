# /// script
# requires-python = ">=3.10"
# dependencies = ["scikit-learn>=1.5", "scipy", "numpy", "joblib"]
# ///
"""Rank the pages whose Jev labels most likely need a human look.

Label errors gather where Jev's answer disagrees with something else that
knows about the page, so each page and preset gets a suspicion score from:

- the linear model, trained without that page's site (5-fold by site), when
  it confidently says the opposite of the vote;
- the rules in features.py, when one that is usually right (a 404 status
  for an error page) contradicts the vote. A rule counts only if at least
  80% of its firings on this dataset agree with the vote: a captcha
  vendor's script, say, loads on plenty of pages that show no captcha;
- the vote itself, when Jev's calls did not agree or were not confident;
- the reworded question, when Jev answers it the other way.

Writes data/review.jsonl, most suspicious first, one line per page with the
presets in doubt; check the top of it before the rest of the gold set.

Run:  uv run research/page_classifier/review.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

import features
from features import PRESETS
from train import DATA, Linear, load, site


def position_bias(rows):
    """Whether Jev scores a question differently when it is asked earlier.
    collect.py asks the same questions in two opposite orders, so each one
    has an earlier and a later score on the same page."""
    shifts, flips = [], 0
    for row in rows:
        orders = row.get("jev_orders")
        if not orders or len(orders) < 2:
            continue
        first, second = row["jev"][0], row["jev"][1]
        for key in orders[0]:
            a, b = first.get(key), second.get(key)
            if a is None or b is None:
                continue
            gap = orders[1].index(key) - orders[0].index(key)  # >0: asked earlier in the first call
            if gap == 0:
                continue
            early, late = (a, b) if gap > 0 else (b, a)
            shifts.append(early - late)
            flips += (early >= 0.5) != (late >= 0.5)
    if shifts:
        print(f"position: asked earlier vs later, mean shift {np.mean(shifts):+.4f}, "
              f"mean |shift| {np.mean(np.abs(shifts)):.4f}, {flips} of {len(shifts)} answers flip at 0.5")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pages", type=Path, default=DATA / "pages.jsonl")
    parser.add_argument("--gold", type=Path, default=DATA / "gold.jsonl")
    parser.add_argument("--confidence", type=float, default=0.7)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    rows = load(args.pages, args.confidence, args.gold)
    oof = np.zeros((len(rows), len(PRESETS)))
    groups = [site(r) for r in rows]
    for train_idx, test_idx in GroupKFold(n_splits=args.folds).split(rows, groups=groups):
        model = Linear().fit([rows[i] for i in train_idx])
        oof[test_idx] = model.predict([rows[i] for i in test_idx])

    queue = []
    rules = [features.rules(row) for row in rows]
    trusted = set()
    for i, preset in enumerate(PRESETS):
        fired = [row["label"][i] for row, r in zip(rows, rules) if r[preset] and row["label"][i] is not None]
        if len(fired) >= 5 and np.mean(fired) >= 0.8:
            trusted.add(preset)
    print("trusted rules: " + (", ".join(sorted(trusted)) or "none"))

    for row, scores, rules in zip(rows, oof, rules):
        doubts = []
        for i, preset in enumerate(PRESETS):
            label = row["label"][i]
            answers = [s for s in (c.get(preset) for c in row["jev"]) if s is not None]
            mean = float(np.mean(answers)) if answers else None
            why, weight = [], 0.0
            if label is None:
                why.append("jev undecided")
                weight += 0.5
            elif abs(scores[i] - label) > 0.5:
                why.append(f"model says {scores[i]:.2f}")
                weight += abs(scores[i] - label)
            if row["wording_doubt"][i]:
                why.append("rewording disagrees")
                weight += 0.7
            if preset in trusted and label == 0 and rules[preset] == 1:
                why.append("rule fired")
                weight += 0.6
            if why:
                doubts.append({"preset": preset, "vote": label, "jev_mean": mean and round(mean, 3),
                               "model": round(float(scores[i]), 3), "why": why, "weight": round(weight, 3)})
        if doubts:
            queue.append({"url": row["url"], "status": row["status"], "title": row["title"],
                          "suspicion": round(sum(d["weight"] for d in doubts), 3), "doubts": doubts})

    position_bias(rows)
    queue.sort(key=lambda q: q["suspicion"], reverse=True)
    out = DATA / "review.jsonl"
    out.write_text("".join(json.dumps(q) + "\n" for q in queue))
    print(f"{len(queue)} of {len(rows)} pages in doubt -> {out}")
    for q in queue[:15]:
        doubts = ", ".join(f"{d['preset']} ({'; '.join(d['why'])})" for d in q["doubts"])
        print(f"  {q['suspicion']:.2f}  {q['url'][:60]:<60}  {doubts}")


if __name__ == "__main__":
    main()
