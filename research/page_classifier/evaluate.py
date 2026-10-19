# /// script
# requires-python = ">=3.10"
# dependencies = ["scikit-learn>=1.5", "numpy"]
# ///
"""Score the local models on the held-out pages from train.py.

Jev's label is train.py's majority vote; a preset the vote left undecided
is not scored, and only pages with every preset decided count toward the
verdict and the cascade.

- Per preset: F1 against the vote for Jev's first call alone (a single
  call, as judge() makes; it is one of the voters, so this flatters it),
  the rules and the linear model, and the linear model's ROC AUC.
- Verdict: how often each one agrees with the vote on judge()'s `ok`, the
  keep-or-skip decision a crawl acts on.
- Cascade: the share of pages where the linear model is sure of every
  preset, and its verdict agreement on those. Jev answers the rest. Sure
  means far from that preset's threshold: --sure is the band at a 0.5
  threshold (below 0.1 or above 0.9), scaled to each side of the others.

The linear model fires each preset at its own threshold, picked by train.py
and read from data/train_meta.json.
- Gold: with data/gold.jsonl (`{"url": ..., "labels": {"is_paywall": true,
  ...}}` per line, checked by hand), each one's accuracy against the truth.

Run:  uv run research/page_classifier/evaluate.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from features import PRESETS

HERE = Path(__file__).parent
DATA = HERE / "data"


def f1(truth, guess):
    if not truth.any() and not guess.any():
        return float("nan")  # nothing to find and nothing found
    return f1_score(truth, guess, zero_division=0)


def auc(truth, scores):
    return roc_auc_score(truth, scores) if 0 < truth.sum() < len(truth) else float("nan")


def fmt(value):
    return "  -  " if np.isnan(value) else f"{value:.2f} "


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--predictions", type=Path, default=DATA / "predictions.jsonl")
    parser.add_argument("--gold", type=Path, default=DATA / "gold.jsonl")
    parser.add_argument("--sure", type=float, default=0.1, help="cascade: sure below this or above 1 minus it")
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.predictions.open()]
    vote = np.array([[np.nan if label is None else label for label in r["jev"]] for r in rows])
    decided = ~np.isnan(vote)
    jev = vote == 1
    first = np.array([[s or 0.0 for s in r["jev_calls"][0]] for r in rows]) >= 0.5
    rules = np.array([r["rules"] for r in rows]) >= 0.5
    meta_path = DATA / "train_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    thresholds = np.array(meta.get("thresholds", [0.5] * len(PRESETS)))
    linear_scores = np.array([r["linear"] for r in rows])
    linear = linear_scores >= thresholds

    print(f"{len(rows)} held-out pages, {1 - decided.mean():.1%} of their preset labels undecided\n")
    print(f"{'preset':<24}{'pos':>5}  {'undec':>5}  {'jev#1':>5}  {'rules':>5}  {'linear':>6}  {'auc':>5}  {'at':>4}")
    for i, preset in enumerate(PRESETS):
        d = decided[:, i]
        truth = jev[d, i]
        print(f"{preset:<24}{int(truth.sum()):>5}  {int((~d).sum()):>5}  {fmt(f1(truth, first[d, i]))} {fmt(f1(truth, rules[d, i]))}  "
              f"{fmt(f1(truth, linear[d, i]))}  {fmt(auc(truth, linear_scores[d, i]))} {thresholds[i]:.2f}")

    whole = decided.all(axis=1)
    ok = ~jev[whole].any(axis=1)
    print(f"\nverdict (ok) agreement with Jev, on the {whole.sum()} pages with every preset decided:")
    for name, guess in (("jev#1", first), ("rules", rules), ("linear", linear)):
        print(f"  {name:<7}{(~guess[whole].any(axis=1) == ok).mean():.1%}")

    band = args.sure / 0.5
    sure = ((linear_scores[whole] < thresholds * band) | (linear_scores[whole] > 1 - (1 - thresholds) * band)).all(axis=1)
    if sure.any():
        agree = (~linear[whole][sure].any(axis=1) == ok[sure]).mean()
        print(f"\ncascade: linear is sure on {sure.mean():.1%} of those pages, agreeing with Jev on {agree:.1%} of them;"
              f" {(~sure).sum()} of {whole.sum()} would still go to Jev")
    else:
        print("\ncascade: linear is sure of no page")

    if meta:
        print(f"\nlinear: {meta['linear_ms_per_page']} ms/page, {meta['model_bytes'] / 1e6:.1f} MB, trained on {meta['train']} pages")

    if args.gold.exists():
        gold = {g["url"]: g["labels"] for g in map(json.loads, args.gold.open())}
        picked = [i for i, r in enumerate(rows) if r["url"] in gold]
        if picked:
            truth = np.array([[bool(gold[rows[i]["url"]].get(p)) for p in PRESETS] for i in picked])
            print(f"\ngold: {len(picked)} hand-checked held-out pages, verdict accuracy")
            for name, guess in (("jev", jev), ("rules", rules), ("linear", linear)):
                print(f"  {name:<7}{(~guess[picked].any(axis=1) == ~truth.any(axis=1)).mean():.1%}")


if __name__ == "__main__":
    main()
