# /// script
# requires-python = ">=3.10"
# dependencies = ["scikit-learn>=1.5", "numpy", "matplotlib"]
# ///
"""Compare every model's out-of-fold scores on the same pages.

Reads data/folds.jsonl (folds.py) and each data/oof/<model>.jsonl that
crossval.py and transformer.py wrote, every page scored by a model that never
saw its site. Each model gets the same treatment:

- per preset, the threshold with the best F1 against Jev's vote, searched
  over the scores the model gives the positives;
- F1 and ROC AUC per preset, against the vote (undecided presets left out);
- verdict agreement with Jev on judge()'s `ok`, on the pages with every
  preset decided, and verdict accuracy on the hand-checked gold pages;
- the cascade: scores calibrated so each threshold sits at 0.5, as the
  browser does, a page is answered locally when every preset is below --sure
  or above 1 - --sure; the share answered and the agreement on those,
  plus a sweep of bands and a plot of the trade-off;
- cost, from data/costs.json when score.py has measured it.

`ensemble` averages the calibrated linear and ft-base scores, and `staged`
runs it only on the pages the linear model is unsure of.

Writes data/comparison.txt and data/comparison.png.

Run:  uv run research/page_classifier/compare.py
"""

import argparse
import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
DATA = HERE / "data"
PRESETS = [
    "is_blocked", "is_captcha", "is_consent_wall", "is_empty_catalog", "is_error_page", "is_login_wall",
    "is_paywall", "is_unsupported_browser", "is_loading", "is_age_gate", "is_region_blocked", "is_rate_limited",
]
ORDER = ["rules", "linear", "frozen", "ft-small", "ft-small-long", "ft-base", "ensemble"]
BANDS = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
TARGETS = [0.995, 0.99, 0.98]


def f1(truth, guess) -> float:
    tp = (truth & guess).sum()
    return 2 * tp / max(truth.sum() + guess.sum(), 1)


def thresholds_for(scores, vote) -> np.ndarray:
    """Each preset's F1-best threshold against the vote."""
    out = np.full(len(PRESETS), 0.5)
    for j in range(len(PRESETS)):
        known = ~np.isnan(vote[:, j])
        truth = vote[known, j] == 1
        if truth.sum() >= 3 and len(np.unique(scores[known, j])) > 2:
            # The best F1 is at some positive's score, so those are the
            # candidates: a fine-tuned model can score a rare preset below
            # any fixed grid, positives included.
            grid = np.unique(scores[known, j][truth])
            grid = grid[(grid > 0) & (grid < 1)]
            out[j] = grid[int(np.argmax([f1(truth, scores[known, j] >= t) for t in grid]))]
    return out


def calibrate(scores, thresholds):
    """Map each threshold to 0.5, linearly on either side (classify_local's
    Preset.calibrated)."""
    t = thresholds[None, :]
    return np.where(scores < t, 0.5 * scores / t, 0.5 + 0.5 * (scores - t) / (1 - t))


def read_scores(path: Path, urls: list[str]) -> np.ndarray:
    rows = [json.loads(line) for line in path.open()]
    assert [r["url"] for r in rows] == urls, f"{path} does not follow folds.jsonl: rerun it"
    return np.array([r["scores"] for r in rows], dtype=float)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--folds", type=Path, default=DATA / "folds.jsonl")
    parser.add_argument("--gold", type=Path, default=DATA / "gold.jsonl")
    parser.add_argument("--sure", type=float, default=0.1)
    parser.add_argument("--skip-source", help="score only pages whose source does not start with this, e.g. mine- for the "
                        "pages mine.py picked as hard (they still train the other folds)")
    parser.add_argument("--oof", type=Path, default=DATA / "oof")
    parser.add_argument("--out", default="comparison", help="name of the .txt and .png written under data/")
    args = parser.parse_args()

    pages = [json.loads(line) for line in args.folds.open()]
    urls = [p["url"] for p in pages]
    keep = np.array([not (args.skip_source and p["source"].startswith(args.skip_source)) for p in pages])
    pages = [p for p, k in zip(pages, keep) if k]
    vote = np.array([[np.nan if v is None else v for v in p["vote"]] for p in pages], dtype=float)
    decided = ~np.isnan(vote)
    whole = decided.all(axis=1)
    jev = vote == 1
    ok = ~jev.any(axis=1)

    scores = {}
    extra = sorted(p.stem for p in args.oof.glob("*.jsonl") if p.stem not in ORDER)
    names_on_disk = [n for n in ORDER if n != "ensemble"] + extra
    for name in names_on_disk:
        path = args.oof / f"{name}.jsonl"
        if path.exists():
            scores[name] = read_scores(path, urls)[keep]
    urls = [p["url"] for p in pages]
    thresholds = {name: (np.full(len(PRESETS), 0.5) if name == "rules" else thresholds_for(s, vote))
                  for name, s in scores.items()}
    calibrated = {name: calibrate(s, thresholds[name]) for name, s in scores.items() if name != "rules"}
    tuned = [n for n in calibrated if n.startswith("ft-")]
    if "linear" in calibrated and tuned:
        # ft-base is the one transformer.py fits and exports; without it, the
        # fine-tuned model with the best verdict.
        best = "ft-base" if "ft-base" in tuned else max(tuned, key=lambda n: (~(calibrated[n][whole] >= 0.5).any(axis=1) == ok[whole]).mean())
        mixed = (calibrated["linear"] + calibrated[best]) / 2
        scores["ensemble"], thresholds["ensemble"] = mixed, thresholds_for(mixed, vote)
        calibrated["ensemble"] = calibrate(mixed, thresholds["ensemble"])
        print(f"ensemble: linear + {best}\n")
    names = [n for n in [*ORDER[:-1], *extra, "ensemble"] if n in scores]
    fired = {n: scores[n] >= thresholds[n] for n in names}
    jev_first = np.array([[s or 0.0 for s in p["jev_calls"][0]] for p in pages]) >= 0.5

    gold = {}
    if args.gold.exists():
        gold = {g["url"]: g["labels"] for g in map(json.loads, args.gold.open())}
    picked = [i for i, u in enumerate(urls) if u in gold]
    truth_ok = np.array([not any(gold[urls[i]].get(p, jev[i, j]) for j, p in enumerate(PRESETS)) for i in picked])

    width = 9
    print(f"{len(pages)} pages, {whole.sum()} with every preset decided, {len(picked)} hand-checked\n")
    print("F1 against Jev's vote (AUC below), out of fold")
    print(f"{'preset':<24}{'pos':>5}" + "".join(f"{n:>{width}}" for n in ["jev#1", *names]))
    for j, preset in enumerate(PRESETS):
        d = decided[:, j]
        truth = jev[d, j]
        cells = [f1(truth, jev_first[d, j])] + [f1(truth, fired[n][d, j]) for n in names]
        print(f"{preset:<24}{int(truth.sum()):>5}" + "".join(f"{c:>{width}.2f}" for c in cells))
        aucs = [roc_auc_score(truth, scores[n][d, j]) if 0 < truth.sum() < len(truth) and n != "rules" else np.nan for n in names]
        print(f"{'':<24}{'':>5}{'':>{width}}" + "".join(f"{'' if np.isnan(a) else f'({a:.2f})':>{width}}" for a in aucs))

    def verdict(guess):
        return (~guess[whole].any(axis=1) == ok[whole]).mean()

    print(f"\n{'':<24}{'':>5}" + "".join(f"{n:>{width}}" for n in ["jev#1", *names]))
    print(f"{'verdict vs Jev':<29}" + "".join(f"{verdict(g):>{width}.1%}" for g in [jev_first, *fired.values()]))
    if picked:
        jev_gold = (~jev[picked].any(axis=1) == truth_ok).mean()
        print(f"{'gold accuracy':<29}{jev_gold:>{width}.1%}"
              + "".join(f"{(~fired[n][picked].any(axis=1) == truth_ok).mean():>{width}.1%}" for n in names)
              + "   (jev#1 column: Jev's vote)")

    print(f"\ncascade: answered locally / agreeing with Jev, on the {whole.sum()} pages with every preset decided")
    cascaded = [n for n in names if n in calibrated]
    print(f"{'--sure':>8}" + "".join(f"{n:>17}" for n in cascaded))
    curves = {n: [] for n in cascaded}
    for band in BANDS:
        cells = []
        for n in cascaded:
            c = calibrated[n][whole]
            sure = ((c < band) | (c > 1 - band)).all(axis=1)
            agree = (~fired[n][whole][sure].any(axis=1) == ok[whole][sure]).mean() if sure.any() else np.nan
            curves[n].append((sure.mean(), agree))
            mark = "*" if band == args.sure else " "
            cells.append(f"{sure.mean():>7.1%} / {agree:>6.1%}{mark}")
        print(f"{band:>8.2f}" + "".join(f"{c:>17}" for c in cells))

    # The summary searches a fine grid of bands: on the coarse one above, a
    # model can miss a target agreement by a hair and lose 10 points of pages.
    fine = np.round(np.arange(0.005, 0.4, 0.005), 3)
    print(f"\nmost pages answered locally at an agreement or better (the band that gives it)")
    print(f"  {'':<20}" + "".join(f"{t:>24.1%}" for t in TARGETS))
    for n in cascaded:
        c, guess_ok = calibrated[n][whole], ~fired[n][whole].any(axis=1)
        points = []
        for band in fine:
            sure = ((c < band) | (c > 1 - band)).all(axis=1)
            if sure.any():
                points.append((sure.mean(), (guess_ok[sure] == ok[whole][sure]).mean(), band))
        cells = []
        for target in TARGETS:
            best = max(((share, band) for share, agree, band in points if agree >= target), default=None)
            cells.append(f"{best[0]:.1%} (--sure {best[1]:g})" if best else "-")
        print(f"  {n:<20}" + "".join(f"{cell:>24}" for cell in cells))

    if "linear" in calibrated and "ensemble" in calibrated:
        print("\nstaged: linear first, the ensemble (so the transformer) only where linear is unsure, Jev for the rest")
        for first, second in ((0.02, 0.15), (0.05, 0.05), (0.05, 0.1), (0.05, 0.15), (0.1, 0.1)):
            lin, ens = calibrated["linear"][whole], calibrated["ensemble"][whole]
            s1 = ((lin < first) | (lin > 1 - first)).all(axis=1)
            s2 = ~s1 & ((ens < second) | (ens > 1 - second)).all(axis=1)
            verdict_ok = np.where(s1, ~fired["linear"][whole].any(axis=1), ~fired["ensemble"][whole].any(axis=1))
            local = s1 | s2
            print(f"  linear {first:.2f}, ensemble {second:.2f}: {local.mean():6.1%} local at "
                  f"{(verdict_ok[local] == ok[whole][local]).mean():6.1%}, transformer on {(~s1).mean():5.1%} of pages")

    costs_path = DATA / "costs.json"
    if costs_path.exists():
        costs = json.loads(costs_path.read_text())
        print("\ncost on this machine's CPU, one page at a time")
        for name, c in costs.items():
            print(f"  {name:<24}{c['ms']:>8.1f} ms/page {c.get('ms_one_thread', float('nan')):>8.1f} ms on 1 thread"
                  f"{c['mb']:>8.1f} MB  {c.get('note', '')}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for n in cascaded:
        xs, ys = zip(*curves[n])
        ax.plot(xs, ys, marker="o", label=n)
    ax.set_xlabel("pages answered locally")
    ax.set_ylabel("verdict agreement with Jev on those")
    ax.set_title("cascade trade-off, out of fold (bands 0.02 to 0.4)")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(DATA / f"{args.out}.png", dpi=150)
    return args.out


if __name__ == "__main__":
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        out = main()
    print(buffer.getvalue(), end="")
    (DATA / f"{out}.txt").write_text(buffer.getvalue())
