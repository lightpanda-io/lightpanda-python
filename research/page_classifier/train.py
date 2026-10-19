# /// script
# requires-python = ">=3.10"
# dependencies = ["scikit-learn>=1.5", "scipy", "numpy", "joblib"]
# ///
"""Train local models to copy Jev's presets, and predict the held-out pages.

Pages are split by site, so no site lands in both train and test: a model
that memorized a site's layout would otherwise look better than it is.

Labels are a majority vote of Jev's calls on the preset question itself: a
preset is yes when most calls score it above --confidence, no when most score
it below 1 - --confidence, and undecided otherwise. An undecided preset is
left out of that preset's training and scoring, which keeps Jev's borderline
answers from teaching the model noise.

The reworded questions (collect.py's REWORDED) do not vote: some ask a
slightly different thing (a login page is a login wall to the preset, but
not "a sign-in needed before seeing what the page is for"), so they would
veto real positives. A preset whose rewording lands on the other side is
marked in `wording_doubt` for review.py instead.

Models, cheapest first:

- rules: the hand-written guesses in features.py.
- linear: logistic regression per preset over word and char n-grams of the
  tree plus the structural features. A weight vector per preset, small and
  easy to port: the n-grams are hashed (features.ngrams), so the browser
  needs the weights of the buckets it can meet, not a vocabulary.

A preset fires when its score reaches that preset's own threshold. Rare
presets score low even on the pages that have them, so a shared 0.5 misses
most of them; each threshold is the one with the best F1 on scores the
training pages got from models trained without their site (--folds).

Writes data/model.joblib and data/predictions.jsonl for evaluate.py.

Run:  uv run research/page_classifier/train.py
"""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import normalize

import features
from features import PRESETS

HERE = Path(__file__).parent
DATA = HERE / "data"


def vote(scores: list, confidence: float) -> int | None:
    """1 or 0 when most calls are confident of it, else None. A call that
    gave no answer votes for neither."""
    majority = len(scores) // 2 + 1
    if sum(s is not None and s > confidence for s in scores) >= majority:
        return 1
    if sum(s is not None and s < 1 - confidence for s in scores) >= majority:
        return 0
    return None


def load(path: Path, confidence: float, gold: Path | None = None) -> list[dict]:
    """The labelled pages, minus duplicates: many domains redirect to the same
    site, which would count it twice and could put it on both sides of the
    split.

    Each row's `vote` is Jev's majority vote, and its `label`, the one the
    models learn, is the same unless the page was checked by hand (label.py
    writes those to `gold`): then the hand label wins."""
    checked = {}
    if gold and gold.exists():
        checked = {g["url"]: g["labels"] for g in map(json.loads, gold.open())}
    rows, seen = [], set()
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            if "error" in row or not row.get("jev"):
                continue
            key = (site(row), hashlib.sha1(row["tree"].encode()).hexdigest())
            if key in seen:
                continue
            seen.add(key)
            row["jev_calls"] = [[call.get(p) for p in PRESETS] for call in row["jev"]]
            row["vote"] = [vote([call.get(p) for call in row["jev"]], confidence) for p in PRESETS]
            reworded = [vote([call.get(f"re_{p}") for call in row["jev"]], confidence) for p in PRESETS]
            row["wording_doubt"] = [a is not None and b is not None and a != b for a, b in zip(row["vote"], reworded)]
            hand = checked.get(row["url"])
            # A preset added after the page was checked keeps Jev's vote.
            row["label"] = [int(hand[p]) if p in hand else v for p, v in zip(PRESETS, row["vote"])] if hand else row["vote"]
            rows.append(row)
    return rows


def site(row: dict) -> str:
    host = row["url"].split("://", 1)[-1].split("/", 1)[0].lower()
    return host.removeprefix("www.")


class HashedTfidf:
    """TF-IDF over one family of hashed n-gram counts, as sklearn's
    TfidfVectorizer(sublinear_tf=True) computes it: 1 + ln(count), times the
    smooth idf ln((1 + pages) / (1 + pages with it)) + 1, the row scaled to
    unit length. Buckets in fewer than min_df training pages are dropped,
    and of the rest only the max_features most frequent are kept (ties go
    to the lower bucket).

    Columns are the kept buckets in increasing order (`buckets`), so the
    weights the browser needs are the ones for those buckets."""

    def __init__(self, max_features: int, min_df: int = 2):
        self.max_features = max_features
        self.min_df = min_df

    def fit(self, counts: list[dict]) -> "HashedTfidf":
        df, total = {}, {}
        for page in counts:
            for b, n in page.items():
                df[b] = df.get(b, 0) + 1
                total[b] = total.get(b, 0) + n
        frequent = sorted((b for b, n in df.items() if n >= self.min_df), key=lambda b: (-total[b], b))
        kept = sorted(frequent[: self.max_features])
        self.buckets = np.array(kept, dtype=np.int64)
        self.idf = np.log((1 + len(counts)) / (1 + np.array([df[b] for b in kept], dtype=float))) + 1
        return self

    def transform(self, counts: list[dict]):
        rows, cols, vals = [], [], []
        for r, page in enumerate(counts):
            keys = np.fromiter(page.keys(), dtype=np.int64, count=len(page))
            tf = np.fromiter(page.values(), dtype=float, count=len(page))
            at = np.searchsorted(self.buckets, keys).clip(max=len(self.buckets) - 1)
            hit = self.buckets[at] == keys
            rows.append(np.full(hit.sum(), r))
            cols.append(at[hit])
            vals.append((1 + np.log(tf[hit])) * self.idf[at[hit]])
        m = csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                       shape=(len(counts), len(self.buckets)))
        return normalize(m)


class Linear:
    """One logistic regression per preset over n-grams plus structure."""

    def __init__(self, c: float = 4.0):
        self.c = c
        self.words = HashedTfidf(max_features=100_000)
        self.chars = HashedTfidf(max_features=200_000)
        self.shape = DictVectorizer()
        self.models: dict[str, LogisticRegression | float] = {}
        self.thresholds = np.full(len(PRESETS), 0.5)

    def matrix(self, rows, fit=False):
        words, chars = zip(*(ngrams(r) for r in rows))
        shapes = [structure(r) for r in rows]
        if fit:
            self.words.fit(words)
            self.chars.fit(chars)
            return hstack([self.words.transform(words), self.chars.transform(chars), self.shape.fit_transform(shapes)]).tocsr()
        return hstack([self.words.transform(words), self.chars.transform(chars), self.shape.transform(shapes)]).tocsr()

    def fit(self, rows):
        x = self.matrix(rows, fit=True)
        for i, preset in enumerate(PRESETS):
            decided = [j for j, r in enumerate(rows) if r["label"][i] is not None]
            y = np.array([rows[j]["label"][i] for j in decided])
            if y.sum() < 3 or y.sum() == len(y):
                self.models[preset] = float(y.mean()) if len(y) else 0.0  # too rare to learn yet
                continue
            model = LogisticRegression(C=self.c, max_iter=2000, class_weight="balanced")
            model.fit(x[decided], y)
            self.models[preset] = model
        return self

    def predict(self, rows) -> np.ndarray:
        x = self.matrix(rows)
        out = np.zeros((len(rows), len(PRESETS)))
        for i, preset in enumerate(PRESETS):
            model = self.models[preset]
            out[:, i] = model if isinstance(model, float) else model.predict_proba(x)[:, 1]
        return out

    def decide(self, scores: np.ndarray) -> np.ndarray:
        return scores >= self.thresholds


def ngrams(row: dict) -> tuple:
    """features.ngrams, kept on the row like structure."""
    if "_ngrams" not in row:
        row["_ngrams"] = features.ngrams(row)
    return row["_ngrams"]


def structure(row: dict) -> dict:
    """features.structure, kept on the row: the HTML regexes are the slow
    part, and the folds would run them again on every pass."""
    if "_structure" not in row:
        row["_structure"] = features.structure(row)
    return row["_structure"]


def best_thresholds(rows: list[dict], folds: int) -> np.ndarray:
    """Each preset's F1-best threshold on out-of-fold scores, grouped by site."""
    oof = np.zeros((len(rows), len(PRESETS)))
    groups = [site(r) for r in rows]
    for train_idx, test_idx in GroupKFold(n_splits=folds).split(rows, groups=groups):
        oof[test_idx] = Linear().fit([rows[i] for i in train_idx]).predict([rows[i] for i in test_idx])
    grid = np.linspace(0.02, 0.98, 49)
    out = np.full(len(PRESETS), 0.5)
    for i in range(len(PRESETS)):
        decided = [j for j, r in enumerate(rows) if r["label"][i] is not None]
        y = np.array([rows[j]["label"][i] for j in decided], dtype=bool)
        if y.sum() < 3:
            continue  # too few to pick from
        f1s = [f1_score(y, oof[decided, i] >= t, zero_division=0) for t in grid]
        out[i] = grid[int(np.argmax(f1s))]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pages", type=Path, default=DATA / "pages.jsonl")
    parser.add_argument("--gold", type=Path, default=DATA / "gold.jsonl", help="hand labels from label.py, which override Jev's")
    parser.add_argument("--confidence", type=float, default=0.7, help="score a vote needs to count as sure")
    parser.add_argument("--test", type=float, default=0.25, help="share of sites held out")
    parser.add_argument("--folds", type=int, default=3, help="folds for picking the thresholds")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    rows = load(args.pages, args.confidence, args.gold)
    labels = [label for r in rows for label in r["label"]]
    hand = sum(r["label"] is not r["vote"] for r in rows)
    print(f"{sum(label is None for label in labels) / len(labels):.1%} of preset labels undecided at {args.confidence}, "
          f"{hand} pages labelled by hand")
    groups = [site(r) for r in rows]
    train_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=args.test, random_state=args.seed).split(rows, groups=groups))
    train = [rows[i] for i in train_idx]
    test = [rows[i] for i in test_idx]
    print(f"{len(rows)} pages from {len(set(groups))} sites: {len(train)} train, {len(test)} test")

    start = perf_counter()
    thresholds = best_thresholds(train, args.folds)
    print("thresholds: " + ", ".join(f"{p.removeprefix('is_')} {t:.2f}" for p, t in zip(PRESETS, thresholds)))
    linear = Linear().fit(train)
    linear.thresholds = thresholds
    print(f"linear trained in {perf_counter() - start:.1f}s")
    start = perf_counter()
    linear_scores = linear.predict(test)
    per_page = (perf_counter() - start) / max(len(test), 1)

    # The deployable model is trained on everything, with the thresholds
    # picked on the training split.
    final = Linear().fit(rows)
    final.thresholds = thresholds
    model_path = DATA / "model.joblib"
    joblib.dump(final, model_path, compress=3)

    with (DATA / "predictions.jsonl").open("w") as out:
        for row, scores in zip(test, linear_scores):
            rules = features.rules(row)
            out.write(json.dumps({
                "url": row["url"],
                "source": row["source"],
                "jev_calls": row["jev_calls"],
                "jev": row["vote"],
                "rules": [rules[p] for p in PRESETS],
                "linear": scores.round(4).tolist(),
            }) + "\n")
    meta = {"linear_ms_per_page": round(per_page * 1000, 2), "model_bytes": model_path.stat().st_size,
            "train": len(train), "test": len(test), "thresholds": thresholds.round(2).tolist()}
    (DATA / "train_meta.json").write_text(json.dumps(meta))
    print(f"{meta['linear_ms_per_page']} ms/page, model {meta['model_bytes'] / 1e6:.1f} MB -> {model_path}")


if __name__ == "__main__":
    # Run as the module `train`, so the pickled model refers to train.Linear
    # and loads anywhere train.py is importable, not only inside this script.
    import train

    train.main()
