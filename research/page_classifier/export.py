# /// script
# requires-python = ">=3.10"
# dependencies = ["scikit-learn>=1.5", "scipy", "numpy", "joblib"]
# ///
"""Write the trained model for the browser, and pages to check it against.

data/model.lpcm holds everything the browser needs to score a page as
train.py's linear model does, and nothing it has to know beforehand except
how features.py computes the features:

    "LPCM", u32 version (3)
    u32 hash bits, u32 count + u32 char n-gram sizes
    u32 presets, each: str name, f32 threshold, u8 kind, f32 value
        kind 0: a logistic regression, value its intercept
        kind 1: a constant score (too few positives to learn), value it
    u32 matchers, each: str subject, str pattern (MATCHERS)
    u32 literal features, each: u32 count + str strings: 1 when the
        ASCII-lowercased HTML holds any of the strings (VENDORS)
    u32 tags, each: str tag (TAGS)
    u32 word buckets, then u32 bucket[] and f32 idf[], buckets increasing
    u32 char buckets, the same
    u32 structure features, in column order, each a u8 kind then:
        kind 0, a measure the browser computes itself: str name
        kind 1, 2, 3, a matcher, literal feature or tag: u32 its index
    f32 weights[columns][presets]: the word columns, the char columns, then
        the structure ones; a column's weights for every preset together

with every number little-endian and each str a u32 byte length then UTF-8.

data/parity.jsonl holds --pages stored pages (half of them pages Jev or a
hand label found a preset on, half others, at random) with the scores the model gives
them, for the browser's parity test to reproduce: url, title, status,
tree, html and scores (one per preset, in the model's order).

Run:  uv run research/page_classifier/export.py
"""

import argparse
import json
import random
import struct
from pathlib import Path

import joblib
import numpy as np

import features
from features import PRESETS
from train import DATA, load


def string(value: str) -> bytes:
    data = value.encode()
    return struct.pack("<I", len(data)) + data


# A structure column's kind, by where features.py defines its feature.
SECTIONS = [
    (1, [name for name, _, _ in features.MATCHERS]),
    (2, [f"vendor_{name}" for name in features.VENDORS]),
    (3, [f"tag_{tag}" for tag in features.TAGS]),
]


def column(name: str) -> bytes:
    """A structure column as the browser finds its feature: the index of the
    matcher, literal feature or tag it is, or the name of a measure."""
    for kind, names in SECTIONS:
        if name in names:
            return struct.pack("<BI", kind, names.index(name))
    return struct.pack("<B", 0) + string(name)


def model_bytes(model) -> bytes:
    out = [b"LPCM", struct.pack("<II", 3, features.HASH_BITS)]
    out.append(struct.pack(f"<I{len(features.CHAR_SIZES)}I", len(features.CHAR_SIZES), *features.CHAR_SIZES))
    out.append(struct.pack("<I", len(PRESETS)))
    columns = len(model.words.buckets) + len(model.chars.buckets) + len(model.shape.feature_names_)
    weights = np.zeros((columns, len(PRESETS)), dtype="<f4")
    for i, preset in enumerate(PRESETS):
        fitted = model.models[preset]
        constant = isinstance(fitted, float)
        out.append(string(preset) + struct.pack("<fBf", model.thresholds[i], int(constant),
                                                fitted if constant else fitted.intercept_[0]))
        if not constant:
            weights[:, i] = fitted.coef_[0]
    out.append(struct.pack("<I", len(features.MATCHERS)))
    out += [string(subject) + string(pattern) for _, subject, pattern in features.MATCHERS]
    out.append(struct.pack("<I", len(features.VENDORS)))
    out += [struct.pack("<I", len(found)) + b"".join(map(string, found)) for found in features.VENDORS.values()]
    out.append(struct.pack("<I", len(features.TAGS)))
    out += [string(tag) for tag in features.TAGS]
    for family in (model.words, model.chars):
        out.append(struct.pack("<I", len(family.buckets)))
        out.append(family.buckets.astype("<u4").tobytes())
        out.append(family.idf.astype("<f4").tobytes())
    out.append(struct.pack("<I", len(model.shape.feature_names_)))
    out += [column(name) for name in model.shape.feature_names_]
    out.append(weights.tobytes())
    return b"".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", type=Path, default=DATA / "model.joblib")
    parser.add_argument("--pages", type=int, default=300, help="pages in the parity file")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    model = joblib.load(args.model)
    data = model_bytes(model)
    out = DATA / "model.lpcm"
    out.write_bytes(data)
    print(f"{len(data) / 1e6:.1f} MB -> {out}")

    rows = load(DATA / "pages.jsonl", 0.7, DATA / "gold.jsonl")
    rng = random.Random(args.seed)
    positive = [r for r in rows if any(r["label"])]
    rest = [r for r in rows if not any(r["label"])]
    half = min(len(positive), args.pages // 2)
    picked = rng.sample(positive, half) + rng.sample(rest, min(len(rest), args.pages - half))
    scores = model.predict(picked)
    parity = DATA / "parity.jsonl"
    with parity.open("w") as f:
        for row, score in zip(picked, scores):
            f.write(json.dumps({key: row.get(key) for key in ("url", "title", "status", "tree", "html")}
                               | {"scores": score.tolist()}) + "\n")
    print(f"{len(picked)} pages, {sum(any(r['label']) for r in picked)} with a preset -> {parity}")


if __name__ == "__main__":
    main()
