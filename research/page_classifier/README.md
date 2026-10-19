# Local page classifier

Can a local model answer the browser's `classify` presets (`isBlocked`,
`isCaptcha`, `isConsentWall`, … 12 in all) well enough to skip TypeSafe's Jev
for most pages? Every model here is a student of Jev: it learns Jev's
answers on crawled pages, and the useful measure is how many pages it can
answer **locally while still agreeing with Jev**, leaving the rest to Jev.

## Pipeline

| step | script | output |
|---|---|---|
| URL list (Tranco + seeds for rare presets) | `seeds.py`, `articles.py` | `data/urls.tsv` |
| crawl, label each page with two Jev calls | `collect.py` (`extend.py` for presets added later) | `data/pages.jsonl` |
| hand-check doubtful labels | `review.py`, `label.py` | `data/gold.jsonl` |
| linear model + browser export | `train.py`, `evaluate.py`, `export.py` | `data/model.lpcm` |
| 5 site-grouped folds | `folds.py` | `data/folds.jsonl` |
| rules + linear, out of fold | `crossval.py` | `data/oof/{rules,linear}.jsonl` |
| transformers, out of fold; final fit; ONNX | `transformer.py` (on a GPU) | `data/oof/*.jsonl`, `data/transformer*` |
| comparison | `compare.py` | `data/comparison.{txt,png}` |
| CPU scorer + timings | `score.py` | `data/costs.json` |

Labels: a preset is yes when both Jev calls score it ≥ 0.7, no when both
score it ≤ 0.3, and undecided (left out of training and scoring) otherwise;
`gold.jsonl` overrides Jev where it was checked by hand.

## Models

- **rules**: hand-written vendor fingerprints and phrases (`features.rules`).
- **linear**: TF-IDF over hashed word and character n-grams of what classify
  sends Jev (url, title, status, semantic tree), plus structural features
  from the HTML; one logistic regression per preset. Runs in the browser
  (`--classify-model`, `src/browser/classify_local.zig`).
- **frozen**: `multilingual-e5-small` embeddings of the whole tree, logistic
  regression on top.
- **ft-small / ft-small-long / ft-small-balanced / ft-base**: e5 encoders
  fine-tuned end to end (first 512 tokens; up to four 512-token windows;
  uncapped positive weights and 8 epochs; the 278M-parameter base model).
- **ensemble**: the mean of linear's and ft-base's calibrated scores.

## Results

5-fold site-grouped cross-validation over 3,394 pages from 2,325 sites
(`compare.py`); every score comes from a model that never saw the page's
site. Agreement is on judge()'s keep-or-skip verdict, over the 2,468 pages
with every preset decided; gold is the 409 hand-checked pages.

| model | verdict vs Jev | gold accuracy | local at ≥ 99% agreement | CPU per page | size |
|---|---|---|---|---|---|
| Jev's vote | — | 91.0% | — | ~250 ms + network, per call | — |
| rules | 76.6% | 70.9% | — | — | — |
| linear | 95.4% | 84.1% | 61.5% | 1.8 ms (Zig) | 17 MB |
| frozen e5-small | 86.3% | 80.7% | 18.1% | | |
| ft-small | 88.4% | 87.5% | 33.9% | | |
| ft-small-long | 95.1% | 85.1% | 43.6% | | |
| ft-small-balanced | 96.1% | 86.3% | 8.2% | | |
| **ft-base** | 95.9% | 87.0% | 69.2% | 68 ms int8 (8 threads), 271 ms (1) | 278 MB |
| **ensemble** | 95.6% | 87.5% | **81.1%** | both | both |

![cascade trade-off](comparison.png)

What the numbers say:

- **The base model is the only transformer that beats the n-grams.** It
  agrees with Jev slightly more often, is right on the hand-checked pages
  more often (87.0% vs 84.1%, against Jev's own 91.0%) and is sure of more
  pages at the same agreement. The small models, frozen or fine-tuned, are
  worse than linear for the cascade.
- **Linear and the transformer make different mistakes, so together they
  are best**: 81% of pages answered locally at 99% agreement, against 62%
  for linear alone.
- **Rare presets stay hard for every transformer.** Paywall (87 positives),
  loading (17), unsupported browser (8) and rate limits (10) have too few
  examples to fine-tune on; linear's keyword features do better on them
  (paywall F1 0.57 vs 0.44). Reading the whole tree (ft-small-long) did not
  help paywalls, so truncation is not the cause; weighting the positives up
  (ft-small-balanced) made the scores overconfident, which ruins the cascade.
- **Int8 quantization** shrinks the ONNX file from 1.1 GB to 278 MB and
  halves CPU time; it flips 0.7% of verdicts against fp32 (all from the
  matmuls: quantizing only the embeddings flips none).

### Staged cascade

Running the transformer only where linear is unsure keeps most of the gain:

| linear band, then ensemble band | answered locally | agreement | pages that run the transformer |
|---|---|---|---|
| 0.05 (linear only) | 61.5% | 99.4% | 0% |
| 0.05, then 0.10 | 76.5% | 99.1% | 38.5% |
| 0.02, then 0.15 | 81.4% | 99.0% | 56.5% |

## Recommendation

Keep the linear model in the browser as the first stage: 1.8 ms and 17 MB
answer about 60–75% of pages. The fine-tuned base model is worth adding as a
second stage only where those extra ~15 points of pages matter more than
~70 ms of CPU and a 278 MB model plus ONNX Runtime and the XLM-R tokenizer in
the browser. A cheaper next step for the rare presets is more data: they
are what both models miss, and a targeted crawl moved paywall F1 from 0.44
to 0.57 for linear.

## Reproduce

```bash
uv run research/page_classifier/folds.py
uv run research/page_classifier/crossval.py
# on a CUDA machine, with data/folds.jsonl and transformer.py copied over:
uv run transformer.py embed
uv run transformer.py cv --variant small        # also small-long, base
uv run transformer.py cv --variant small --epochs 8 --pos-cap 1000 --out data/oof/ft-small-balanced.jsonl
uv run transformer.py fit --variant base && uv run transformer.py export
# back here, with data/oof/*, data/transformer*.onnx and data/transformer/ copied back:
uv run research/page_classifier/score.py bench
uv run research/page_classifier/compare.py
uv run research/page_classifier/score.py page --url https://example.com   # needs TYPESAFE_API_KEY
```

On a Quadro RTX 5000, a 5-fold run takes 12 minutes for e5-small and
35 for e5-base.
