# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#     "torch>=2.6",
#     "transformers>=4.46",
#     "scikit-learn>=1.5",
#     "numpy",
#     "onnx",
#     "onnxruntime",
# ]
#
# [tool.uv.sources]
# torch = { index = "pytorch-cu126" }
#
# [[tool.uv.index]]
# name = "pytorch-cu126"
# url = "https://download.pytorch.org/whl/cu126"
# explicit = true
# ///
"""Fine-tune a pretrained multilingual transformer on the classify presets.

The linear model sees n-grams; a transformer reads the page. This trains
intfloat/multilingual-e5 encoders (pretrained on text in ~100 languages) to
answer the 12 presets from what classify sends Jev: url, title, status and
the semantic tree. One sigmoid per preset over the mean-pooled encoding, and
a preset Jev left undecided adds nothing to the loss, as in train.py.

The tree runs to ~2.5k tokens and the encoders read 512 at a time, so a
variant reads either the first 512 tokens or up to `chunks` windows of them,
taking each preset's highest score across the windows.

Subcommands, all reading data/folds.jsonl (folds.py):

  embed    frozen e5-small embeddings (every window, averaged) + one logistic
           regression per preset, cross-validated -> data/oof/frozen.jsonl
  cv       fine-tune --variant on each fold's other folds and score the fold
           -> data/oof/ft-<variant>.jsonl
  fit      fine-tune --variant on every page -> data/transformer/
  export   data/transformer/ -> data/transformer.onnx and its int8
           quantization data/transformer.int8.onnx, checked against torch

Runs on a CUDA GPU when there is one (fp16), else on the CPU (slowly).

Run:  uv run research/page_classifier/transformer.py cv --variant small
"""

import argparse
import json
import math
import random
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch import nn
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

HERE = Path(__file__).parent
DATA = HERE / "data"
PRESETS = [
    "is_blocked", "is_captcha", "is_consent_wall", "is_empty_catalog", "is_error_page", "is_login_wall",
    "is_paywall", "is_unsupported_browser", "is_loading", "is_age_gate", "is_region_blocked", "is_rate_limited",
]
ENCODERS = {"small": "intfloat/multilingual-e5-small", "base": "intfloat/multilingual-e5-base"}
# encoder, windows of 512 tokens read per page, pages per step, steps per update
VARIANTS = {
    "small": ("small", 1, 16, 1),
    "small-long": ("small", 4, 4, 4),
    "base": ("base", 1, 8, 2),
}
MAX_TOKENS = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def text(row: dict) -> str:
    """The page as classify sends it to Jev, after e5's passage prefix."""
    return f"passage: {row['url']}\n{row['title'] or ''}\nstatus {row['status'] or 0}\n{row['tree'] or ''}"


def windows(tokenizer, rows: list[dict], limit: int) -> list[list[list[int]]]:
    """Each page's tokens cut into at most `limit` windows of MAX_TOKENS,
    each wrapped in the encoder's start and end tokens."""
    body = MAX_TOKENS - 2
    ids = tokenizer([text(r) for r in rows], add_special_tokens=False)["input_ids"]
    return [
        [[tokenizer.cls_token_id, *seq[i:i + body], tokenizer.sep_token_id] for i in range(0, max(len(seq), 1), body)][:limit]
        for seq in ids
    ]


def batch(pages: list[list[list[int]]], pad: int):
    """Pad a batch of pages' windows into one tensor; `owner` says which page
    each window belongs to."""
    flat = [w for page in pages for w in page]
    owner = torch.tensor([i for i, page in enumerate(pages) for _ in page])
    width = max(len(w) for w in flat)
    ids = torch.full((len(flat), width), pad)
    mask = torch.zeros((len(flat), width), dtype=torch.long)
    for i, w in enumerate(flat):
        ids[i, :len(w)] = torch.tensor(w)
        mask[i, :len(w)] = 1
    return ids.to(DEVICE), mask.to(DEVICE), owner.to(DEVICE)


def pool(hidden, mask):
    """Mean of the token encodings, padding left out (as e5 is trained)."""
    m = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * m).sum(1) / m.sum(1).clamp(min=1)


class Classifier(nn.Module):
    def __init__(self, encoder: str, **options):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder, **options)
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(self.encoder.config.hidden_size, len(PRESETS))

    def forward(self, ids, mask):
        """One logit per preset for each window."""
        return self.head(self.dropout(pool(self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state, mask)))


def page_logits(model, ids, mask, owner, pages: int):
    """Each preset's highest logit across a page's windows."""
    logits = model(ids, mask).float()
    out = torch.full((pages, len(PRESETS)), -torch.inf, device=logits.device)
    return out.scatter_reduce(0, owner.unsqueeze(1).expand_as(logits), logits, "amax")


def labels_of(rows: list[dict]) -> np.ndarray:
    """The labels to learn, with -1 where Jev left a preset undecided."""
    return np.array([[-1 if v is None else v for v in r["label"]] for r in rows], dtype=np.float32)


def train(variant: str, rows: list[dict], pages: list, epochs: int, seed: int, pos_cap: float = 20) -> Classifier:
    encoder, _, per_step, accumulate = VARIANTS[variant]
    torch.manual_seed(seed)
    rng = random.Random(seed)
    tokenizer = AutoTokenizer.from_pretrained(ENCODERS[encoder])
    model = Classifier(ENCODERS[encoder]).to(DEVICE)

    y = labels_of(rows)
    pos, neg = (y == 1).sum(0), (y == 0).sum(0)
    pos_weight = torch.tensor(np.clip(neg / np.maximum(pos, 1), 1, pos_cap), device=DEVICE)
    loss_of = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": 3e-5},
        {"params": model.head.parameters(), "lr": 1e-3},
    ], weight_decay=0.01)
    updates = epochs * math.ceil(len(rows) / (per_step * accumulate))
    schedule = get_linear_schedule_with_warmup(optimizer, int(0.1 * updates), updates)
    cuda = DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=cuda)

    order = list(range(len(rows)))
    for epoch in range(epochs):
        start, total = perf_counter(), 0.0
        model.train()
        rng.shuffle(order)
        steps = [order[i:i + per_step] for i in range(0, len(order), per_step)]
        for step, chosen in enumerate(steps):
            ids, mask, owner = batch([pages[i] for i in chosen], tokenizer.pad_token_id)
            target = torch.tensor(y[chosen], device=DEVICE)
            with torch.autocast(DEVICE.type, dtype=torch.float16, enabled=cuda):
                logits = page_logits(model, ids, mask, owner, len(chosen))
            known = (target >= 0).float()
            loss = (loss_of(logits, target.clamp(min=0)) * known).sum() / known.sum().clamp(min=1)
            scaler.scale(loss / accumulate).backward()
            if (step + 1) % accumulate == 0 or step + 1 == len(steps):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                schedule.step()
            total += loss.item()
        print(f"  epoch {epoch + 1}/{epochs}: loss {total / len(steps):.4f}, {perf_counter() - start:.0f}s", flush=True)
    return model


@torch.no_grad()
def predict(model: Classifier, pages: list, pad: int, per_step: int = 32) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(pages), per_step):
        chosen = pages[i:i + per_step]
        ids, mask, owner = batch(chosen, pad)
        with torch.autocast(DEVICE.type, dtype=torch.float16, enabled=DEVICE.type == "cuda"):
            out.append(torch.sigmoid(page_logits(model, ids, mask, owner, len(chosen))).cpu().numpy())
    return np.concatenate(out)


def mean_auc(rows: list[dict], scores: np.ndarray) -> float:
    y, aucs = labels_of(rows), []
    for j in range(len(PRESETS)):
        known = y[:, j] >= 0
        if len(set(y[known, j])) == 2:
            aucs.append(roc_auc_score(y[known, j], scores[known, j]))
    return float(np.mean(aucs))


def write(path: Path, rows: list[dict], scores: np.ndarray) -> None:
    path.parent.mkdir(exist_ok=True)
    with path.open("w") as out:
        for row, s in zip(rows, scores):
            out.write(json.dumps({"url": row["url"], "scores": np.round(s, 5).tolist()}) + "\n")


def read_rows(path: Path, limit: int | None) -> list[dict]:
    rows = [json.loads(line) for line in path.open()]
    return rows[:limit] if limit else rows


def cmd_cv(args):
    rows = read_rows(args.folds, args.limit)
    encoder, limit, _, _ = VARIANTS[args.variant]
    tokenizer = AutoTokenizer.from_pretrained(ENCODERS[encoder])
    pages = windows(tokenizer, rows, limit)
    fold = np.array([r["fold"] for r in rows])
    scores = np.zeros((len(rows), len(PRESETS)), dtype=np.float32)
    for f in sorted(set(fold)):
        start = perf_counter()
        fit_idx, test_idx = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
        print(f"fold {f}: training on {len(fit_idx)} pages", flush=True)
        model = train(args.variant, [rows[i] for i in fit_idx], [pages[i] for i in fit_idx], args.epochs, args.seed + int(f), args.pos_cap)
        scores[test_idx] = predict(model, [pages[i] for i in test_idx], tokenizer.pad_token_id)
        print(f"fold {f}: mean AUC {mean_auc([rows[i] for i in test_idx], scores[test_idx]):.3f}, "
              f"{perf_counter() - start:.0f}s", flush=True)
        del model
        torch.cuda.empty_cache()
    out = args.out or DATA / "oof" / f"ft-{args.variant}.jsonl"
    write(out, rows, scores)
    print(f"{len(rows)} pages -> {out}")


@torch.no_grad()
def cmd_embed(args):
    rows = read_rows(args.folds, args.limit)
    tokenizer = AutoTokenizer.from_pretrained(ENCODERS["small"])
    encoder = AutoModel.from_pretrained(ENCODERS["small"]).to(DEVICE).eval()
    pages = windows(tokenizer, rows, 64)  # every window: the whole tree
    start, embedded = perf_counter(), []
    for i in range(0, len(pages), 16):
        chosen = pages[i:i + 16]
        ids, mask, owner = batch(chosen, tokenizer.pad_token_id)
        with torch.autocast(DEVICE.type, dtype=torch.float16, enabled=DEVICE.type == "cuda"):
            pooled = pool(encoder(input_ids=ids, attention_mask=mask).last_hidden_state, mask).float()
        sums = torch.zeros((len(chosen), pooled.shape[1]), device=DEVICE).index_add_(0, owner, pooled)
        counts = torch.bincount(owner, minlength=len(chosen)).unsqueeze(1)
        embedded.append(nn.functional.normalize(sums / counts, dim=1).cpu().numpy())
    x = np.concatenate(embedded)
    print(f"embedded {len(rows)} pages in {perf_counter() - start:.0f}s", flush=True)

    y, fold = labels_of(rows), np.array([r["fold"] for r in rows])
    scores = np.zeros((len(rows), len(PRESETS)), dtype=np.float32)
    for f in sorted(set(fold)):
        fit_idx, test_idx = fold != f, fold == f
        for j in range(len(PRESETS)):
            known = fit_idx & (y[:, j] >= 0)
            if len(set(y[known, j])) < 2:
                continue
            lr = LogisticRegression(C=4.0, class_weight="balanced", max_iter=2000)
            scores[test_idx, j] = lr.fit(x[known], y[known, j]).predict_proba(x[test_idx])[:, 1]
    out = args.out or DATA / "oof" / "frozen.jsonl"
    write(out, rows, scores)
    print(f"mean AUC {mean_auc(rows, scores):.3f}; {len(rows)} pages -> {out}")


def best_thresholds(rows: list[dict], scores: np.ndarray) -> list[float]:
    """Each preset's threshold with the best F1 on out-of-fold scores."""
    y, out = labels_of(rows), []
    for j in range(len(PRESETS)):
        known = y[:, j] >= 0
        truth = y[known, j] == 1
        # The best F1 is at some positive's score (compare.py does the same).
        grid = np.unique(scores[known, j][truth])
        grid = grid[(grid > 0) & (grid < 1)]
        if truth.sum() < 3 or not len(grid):
            out.append(0.5)
            continue
        f1s = []
        for t in grid:
            guess = scores[known, j] >= t
            tp = (guess & truth).sum()
            f1s.append(2 * tp / max(guess.sum() + truth.sum(), 1))
        out.append(float(grid[int(np.argmax(f1s))]))
    return out


def cmd_fit(args):
    rows = read_rows(args.folds, args.limit)
    encoder, limit, _, _ = VARIANTS[args.variant]
    tokenizer = AutoTokenizer.from_pretrained(ENCODERS[encoder])
    model = train(args.variant, rows, windows(tokenizer, rows, limit), args.epochs, args.seed, args.pos_cap)
    oof_path = DATA / "oof" / f"ft-{args.variant}.jsonl"
    oof = np.array([json.loads(line)["scores"] for line in oof_path.open()]) if oof_path.exists() else None
    thresholds = best_thresholds(rows, oof) if oof is not None and len(oof) == len(rows) else [0.5] * len(PRESETS)

    out = args.out or DATA / "transformer"
    out.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(out / "encoder")
    tokenizer.save_pretrained(out / "encoder")
    torch.save(model.head.state_dict(), out / "head.pt")
    meta = {"variant": args.variant, "encoder": ENCODERS[encoder], "windows": limit, "max_tokens": MAX_TOKENS,
            "presets": PRESETS, "thresholds": thresholds, "trained_on": len(rows), "epochs": args.epochs}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"thresholds: {thresholds}\n-> {out}")


class Exported(nn.Module):
    """The classifier as the ONNX file runs it: windows in, a sigmoid per
    preset per window out (the scorer takes the max across a page's windows)."""

    def __init__(self, model: Classifier):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return torch.sigmoid(self.model(input_ids, attention_mask))


def load_trained(path: Path, **options) -> tuple[Classifier, dict]:
    meta = json.loads((path / "meta.json").read_text())
    model = Classifier(str(path / "encoder"), **options)
    model.head.load_state_dict(torch.load(path / "head.pt", map_location="cpu"))
    return model.eval(), meta


@torch.no_grad()
def cmd_export(args):
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic

    global DEVICE
    DEVICE = torch.device("cpu")  # ONNX Runtime checks against torch on the CPU
    path = args.model or DATA / "transformer"
    # Eager attention traces into plain ops; SDPA's mask handling branches on
    # the example's shapes, which the trace would then bake in.
    model, meta = load_trained(path, attn_implementation="eager")
    tokenizer = AutoTokenizer.from_pretrained(path / "encoder")
    fp32, int8 = DATA / "transformer.onnx", DATA / "transformer.int8.onnx"
    # A full window and a padded one, so neither the length nor the padding
    # is a special case in the trace.
    ids = torch.full((2, MAX_TOKENS), 1000)
    ids[:, 0], ids[0, -1], ids[1, 99] = tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.sep_token_id
    mask = torch.ones_like(ids)
    ids[1, 100:], mask[1, 100:] = tokenizer.pad_token_id, 0
    torch.onnx.export(
        Exported(model).eval(), (ids, mask), str(fp32), input_names=["input_ids", "attention_mask"],
        output_names=["scores"], dynamic_axes={"input_ids": {0: "windows", 1: "tokens"},
                                               "attention_mask": {0: "windows", 1: "tokens"}, "scores": {0: "windows"}},
        opset_version=17, dynamo=False,
    )
    # One scale per tensor, embeddings included: on 300 pages it flipped 0.7%
    # of verdicts, all from the matmuls (quantizing only the embeddings
    # flipped none); one scale per channel flipped 1.7%.
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)

    # Check both files against torch on real pages. export() puts the modules
    # back in the mode it found them, so make sure dropout is off.
    model.eval()
    rows = read_rows(args.folds, 100)
    pages = windows(tokenizer, rows, meta["windows"])
    reference = np.concatenate([torch.sigmoid(page_logits(model, *batch([p], tokenizer.pad_token_id), 1)).numpy() for p in pages])
    for name in (fp32, int8):
        session = ort.InferenceSession(str(name), providers=["CPUExecutionProvider"])
        got = []
        for p in pages:
            ids_, mask_, _ = batch([p], tokenizer.pad_token_id)
            got.append(session.run(None, {"input_ids": ids_.cpu().numpy(), "attention_mask": mask_.cpu().numpy()})[0].max(0))
        got = np.array(got)
        flips = ((got >= np.array(meta["thresholds"])) != (reference >= np.array(meta["thresholds"]))).any(axis=1).mean()
        print(f"{name.name}: {name.stat().st_size / 1e6:.0f} MB, max |diff| vs torch {np.abs(got - reference).max():.2e}, "
              f"{flips:.1%} of pages fire a different preset")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=["embed", "cv", "fit", "export"])
    parser.add_argument("--variant", choices=list(VARIANTS), default="small")
    parser.add_argument("--folds", type=Path, default=DATA / "folds.jsonl")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--pos-cap", type=float, default=20, help="cap on a preset's positive weight (negatives per positive)")
    parser.add_argument("--limit", type=int, help="use only the first N pages (smoke tests)")
    parser.add_argument("--model", type=Path, help="export: the trained model directory")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    print(f"device: {DEVICE}" + (f" ({torch.cuda.get_device_name()})" if DEVICE.type == "cuda" else ""), flush=True)
    {"embed": cmd_embed, "cv": cmd_cv, "fit": cmd_fit, "export": cmd_export}[args.command](args)


if __name__ == "__main__":
    main()
