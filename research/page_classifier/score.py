# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["onnxruntime", "tokenizers", "numpy", "lightpanda"]
#
# [tool.uv.sources]
# lightpanda = { path = "../..", editable = true }
# ///
"""Answer the classify presets with the fine-tuned transformer, on the CPU.

Needs only ONNX Runtime and the tokenizer: data/transformer.onnx (or its
int8 quantization) from `transformer.py export`, and data/transformer/ for
the tokenizer, the window count and the thresholds. Scores are calibrated as
the browser calibrates the linear model's (a preset's threshold maps to
0.5), and a page is sure when every preset is below --sure or above 1 minus
it: the pages a cascade would answer without Jev.

  page     score pages: --url URLs, loaded in the browser (classify's own
           input, so TYPESAFE_API_KEY is needed), or --from-folds N pages
           already crawled
  bench    time the ONNX files and the linear model on this CPU, one page
           at a time, and write data/costs.json for compare.py

Run:  uv run research/page_classifier/score.py page --url https://example.com
"""

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

HERE = Path(__file__).parent
DATA = HERE / "data"


class Scorer:
    def __init__(self, onnx: Path = DATA / "transformer.int8.onnx", model: Path = DATA / "transformer", threads: int = 0):
        self.meta = json.loads((model / "meta.json").read_text())
        self.presets = self.meta["presets"]
        self.thresholds = np.array(self.meta["thresholds"])
        self.tokenizer = Tokenizer.from_file(str(model / "encoder" / "tokenizer.json"))
        self.start, self.end = self.tokenizer.token_to_id("<s>"), self.tokenizer.token_to_id("</s>")
        self.pad = self.tokenizer.token_to_id("<pad>")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(onnx), options, providers=["CPUExecutionProvider"])

    def windows(self, page: dict) -> tuple[np.ndarray, np.ndarray]:
        """The page as transformer.py's windows(): classify's input after e5's
        passage prefix, cut into windows of max_tokens."""
        text = f"passage: {page['url']}\n{page.get('title') or ''}\nstatus {page.get('status') or 0}\n{page.get('tree') or ''}"
        ids = self.tokenizer.encode(text, add_special_tokens=False).ids
        body = self.meta["max_tokens"] - 2
        cut = [[self.start, *ids[i:i + body], self.end] for i in range(0, max(len(ids), 1), body)][:self.meta["windows"]]
        width = max(len(w) for w in cut)
        input_ids = np.full((len(cut), width), self.pad, dtype=np.int64)
        mask = np.zeros((len(cut), width), dtype=np.int64)
        for i, w in enumerate(cut):
            input_ids[i, :len(w)] = w
            mask[i, :len(w)] = 1
        return input_ids, mask

    def scores(self, page: dict) -> np.ndarray:
        """Each preset's score, the highest across the page's windows."""
        input_ids, mask = self.windows(page)
        return self.session.run(None, {"input_ids": input_ids, "attention_mask": mask})[0].max(axis=0)

    def calibrated(self, page: dict) -> np.ndarray:
        s, t = self.scores(page), self.thresholds
        return np.where(s < t, 0.5 * s / t, 0.5 + 0.5 * (s - t) / (1 - t))

    def judge(self, page: dict, sure: float = 0.1) -> dict:
        """Calibrated scores by preset name, and whether the cascade would
        answer this page without Jev."""
        c = self.calibrated(page)
        return {"scores": dict(zip(self.presets, c.round(3).tolist())), "fired": [p for p, v in zip(self.presets, c) if v >= 0.5],
                "sure": bool(((c < sure) | (c > 1 - sure)).all())}


def from_browser(urls: list[str]) -> list[dict]:
    """classify's own input for each url, as the browser builds it."""
    from lightpanda import Browser

    pages = []
    with Browser() as browser:
        for url in urls:
            with browser.new_session() as session:
                session.goto(url=url)
                result = session.classify(questions={"isBlocked": True}, include_input=True)
                pages.append({**result["input"], "url": result["input"].get("url") or url})
    return pages


def cmd_page(args):
    scorer = Scorer(args.onnx, args.model)
    if args.url:
        pages = from_browser(args.url)
    else:
        pages = [json.loads(line) for _, line in zip(range(args.from_folds), (DATA / "folds.jsonl").open())]
    for page in pages:
        start = perf_counter()
        verdict = scorer.judge(page, args.sure)
        took = (perf_counter() - start) * 1000
        jev = [p for p, v in zip(scorer.presets, page.get("vote") or []) if v == 1]
        print(f"{page['url'][:60]:<60}  {', '.join(verdict['fired']) or 'ok':<44}"
              f"{'sure' if verdict['sure'] else 'ask Jev':<9}{took:6.0f} ms" + (f"   Jev: {', '.join(jev) or 'ok'}" if "vote" in page else ""))


def cmd_bench(args):
    pages = [json.loads(line) for _, line in zip(range(args.pages), (DATA / "folds.jsonl").open())]
    costs = {}
    meta_path = DATA / "train_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        ms = meta["linear_ms_per_page"]
        costs["linear (Python)"] = {"ms": ms, "ms_one_thread": ms, "mb": meta["model_bytes"] / 1e6, "note": "train.py, sklearn"}
    lpcm = DATA / "model.lpcm"
    if lpcm.exists():
        costs["linear (Zig, browser)"] = {"ms": 1.76, "ms_one_thread": 1.76, "mb": lpcm.stat().st_size / 1e6, "note": "release build, measured by the parity test"}
    variant = json.loads((args.model / "meta.json").read_text())["variant"]
    for onnx in (DATA / "transformer.onnx", DATA / "transformer.int8.onnx"):
        if not onnx.exists():
            continue
        cell = {"mb": onnx.stat().st_size / 1e6, "note": f"ft-{variant}, ONNX Runtime"}
        for key, threads in (("ms", 0), ("ms_one_thread", 1)):
            scorer = Scorer(onnx, args.model, threads)
            scorer.scores(pages[0])  # warm up
            start = perf_counter()
            for page in pages:
                scorer.scores(page)
            cell[key] = (perf_counter() - start) * 1000 / len(pages)
        costs[f"ft-{variant} {'int8' if 'int8' in onnx.name else 'fp32'}"] = cell
        print(f"{onnx.name}: {cell['ms']:.1f} ms/page on {os.cpu_count()} threads, {cell['ms_one_thread']:.1f} ms on 1, {cell['mb']:.0f} MB")
    (DATA / "costs.json").write_text(json.dumps(costs, indent=1))
    print(f"-> {DATA / 'costs.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=["page", "bench"])
    parser.add_argument("--url", nargs="+", help="page: urls to load and score")
    parser.add_argument("--from-folds", type=int, default=10, help="page: score this many crawled pages instead")
    parser.add_argument("--pages", type=int, default=100, help="bench: pages to time")
    parser.add_argument("--sure", type=float, default=0.1)
    parser.add_argument("--onnx", type=Path, default=DATA / "transformer.int8.onnx")
    parser.add_argument("--model", type=Path, default=DATA / "transformer")
    args = parser.parse_args()
    {"page": cmd_page, "bench": cmd_bench}[args.command](args)


if __name__ == "__main__":
    main()
