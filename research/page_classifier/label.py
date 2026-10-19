# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Check pages by hand, most suspicious first, into the gold set.

Walks data/review.jsonl (from review.py) and shows each page as Jev saw it:
url, status, title, the start of the semantic tree, Jev's score and the
model's for every preset, and why the page is in doubt. Type what the page
really is and it is appended to data/gold.jsonl, every preset true or false.
Pages already in the gold set are skipped, so a session can stop anytime and
resume later.

At the prompt:
  (enter)          none: the page shows its content
  2 3 / consent    the presets that apply, by number or by a name's prefix
  j                Jev was right: take its vote as it stands
  t                show the whole tree
  h                show the HTML without scripts and styles: where the tree
                   is nearly empty (a challenge or app shell), the markup
                   often still says what the page is. Shown by itself when
                   the tree has SHORT_TREE lines or fewer.
  o                open the url in your browser (it may not match what
                   Lightpanda saw: the tree is the record)
  s                skip this page for now
  u                undo the last label
  q                quit

Run:  uv run research/page_classifier/label.py
"""

import argparse
import json
import re
import sys
import webbrowser
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE / "data"

PRESETS = [
    "is_blocked",
    "is_captcha",
    "is_consent_wall",
    "is_empty_catalog",
    "is_error_page",
    "is_login_wall",
    "is_paywall",
    "is_unsupported_browser",
    "is_loading",
    "is_age_gate",
    "is_region_blocked",
    "is_rate_limited",
]
NAMES = [p.removeprefix("is_") for p in PRESETS]
PREVIEW_LINES = 40
SHORT_TREE = 4
HTML_BYTES = 3000

SCRIPTS = re.compile(r"<(script|style|svg)\b.*?</\1>", re.IGNORECASE | re.DOTALL)

TTY = sys.stdout.isatty()


def paint(text, color):
    codes = {"green": "32", "red": "31", "yellow": "33", "dim": "2", "bold": "1"}
    return f"\033[{codes[color]}m{text}\033[0m" if TTY else text


def parse(answer: str) -> list[str] | None:
    """The presets an answer names, or None when a word matches no preset."""
    picked = []
    for word in answer.replace(",", " ").split():
        if word.isdigit() and 1 <= int(word) <= len(PRESETS):
            picked.append(PRESETS[int(word) - 1])
            continue
        matches = [p for p, name in zip(PRESETS, NAMES) if name.startswith(word.lower())]
        if len(matches) != 1:
            print(paint(f"  '{word}' is {'ambiguous' if matches else 'no preset'}", "red"))
            return None
        picked.append(matches[0])
    return picked


def jev_scores(page: dict) -> list[float | None]:
    """Each preset's mean over Jev's calls."""
    out = []
    for preset in PRESETS:
        scores = [c[preset] for c in page["jev"] if c.get(preset) is not None]
        out.append(sum(scores) / len(scores) if scores else None)
    return out


def markup(page: dict) -> str:
    """The page's HTML without scripts, styles and SVG, whitespace squeezed."""
    html = re.sub(r"\s+", " ", SCRIPTS.sub("", page.get("html") or "")).strip()
    more = f" … {len(html) - HTML_BYTES} more bytes" if len(html) > HTML_BYTES else ""
    return html[:HTML_BYTES] + more


def show(item: dict, page: dict, position: str):
    print("\n" + paint("─" * 78, "dim"))
    print(f"{paint(position, 'dim')}  {paint(page['url'], 'bold')}")
    print(f"status {page.get('status')}   title {page.get('title')!r}")
    tree = page["tree"].splitlines()
    print(paint("\n".join(tree[:PREVIEW_LINES]), "dim"))
    if len(tree) > PREVIEW_LINES:
        print(paint(f"  … {len(tree) - PREVIEW_LINES} more lines (t shows them)", "dim"))
    if len(tree) <= SHORT_TREE:
        print(paint("the tree is nearly empty; the HTML:", "yellow"))
        print(markup(page))

    doubts = {d["preset"]: d for d in item["doubts"]}
    print()
    for i, (preset, score) in enumerate(zip(PRESETS, jev_scores(page)), 1):
        doubt = doubts.get(preset)
        jev = "  -  " if score is None else f"{score:.2f} "
        mark = paint("yes", "red") if score is not None and score >= 0.5 else "   "
        model = f"model {doubt['model']:.2f}" if doubt else ""
        why = paint("; ".join(doubt["why"]), "yellow") if doubt else ""
        print(f"  {i} {NAMES[i - 1]:<20} jev {jev}{mark}  {model:<12} {why}")
    print(paint("\n  every preset that applies, space separated (1 2, or bl cap) · enter: none · "
                "j: as jev · t: tree · h: html · o: open · s: skip · u: undo · q: quit", "dim"))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--review", type=Path, default=DATA / "review.jsonl")
    parser.add_argument("--pages", type=Path, default=DATA / "pages.jsonl")
    parser.add_argument("--gold", type=Path, default=DATA / "gold.jsonl")
    args = parser.parse_args()

    pages = {}
    for line in args.pages.open():
        page = json.loads(line)
        if "error" not in page:
            pages.setdefault(page["url"], page)
    queue = [item for item in map(json.loads, args.review.open()) if item["url"] in pages]
    done = {json.loads(line)["url"] for line in args.gold.open()} if args.gold.exists() else set()
    todo = [item for item in queue if item["url"] not in done]
    print(f"{len(done)} pages in the gold set, {len(todo)} of {len(queue)} left in the review queue")

    labelled = []  # indices into todo, in the order they were saved
    i = 0
    while i < len(todo):
        item = todo[i]
        page = pages[item["url"]]
        show(item, page, f"[{len(done) + len(labelled) + 1}, {len(todo) - i} left]")
        while True:
            try:
                answer = input(paint("labels> ", "green")).strip()
            except EOFError:
                answer = "q"
            if answer == "q":
                print(f"{len(labelled)} labelled this session -> {args.gold}")
                return
            if answer == "t":
                print(page["tree"])
                continue
            if answer == "h":
                print(markup(page))
                continue
            if answer == "o":
                webbrowser.open(page["url"])
                continue
            if answer == "s":
                i += 1
                break
            if answer == "u":
                if not labelled:
                    print(paint("  nothing to undo", "red"))
                    continue
                lines = args.gold.read_text().splitlines(keepends=True)
                args.gold.write_text("".join(lines[:-1]))
                i = labelled.pop()
                print(paint("  undone", "dim"))
                break
            if answer == "j":
                picked = [p for p, s in zip(PRESETS, jev_scores(page)) if s is not None and s >= 0.5]
            else:
                picked = parse(answer)
                if picked is None:
                    continue
            labels = {p: p in picked for p in PRESETS}
            with args.gold.open("a") as f:
                f.write(json.dumps({"url": page["url"], "labels": labels, "by": "human"}) + "\n")
            labelled.append(i)
            print(paint("  saved: " + (", ".join(NAMES[PRESETS.index(p)] for p in picked) or "none"), "dim"))
            i += 1
            break
    print(f"review queue done, {len(labelled)} labelled this session -> {args.gold}")


if __name__ == "__main__":
    main()
