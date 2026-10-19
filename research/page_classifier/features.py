"""What the local models see of a page, shared by train.py and evaluate.py.

Three views of one row from collect.py:

- `text`: the same input Jev gets (url, title, status, semantic tree), for
  the bag-of-n-grams model.
- `structure`: a few dozen numbers and flags from the tree and the HTML,
  such as the status, the password field and known captcha or consent
  vendors' scripts.
- `rules`: hand-written preset guesses from those flags, the baseline any
  model has to beat.

The n-grams of `text` (`ngrams`) are defined on its UTF-8 bytes, so that the
browser can compute the very same ones without Unicode tables: only ASCII
letters are lowercased, and every byte from 0x80 up counts as a word
character. Each n-gram is hashed with MurmurHash3 (x86, 32 bits, seed 0)
into one of 2**HASH_BITS buckets, so the model needs no vocabulary.
"""

import re
from collections import Counter

from sklearn.utils import murmurhash3_32

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

# Fingerprints in the HTML (script urls, globals, ids) of the vendors that
# serve each kind of wall: plain lowercase ASCII strings, found in the HTML
# with its ASCII letters lowercased. A regex would do the same, but costs the
# browser 10ms a page over 100KB of HTML for nothing a substring can't do.
VENDORS = {
    "captcha": ["recaptcha", "hcaptcha", "challenges.cloudflare.com/turnstile", "cf-turnstile", "funcaptcha",
                "arkoselabs", "geetest", "captcha-delivery"],
    "challenge": ["cf-chl", "cf_chl", "/cdn-cgi/challenge-platform", "just a moment...", "attention required! | cloudflare",
                  "perimeterx", "px-captcha", "_incapsula_", "datadome", "akamai bot"],
    "consent": ["onetrust", "optanon", "cookiebot", "didomi", "usercentrics", "quantcast", "__tcfapi", "trustarc",
                "cookielaw", "sourcepoint", "consentmanager", "cookieyes", "iubenda", "klaro", "osano", "cmp."],
    "paywall": ["piano.io", "tinypass", "poool", "laterpay", "zephr", "paywall", "subscriber-only", "premium-content",
                "metered"],
    "spinner": ["spinner", "skeleton", "shimmer", "loading...", 'aria-busy="true"'],
}
assert all(s.isascii() and s == s.lower() for found in VENDORS.values() for s in found)

# Phrases in the visible text, all languages folded into one pattern each.
PHRASES = {
    "not_found": r"not found|404|page (?:does not|doesn't) exist|no longer available|introuvable|nicht gefunden|no encontrada|non trovata",
    "no_results": r"no results|no matches|0 results|nothing found|did not match|keine ergebnisse|aucun résultat|sin resultados|nessun risultato|geen resultaten",
    "sign_in": r"sign in|log in|login|anmelden|se connecter|iniciar sesión|accedi|inloggen",
    "subscribe": r"subscribe|subscription|abonn|suscr|abbonati|already a subscriber|unlock this article",
    "cookies": r"cookie|consent|privacy settings|datenschutz|confidentialité|accept all|alle akzeptieren|tout accepter|aceptar todo",
    "javascript": r"enable javascript|javascript is (?:disabled|required)|browser is not supported|unsupported browser|update your browser|activez javascript|javascript aktivieren",
    "robot": r"are you a robot|not a robot|verify you are human|unusual traffic|access denied|checking your browser|too many requests",
    "loading": r"loading|chargement|wird geladen|cargando|caricamento",
    "adult": r"(?<![0-9])18\+|over 18|18 years|of legal age|date of birth|enter your birth|i am 18|adults? only|mayor de edad|maggiorenne|volljährig|majeur",
    "region": r"not available in your (?:country|region|location)|unavailable in your (?:country|region)|your (?:country|region|location) is not (?:supported|allowed)|geo-?(?:blocked|restricted)|restricted (?:territory|jurisdiction)|no disponible en tu (?:país|región)",
    "rate": r"too many requests|rate limit|slow down|try again (?:later|in a (?:few|minute))|request limit|not able to serve your requests",
}

# Every regex feature: its name, the text it searches and its pattern. The
# browser runs the same patterns with PCRE2 (export.py ships them in the
# model), both sides case-insensitive with Unicode case folding, so they use
# nothing whose meaning differs between Python's re and PCRE2: no \b, \w,
# \d or \s, which are ASCII-only there. The texts:
#   html     the page's HTML, as collect.py stores it
#   visible  the title, a newline, then the semantic tree
#   title    the title
#   path     the url's path
#   url      the whole url
MATCHERS = [
    ("password", "html", r"<input[^>]+type=[\"']?password"),
    ("path_login", "path", r"log-?in|sign-?in|account|auth"),
    ("path_search", "url", r"search|[?&](?:q|s|query|_nkw|st|searchTerm)="),
    *((f"{where}_{name}", subject, pattern) for name, pattern in PHRASES.items()
      for where, subject in (("says", "visible"), ("title", "title"))),
]
MATCHER_RES = [(name, subject, re.compile(pattern, re.IGNORECASE)) for name, subject, pattern in MATCHERS]

# Tags counted in the HTML: `<` and the name in any ASCII case, then a byte
# that cannot continue a name (not an ASCII letter, digit or `_`).
TAGS = ["form", "input", "iframe", "script", "a", "img", "dialog"]
NAME_BYTES = frozenset(b"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_")


def split_url(url: str) -> tuple[str, str, str]:
    """The host, path and query of an http(s) url, as urlsplit cuts them."""
    rest = url.split("://", 1)[1] if "://" in url else url
    rest, _, query = rest.split("#", 1)[0].partition("?")
    host, slash, path = rest.partition("/")
    return host, slash + path, query


def text(row: dict) -> str:
    """The page as Jev sees it, flattened for an n-gram vectorizer."""
    host, path, query = split_url(row.get("url") or "")
    path = re.sub(r"[/_.?=&-]+", " ", path + " " + query)
    return f"url {host} {path}\nstatus {row.get('status') or 0}\ntitle {row.get('title') or ''}\n{row.get('tree') or ''}"


def count_tag(html: bytes, tag: bytes) -> int:
    """How often `<tag` opens in the HTML (html already ASCII-lowercased)."""
    found, at = 0, html.find(b"<" + tag)
    while at >= 0:
        end = at + 1 + len(tag)
        found += end == len(html) or html[end] not in NAME_BYTES
        at = html.find(b"<" + tag, end)
    return found


HASH_BITS = 20
CHAR_SIZES = (3, 4, 5)
LOWER = bytes.maketrans(b"ABCDEFGHIJKLMNOPQRSTUVWXYZ", b"abcdefghijklmnopqrstuvwxyz")
# Two or more word bytes: ASCII letters (already lowercased), digits, '_',
# and any byte of a UTF-8 multi-byte sequence.
WORD = re.compile(rb"[0-9a-z_\x80-\xff]{2,}")
_buckets: dict[bytes, int] = {}


def bucket(ngram: bytes) -> int:
    """The n-gram's hash bucket (Zig: Murmur3_32.hashWithSeed(ngram, 0))."""
    found = _buckets.get(ngram)
    if found is None:
        found = _buckets[ngram] = murmurhash3_32(ngram, seed=0, positive=True) & ((1 << HASH_BITS) - 1)
    return found


def words(data: bytes) -> list[bytes]:
    """Every word, then every two adjacent words joined by a space."""
    found = WORD.findall(data)
    return found + [a + b" " + b for a, b in zip(found, found[1:])]


def chars(data: bytes) -> list[bytes]:
    """sklearn's char_wb: each piece between ASCII whitespace padded with a space
    on both sides, cut into windows of CHAR_SIZES bytes. A padded piece no
    longer than a size counts once, whole, and stops the larger sizes."""
    out = []
    for piece in data.split():
        padded = b" " + piece + b" "
        for n in CHAR_SIZES:
            if len(padded) <= n:
                out.append(padded)
                break
            out += [padded[i : i + n] for i in range(len(padded) - n + 1)]
    return out


def ngrams(row: dict) -> tuple[Counter, Counter]:
    """How often each word and each char bucket occurs in the page's text."""
    data = text(row).encode().translate(LOWER)
    return Counter(map(bucket, words(data))), Counter(map(bucket, chars(data)))


def structure(row: dict) -> dict[str, float]:
    html = row.get("html") or ""
    tree = row.get("tree") or ""
    title = row.get("title") or ""
    status = row.get("status") or 0
    lines = tree.split("\n") if tree else []
    subjects = {"html": html, "visible": f"{title}\n{tree}", "title": title,
                "path": split_url(row.get("url") or "")[1], "url": row.get("url") or ""}
    out = {
        "status_2xx": float(200 <= status < 300),
        "status_3xx": float(300 <= status < 400),
        "status_4xx": float(400 <= status < 500),
        "status_403": float(status == 403),
        "status_404": float(status in (404, 410)),
        "status_429": float(status == 429),
        "status_5xx": float(status >= 500),
        "status_none": float(not status),
        "tree_lines": min(len(lines), 2000) / 2000,
        "tree_bytes": min(len(tree.encode()), 8192) / 8192,
        "tree_tiny": float(len(lines) < 15),
        "html_bytes": min(len(html.encode()), 262144) / 262144,
        "links": min(sum("link" in line for line in lines), 300) / 300,
        "textboxes": min(sum("textbox" in line for line in lines), 20) / 20,
        "buttons": min(sum("button" in line for line in lines), 50) / 50,
    }
    lowered = html.encode().translate(LOWER)
    for tag in TAGS:
        out[f"tag_{tag}"] = min(count_tag(lowered, tag.encode()), 200) / 200
    for name, found in VENDORS.items():
        out[f"vendor_{name}"] = float(any(s.encode() in lowered for s in found))
    for name, subject, pattern in MATCHER_RES:
        out[name] = float(bool(pattern.search(subjects[subject])))
    return out


def rules(row: dict) -> dict[str, float]:
    """Hand-written guesses at each preset, as 0/1 scores."""
    f = structure(row)
    tiny = f["tree_tiny"] or f["tree_lines"] < 0.03
    blocked = f["vendor_challenge"] or f["status_429"] or (f["status_403"] and f["says_robot"])
    return {
        "is_blocked": float(bool(blocked)),
        "is_captcha": float(bool(f["vendor_captcha"] and (tiny or f["says_robot"]))),
        "is_consent_wall": float(bool(f["vendor_consent"] and f["says_cookies"])),
        "is_empty_catalog": float(bool(f["path_search"] and f["says_no_results"])),
        "is_error_page": float(bool(f["status_404"] or f["status_5xx"] or f["title_not_found"])),
        "is_login_wall": float(bool(f["password"] and (tiny or f["path_login"]))),
        "is_paywall": float(bool(f["vendor_paywall"] and f["says_subscribe"])),
        "is_unsupported_browser": float(bool(f["says_javascript"] and tiny)),
        "is_loading": float(bool(f["vendor_spinner"] and tiny and not f["status_4xx"])),
        "is_age_gate": float(bool(f["says_adult"] and (tiny or f["tag_dialog"]))),
        "is_region_blocked": float(bool(f["says_region"])),
        "is_rate_limited": float(bool(f["status_429"] or (f["says_rate"] and tiny))),
    }
