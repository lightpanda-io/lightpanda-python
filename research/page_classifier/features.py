"""What the local models see of a page, shared by train.py and evaluate.py.

Three views of one row from collect.py:

- `text`: the same input Jev gets (url, title, status, semantic tree), for
  the bag-of-n-grams model.
- `structure`: a few dozen numbers and flags from the tree and the HTML,
  such as the status, the password field and known captcha or consent
  vendors' scripts.
- `rules`: hand-written preset guesses from those flags, the baseline any
  model has to beat.
"""

import re
from urllib.parse import urlsplit

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
# serve each kind of wall. Matched case-insensitively.
VENDORS = {
    "captcha": r"recaptcha|hcaptcha|challenges\.cloudflare\.com/turnstile|cf-turnstile|funcaptcha|arkoselabs|geetest|captcha-delivery",
    "challenge": r"cf-chl|cf_chl|/cdn-cgi/challenge-platform|just a moment\.\.\.|attention required! \| cloudflare|perimeterx|px-captcha|_incapsula_|datadome|akamai bot",
    "consent": r"onetrust|optanon|cookiebot|didomi|usercentrics|quantcast|__tcfapi|trustarc|cookielaw|sourcepoint|consentmanager|cookieyes|iubenda|klaro|osano|cmp\.",
    "paywall": r"piano\.io|tinypass|poool|laterpay|zephr|paywall|subscriber-only|premium-content|metered",
    "spinner": r"spinner|skeleton|shimmer|loading\.\.\.|aria-busy=\"true\"",
}
VENDOR_RES = {name: re.compile(pattern, re.IGNORECASE) for name, pattern in VENDORS.items()}

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
    "adult": r"\b18\+|over 18|18 years|of legal age|date of birth|enter your birth|i am 18|adults? only|mayor de edad|maggiorenne|volljährig|majeur",
    "region": r"not available in your (?:country|region|location)|unavailable in your (?:country|region)|your (?:country|region|location) is not (?:supported|allowed)|geo-?(?:blocked|restricted)|restricted (?:territory|jurisdiction)|no disponible en tu (?:país|región)",
    "rate": r"too many requests|rate limit|slow down|try again (?:later|in a (?:few|minute))|request limit|not able to serve your requests",
}
PHRASE_RES = {name: re.compile(pattern, re.IGNORECASE) for name, pattern in PHRASES.items()}

PASSWORD = re.compile(r"<input[^>]+type=[\"']?password", re.IGNORECASE)
TAGS = {tag: re.compile(rf"<{tag}\b", re.IGNORECASE) for tag in ("form", "input", "iframe", "script", "a", "img", "dialog")}


def text(row: dict) -> str:
    """The page as Jev sees it, flattened for an n-gram vectorizer."""
    parts = urlsplit(row.get("url") or "")
    path = re.sub(r"[/_.?=&-]+", " ", parts.path + " " + parts.query)
    return f"url {parts.netloc} {path}\nstatus {row.get('status')}\ntitle {row.get('title') or ''}\n{row.get('tree') or ''}"


def structure(row: dict) -> dict[str, float]:
    html = row.get("html") or ""
    tree = row.get("tree") or ""
    title = row.get("title") or ""
    status = row.get("status") or 0
    lines = tree.splitlines()
    visible = f"{title}\n{tree}"
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
        "tree_bytes": min(len(tree), 8192) / 8192,
        "tree_tiny": float(len(lines) < 15),
        "html_bytes": min(len(html), 262144) / 262144,
        "links": min(sum("link" in line for line in lines), 300) / 300,
        "textboxes": min(sum("textbox" in line for line in lines), 20) / 20,
        "buttons": min(sum("button" in line for line in lines), 50) / 50,
        "password": float(bool(PASSWORD.search(html))),
        "path_login": float(bool(re.search(r"log-?in|sign-?in|account|auth", urlsplit(row.get("url") or "").path, re.IGNORECASE))),
        "path_search": float(bool(re.search(r"search|[?&](q|s|query|_nkw|st|searchTerm)=", row.get("url") or "", re.IGNORECASE))),
    }
    for tag, pattern in TAGS.items():
        out[f"tag_{tag}"] = min(len(pattern.findall(html)), 200) / 200
    for name, pattern in VENDOR_RES.items():
        out[f"vendor_{name}"] = float(bool(pattern.search(html)))
    for name, pattern in PHRASE_RES.items():
        out[f"says_{name}"] = float(bool(pattern.search(visible)))
        out[f"title_{name}"] = float(bool(pattern.search(title)))
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
