# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Write the URL list `collect.py` crawls, one `source<TAB>url` per line.

Most pages come from the Tranco top list, which is mostly ordinary content.
The presets are rare there, so each domain may also get a made-up path (an
error page) or a login path (a login wall), and hand-picked seeds cover the
rest: paywalled news, EU sites behind cookie walls, gibberish searches.

Run:  uv run research/page_classifier/seeds.py --domains 3000
"""

import argparse
import csv
import io
import random
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE / "data"
TRANCO = "https://tranco-list.eu/top-1m.csv.zip"

# News and magazines that meter or hard-wall their articles.
PAYWALLS = [
    "https://www.wsj.com/",
    "https://www.ft.com/",
    "https://www.nytimes.com/",
    "https://www.economist.com/",
    "https://www.bloomberg.com/",
    "https://www.washingtonpost.com/",
    "https://www.theathletic.com/",
    "https://www.newyorker.com/",
    "https://www.theatlantic.com/",
    "https://www.wired.com/",
    "https://www.lemonde.fr/",
    "https://www.lefigaro.fr/",
    "https://www.zeit.de/index",
    "https://www.faz.net/",
    "https://www.sueddeutsche.de/",
    "https://www.elpais.com/",
    "https://www.corriere.it/",
    "https://www.nrc.nl/",
    "https://www.thetimes.co.uk/",
    "https://www.telegraph.co.uk/",
    "https://www.businessinsider.com/",
    "https://www.barrons.com/",
    "https://www.medium.com/",
    "https://www.substack.com/",
    "https://www.haaretz.com/",
    "https://www.nikkei.com/",
    "https://www.scmp.com/",
    "https://www.lesechos.fr/",
    "https://www.handelsblatt.com/",
    "https://www.spiegel.de/plus/",
]

# Search pages that print "no results" for a query nobody types.
NONSENSE = "xqzzzvqkplorbt"
EMPTY_SEARCHES = [
    "https://www.npmjs.com/search?q={q}",
    "https://github.com/search?q={q}&type=repositories",
    "https://en.wikipedia.org/w/index.php?search={q}&fulltext=1",
    "https://www.ebay.com/sch/i.html?_nkw={q}",
    "https://pypi.org/search/?q={q}",
    "https://crates.io/search?q={q}",
    "https://stackoverflow.com/search?q={q}",
    "https://www.etsy.com/search?q={q}",
    "https://www.bestbuy.com/site/searchpage.jsp?st={q}",
    "https://www.target.com/s?searchTerm={q}",
    "https://www.walmart.com/search?q={q}",
    "https://www.ikea.com/us/en/search/?q={q}",
    "https://www.zalando.de/katalog/?q={q}",
    "https://www.otto.de/suche/{q}/",
    "https://www.fnac.com/SearchResult/ResultList.aspx?Search={q}",
    "https://www.elcorteingles.es/search-nwx/1/?s={q}",
    "https://www.bol.com/nl/nl/s/?searchtext={q}",
    "https://www.allegro.pl/listing?string={q}",
    "https://www.rakuten.com/search/{q}",
    "https://www.newegg.com/p/pl?d={q}",
    "https://www.goodreads.com/search?q={q}",
    "https://www.imdb.com/find/?q={q}",
    "https://www.discogs.com/search/?q={q}",
    "https://www.thingiverse.com/search?q={q}",
    "https://archive.org/search?query={q}",
    "https://news.ycombinator.com/item?id=0",
    "https://lobste.rs/search?q={q}",
    "https://www.reddit.com/search/?q={q}",
    "https://hub.docker.com/search?q={q}",
    "https://www.youtube.com/results?search_query={q}",
]

LOGIN_PATHS = ["/login", "/signin", "/account", "/user/login", "/my-account"]


def tranco(limit: int) -> list[str]:
    """The top `limit` domains, downloaded once and cached under data/."""
    cached = DATA / "tranco.csv"
    if not cached.exists():
        DATA.mkdir(exist_ok=True)
        with urllib.request.urlopen(TRANCO) as resp:
            with zipfile.ZipFile(io.BytesIO(resp.read())) as zf:
                cached.write_bytes(zf.read(zf.namelist()[0]))
    with cached.open() as f:
        return [row[1] for _, row in zip(range(limit), csv.reader(f))]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--domains", type=int, default=3000, help="Tranco domains to take")
    parser.add_argument("--missing", type=float, default=0.15, help="share of domains also given a made-up path")
    parser.add_argument("--login", type=float, default=0.10, help="share of domains also given a login path")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=Path, default=DATA / "urls.tsv")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = []
    for domain in tranco(args.domains):
        rows.append(("tranco", f"https://{domain}/"))
        if rng.random() < args.missing:
            slug = "".join(rng.choices("abcdefghijklmnopqrstuvwxyz", k=12))
            rows.append(("missing", f"https://{domain}/{slug}"))
        if rng.random() < args.login:
            rows.append(("login", f"https://{domain}{rng.choice(LOGIN_PATHS)}"))
    rows += [("paywall", url) for url in PAYWALLS]
    rows += [("empty", url.format(q=NONSENSE)) for url in EMPTY_SEARCHES]

    rng.shuffle(rows)  # mixes slow and fast hosts across the crawl
    args.out.parent.mkdir(exist_ok=True)
    args.out.write_text("".join(f"{source}\t{url}\n" for source, url in rows))
    print(f"{len(rows)} urls -> {args.out}")


if __name__ == "__main__":
    main()
