# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Write the URL list `collect.py` crawls, one `source<TAB>url` per line.

Most pages come from the Tranco top list, which is mostly ordinary content.
The presets are rare there, so each domain may also get a made-up path (an
error page) or a login path (a login wall), and hand-picked seeds cover the
rest: paywalled news, EU sites behind cookie walls, gibberish searches,
age-gated drink, tobacco and adult sites, and sites that serve only some
countries (the crawl runs from Spain). A few hosts rate-limit even a first
request that looks automated, so those seed the rate limits.

The `article` rows articles.py appended are kept.

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
    "https://www.theinformation.com/",
    "https://www.statnews.com/",
    "https://www.foreignaffairs.com/",
    "https://www.newscientist.com/",
    "https://www.mediapart.fr/",
    "https://www.lequipe.fr/",
    "https://www.nzz.ch/",
    "https://www.welt.de/",
    "https://www.bild.de/",
    "https://www.tagesspiegel.de/",
    "https://www.repubblica.it/",
    "https://www.elconfidencial.com/",
    "https://www.lavanguardia.com/",
    "https://www.expansion.com/",
    "https://www.latimes.com/",
    "https://www.bostonglobe.com/",
    "https://www.seattletimes.com/",
    "https://www.chicagotribune.com/",
    "https://www.theaustralian.com.au/",
    "https://www.smh.com.au/",
    "https://www.theglobeandmail.com/",
    "https://www.politico.eu/",
    "https://www.newsweek.com/",
    "https://www.forbes.com/",
    "https://www.fortune.com/",
    "https://www.ara.cat/",
    # More publishers, so the paywalls in the data are not a few sites'.
    "https://www.miamiherald.com/",
    "https://www.sacbee.com/",
    "https://www.kansascity.com/",
    "https://www.dallasnews.com/",
    "https://www.houstonchronicle.com/",
    "https://www.sfchronicle.com/",
    "https://www.startribune.com/",
    "https://www.denverpost.com/",
    "https://www.mercurynews.com/",
    "https://www.ajc.com/",
    "https://www.inquirer.com/",
    "https://www.post-gazette.com/",
    "https://www.tampabay.com/",
    "https://www.baltimoresun.com/",
    "https://www.newsday.com/",
    "https://www.usatoday.com/",
    "https://www.indystar.com/",
    "https://www.freep.com/",
    "https://www.jsonline.com/",
    "https://www.azcentral.com/",
    "https://www.vanityfair.com/",
    "https://harpers.org/",
    "https://www.nationalreview.com/",
    "https://www.thenation.com/",
    "https://newrepublic.com/",
    "https://www.technologyreview.com/",
    "https://www.scientificamerican.com/",
    "https://hbr.org/",
    "https://www.marketwatch.com/",
    "https://www.investors.com/",
    "https://puck.news/",
    "https://www.theverge.com/",
    "https://www.independent.co.uk/",
    "https://www.spectator.co.uk/",
    "https://www.newstatesman.com/",
    "https://www.heraldscotland.com/",
    "https://www.scotsman.com/",
    "https://www.irishtimes.com/",
    "https://www.independent.ie/",
    "https://www.afr.com/",
    "https://www.theage.com.au/",
    "https://www.heraldsun.com.au/",
    "https://www.dailytelegraph.com.au/",
    "https://www.couriermail.com.au/",
    "https://www.nzherald.co.nz/",
    "https://www.thestar.com/",
    "https://nationalpost.com/",
    "https://www.ledevoir.com/",
    "https://www.liberation.fr/",
    "https://www.lexpress.fr/",
    "https://www.lepoint.fr/",
    "https://www.leparisien.fr/",
    "https://www.lopinion.fr/",
    "https://www.ouest-france.fr/",
    "https://www.stern.de/",
    "https://www.focus.de/",
    "https://rp-online.de/",
    "https://www.abendblatt.de/",
    "https://www.ksta.de/",
    "https://www.morgenpost.de/",
    "https://www.wiwo.de/",
    "https://www.heise.de/",
    "https://www.derstandard.at/",
    "https://www.diepresse.com/",
    "https://www.tagesanzeiger.ch/",
    "https://www.lastampa.it/",
    "https://www.ilsole24ore.com/",
    "https://www.ilfattoquotidiano.it/",
    "https://www.ilmessaggero.it/",
    "https://www.elmundo.es/",
    "https://www.abc.es/",
    "https://www.elperiodico.com/",
    "https://www.lavozdegalicia.es/",
    "https://www.elcorreo.com/",
    "https://www.diariovasco.com/",
    "https://www.volkskrant.nl/",
    "https://www.trouw.nl/",
    "https://www.parool.nl/",
    "https://www.ad.nl/",
    "https://fd.nl/",
    "https://www.standaard.be/",
    "https://www.demorgen.be/",
    "https://www.dn.se/",
    "https://www.svd.se/",
    "https://www.aftenposten.no/",
    "https://politiken.dk/",
    "https://www.berlingske.dk/",
    "https://www.hs.fi/",
    "https://wyborcza.pl/",
    "https://www.publico.pt/",
    "https://expresso.pt/",
    "https://www.asahi.com/",
    "https://mainichi.jp/",
    "https://www.yomiuri.co.jp/",
    "https://www.japantimes.co.jp/",
    "https://www.straitstimes.com/",
    "https://www.thehindu.com/",
    "https://economictimes.indiatimes.com/",
    "https://www.livemint.com/",
    "https://www.business-standard.com/",
    "https://www.clarin.com/",
    "https://www.lanacion.com.ar/",
    "https://www.eltiempo.com/",
    "https://www1.folha.uol.com.br/",
    "https://www.estadao.com.br/",
    "https://oglobo.globo.com/",
    "https://www.businesslive.co.za/",
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

# Drink, tobacco, game and adult sites that ask the visitor's age first.
AGE_GATES = [
    "https://www.jackdaniels.com/",
    "https://www.johnniewalker.com/",
    "https://www.jamesonwhiskey.com/",
    "https://www.heineken.com/",
    "https://www.guinness.com/",
    "https://www.budweiser.com/",
    "https://www.coronausa.com/",
    "https://www.bacardi.com/",
    "https://www.absolut.com/",
    "https://www.hennessy.com/",
    "https://www.greygoose.com/",
    "https://www.glenfiddich.com/",
    "https://www.makersmark.com/",
    "https://www.buffalotracedistillery.com/",
    "https://www.juul.com/",
    "https://www.vuse.com/",
    "https://www.marlboro.com/",
    "https://store.steampowered.com/app/271590/",
    "https://store.steampowered.com/app/1091500/",
    "https://www.smirnoff.com/",
    "https://www.captainmorgan.com/",
    "https://www.baileys.com/",
    "https://www.ketelone.com/",
    "https://www.bulleit.com/",
    "https://www.donjulio.com/",
    "https://www.crownroyal.com/",
    "https://www.tanqueray.com/",
    "https://www.chivas.com/",
    "https://www.beefeatergin.com/",
    "https://www.malibudrinks.com/",
    "https://www.havana-club.com/",
    "https://www.martell.com/",
    "https://www.theglenlivet.com/",
    "https://www.jimbeam.com/",
    "https://www.laphroaig.com/",
    "https://www.courvoisier.com/",
    "https://www.woodfordreserve.com/",
    "https://www.oldforester.com/",
    "https://www.herradura.com/",
    "https://www.stellaartois.com/",
    "https://www.michelobultra.com/",
    "https://www.amstel.com/",
    "https://www.desperados.com/",
    "https://www.carlsberg.com/",
    "https://www.coorslight.com/",
    "https://www.millerlite.com/",
    "https://www.bluemoonbrewingco.com/",
    "https://www.modelousa.com/",
    "https://www.pacificobeer.com/",
    "https://www.patron.com/",
    "https://www.bombaysapphire.com/",
    "https://www.dewars.com/",
    "https://www.martini.com/",
    "https://www.josecuervo.com/",
    "https://www.aperol.com/",
    "https://www.yellowtailwine.com/",
    "https://www.pornhub.com/",
    "https://xhamster.com/",
    "https://www.redtube.com/",
    "https://spankbang.com/",
]

# Streaming, betting, news and Japanese sites that refuse visitors from
# outside their country, as seen from Spain.
REGION_BLOCKS = [
    "https://www.hulu.com/",
    "https://www.sling.com/",
    "https://www.fubo.tv/",
    "https://www.philo.com/",
    "https://www.cbs.com/live-tv/",
    "https://www.itv.com/watch",
    "https://www.channel4.com/",
    "https://tver.jp/",
    "https://abema.tv/",
    "https://radiko.jp/",
    "https://www.fanduel.com/",
    "https://sportsbook.draftkings.com/",
    "https://sports.betmgm.com/",
    "https://www.caesars.com/sportsbook-and-casino",
    "https://www.chicagotribune.com/",
    "https://www.nydailynews.com/",
    "https://www.baltimoresun.com/",
    "https://www.orlandosentinel.com/",
    "https://www.sun-sentinel.com/",
    "https://www.courant.com/",
    "https://www.stltoday.com/",
    "https://buffalonews.com/",
    "https://richmond.com/",
    # The Tribune's sister papers, which share its "not available in your
    # region" page.
    "https://www.mcall.com/",
    "https://www.dailypress.com/",
    "https://www.pilotonline.com/",
    "https://www.capitalgazette.com/",
    "https://www.carrollcountytimes.com/",
    "https://www.virginiagazette.com/",
    "https://www.denverpost.com/",
    "https://www.mercurynews.com/",
    "https://www.eastbaytimes.com/",
    "https://www.twincities.com/",
    "https://www.ocregister.com/",
    "https://www.dailybreeze.com/",
    "https://www.presstelegram.com/",
    "https://www.pe.com/",
    "https://www.sbsun.com/",
    "https://www.dailybulletin.com/",
    "https://www.redlandsdailyfacts.com/",
    "https://www.whittierdailynews.com/",
    "https://www.sgvtribune.com/",
    "https://www.pasadenastarnews.com/",
    "https://www.dailynews.com/",
    "https://www.bostonherald.com/",
    "https://www.dailycamera.com/",
    "https://www.timescall.com/",
    "https://www.reporterherald.com/",
    "https://www.lowellsun.com/",
    "https://www.sentinelandenterprise.com/",
    "https://www.saratogian.com/",
    "https://www.troyrecord.com/",
    "https://www.dailyfreeman.com/",
    "https://www.oneidadispatch.com/",
    "https://www.timesherald.com/",
    "https://www.pottsmerc.com/",
    "https://www.dailylocal.com/",
    "https://www.delcotimes.com/",
    "https://www.thereporteronline.com/",
    "https://www.morningjournal.com/",
    "https://www.news-herald.com/",
    "https://www.macombdaily.com/",
    "https://www.dailytribune.com/",
    "https://www.theoaklandpress.com/",
    "https://www.record-bee.com/",
    "https://www.times-standard.com/",
    "https://www.ukiahdailyjournal.com/",
    "https://www.marinij.com/",
    "https://www.montereyherald.com/",
    "https://www.santacruzsentinel.com/",
    "https://www.chicoer.com/",
    "https://www.orovillemr.com/",
    "https://www.redbluffdailynews.com/",
    "https://www.willitsnews.com/",
    "https://www.thecalifornian.com/",
]

# Hosts that answered the first crawl's first request with a 429, and others
# known to rate-limit visitors that look automated: their travel-group
# sisters, Google's search pages and Reddit.
RATE_LIMITS = [
    "https://www.avito.ru/moskva",
    "https://www.hyatt.com/explore-hotels",
    "https://www.expedia.com/Hotels",
    "https://www.expedia.co.uk/",
    "https://www.expedia.de/",
    "https://www.expedia.es/",
    "https://www.vrbo.com/",
    "https://www.orbitz.com/",
    "https://www.travelocity.com/",
    "https://www.wotif.com/",
    "https://www.ebookers.com/",
    "https://www.cheaptickets.com/",
    "https://venturebeat.com/ai/",
    "https://www.adjust.com/blog/",
    "https://www.glenfiddich.com/",
    "https://ipapi.co/json/",
    "https://www.google.com/search?q=headless+browser",
    "https://scholar.google.com/scholar?q=headless+browser",
    "https://www.google.com/maps/search/cafe",
    "https://old.reddit.com/",
    "https://www.reddit.com/r/programming/",
    "https://www.reddit.com/r/AskReddit/top/",
    "https://api.github.com/search/repositories?q=browser",
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "https://www.linkedin.com/jobs/search?keywords=engineer",
    "https://www.instagram.com/explore/",
    "https://x.com/explore",
    "https://www.tiktok.com/explore",
    "https://www.zillow.com/homes/",
    "https://www.realtor.com/realestateandhomes-search/Austin_TX",
    "https://www.ticketmaster.com/",
    "https://www.stubhub.com/",
    "https://www.footlocker.com/",
    "https://www.nike.com/w/new-shoes",
    "https://www.ryanair.com/",
    "https://www.southwest.com/",
    "https://www.kayak.com/",
    "https://www.skyscanner.net/",
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
    rows += [("age", url) for url in AGE_GATES]
    rows += [("region", url) for url in REGION_BLOCKS]
    rows += [("rate", url) for url in RATE_LIMITS]

    rng.shuffle(rows)  # mixes slow and fast hosts across the crawl
    if args.out.exists():
        rows += [tuple(line.split("\t")) for line in args.out.read_text().splitlines() if line.startswith("article\t")]
    args.out.parent.mkdir(exist_ok=True)
    args.out.write_text("".join(f"{source}\t{url}\n" for source, url in rows))
    print(f"{len(rows)} urls -> {args.out}")


if __name__ == "__main__":
    main()
