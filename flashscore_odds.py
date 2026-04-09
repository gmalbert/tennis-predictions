"""
flashscore_odds.py
------------------
Fetch pre-match decimal odds for upcoming ATP/WTA singles matches from Flashscore.
Uses two HTTP endpoints — no Playwright / no headless browser required.

Endpoints used
--------------
1. Live-feed:
     GET https://www.flashscore.com/x/feed/f_2_0_2_en-gb_1
     Header: x-fsign: SW9D1eZo
     Returns all live + scheduled matches in a proprietary delimited format.

2. Odds GraphQL (persisted query):
     GET https://global.ds.lsapp.eu/odds/pq_graphql
     Params: _hash=oce, eventId=<AA>, projectId=2, geoIpCode, geoIpSubdivisionCode
     Returns JSON with bookmaker odds for all bet types.

Match-winner odds are identified by:
     bettingType == "HOME_AWAY" AND bettingScope == "FULL_TIME"

Key identifiers
---------------
- Match ID (eventId):  the ``AA`` field in the live-feed section.
- Player 1 (p1):       live-feed ``AE`` field  (HOME side, odds[0]).
- Player 2 (p2):       live-feed ``AF`` field  (AWAY side, odds[1]).
- Tournament:          live-feed ``ZA`` field.
- Surface:             last comma-segment of ``ZA``.

Public API
----------
    from flashscore_odds import fetch_flashscore_odds

    rows = fetch_flashscore_odds()
    # returns list[dict] — one row per match × bookmaker

    # Each dict:
    # {
    #   "event_id":       str,   # AA from live feed
    #   "p1":             str,   # player 1 name  (HOME)
    #   "p2":             str,   # player 2 name  (AWAY)
    #   "wu":             str,   # player 1 Flashscore URL slug
    #   "wv":             str,   # player 2 Flashscore URL slug
    #   "tourney":        str,   # full tournament header from ZA
    #   "surface":        str,   # e.g. "clay", "hard", "grass"
    #   "ts_start":       int,   # Unix UTC timestamp of match start
    #   "bookmaker_id":   int,
    #   "bookmaker_name": str,
    #   "p1_odds":        float, # current decimal odds for player 1
    #   "p2_odds":        float, # current decimal odds for player 2
    #   "p1_opening":     float, # opening odds for player 1
    #   "p2_opening":     float, # opening odds for player 2
    # }

Cross-repo reuse
----------------
Copy this file into any project.  The only runtime dependency is ``requests``
(already a standard requirement in most data-science stacks).

To change the geoIp region (which bookmakers are shown), edit the module-level
``GEO_IP_CODE`` / ``GEO_IP_SUB`` constants, or pass them via fetch_match_odds().
Common values:
  US+USNH  →  bet365.us, BetMGM.us, FanDuel
  DE       →  bwin, bet365.de, Unibet (omit geoIpSubdivisionCode)
  GB       →  Bet365, William Hill, Betfair
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import requests

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

FS_LIVE_URL = "https://www.flashscore.com/x/feed/f_2_0_2_en-gb_1"
FS_ODDS_URL = "https://global.ds.lsapp.eu/odds/pq_graphql"

FS_HEADERS = {
    "x-fsign":         "SW9D1eZo",
    "User-Agent":      ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36"),
    "Referer":         "https://www.flashscore.com/",
    "Accept":          "*/*",
    "Accept-Language": "en-GB,en;q=0.9",
    "Origin":          "https://www.flashscore.com",
}

# GeoIP parameters — controls which bookmakers are returned.
# These were captured from a US-based browser session; adjust as needed.
GEO_IP_CODE = "US"
GEO_IP_SUB  = "USNH"

_SEP_FIELD = chr(0xAC)   # ¬  — section field separator
_SEP_KV    = chr(0xF7)   # ÷  — key/value separator


# ── Internal feed parser ──────────────────────────────────────────────────────

def _parse_fields(section: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in section.split(_SEP_FIELD):
        if _SEP_KV in part:
            k, _, v = part.partition(_SEP_KV)
            out[k] = v
    return out


# ── Public functions ──────────────────────────────────────────────────────────

def fetch_live_singles_matches(
    *,
    feed_url: str = FS_LIVE_URL,
    include_live: bool = True,
    session: Optional[requests.Session] = None,
) -> list[dict]:
    """
    Parse the Flashscore live feed and return all ATP/WTA singles match records.

    Parameters
    ----------
    feed_url:     Override the live-feed URL (useful for testing with a local file).
    include_live: If False, skip matches that are already in progress
                  (identified by having non-zero game scores in the feed).
    session:      Optional requests.Session to reuse.

    Returns
    -------
    list[dict] with keys: aa, p1, p2, wu, wv, px, py, ts_start, ab, tourney, surface
    """
    s = session or requests.Session()
    resp = s.get(feed_url, headers=FS_HEADERS, timeout=15)
    resp.raise_for_status()
    raw = resp.text

    matches: list[dict] = []
    in_singles   = False
    cur_tourney  = ""
    cur_surface  = ""

    for section in raw.split("~"):
        fields = _parse_fields(section)

        # ZA marks the start of a tournament block.
        if "ZA" in fields:
            za        = fields["ZA"]
            za_lo     = za.lower()
            in_singles = "singles" in za_lo and ("atp" in za_lo or "wta" in za_lo)
            cur_tourney = za
            cur_surface = za.rsplit(",", 1)[-1].strip() if "," in za else ""
            continue

        if not (in_singles and "AA" in fields and "WU" in fields):
            continue

        # Skip finished matches
        if fields.get("AB") == "3":
            continue

        # Optionally skip in-progress games (non-zero game scores)
        if not include_live:
            if fields.get("GRA", "0") not in ("0", "") or \
               fields.get("GRB", "0") not in ("0", ""):
                continue

        matches.append({
            "aa":       fields["AA"],
            "p1":       fields.get("AE", ""),
            "p2":       fields.get("AF", ""),
            "wu":       fields.get("WU", ""),
            "wv":       fields.get("WV", ""),
            "px":       fields.get("PX", ""),
            "py":       fields.get("PY", ""),
            "ts_start": int(fields.get("AD", 0) or 0),
            "ab":       fields.get("AB", "?"),
            "tourney":  cur_tourney,
            "surface":  cur_surface,
        })

    log.debug("fetch_live_singles_matches: found %d matches", len(matches))
    return matches


def fetch_match_odds(
    event_id: str,
    *,
    geo_code: str = GEO_IP_CODE,
    geo_sub:  str = GEO_IP_SUB,
    session:  Optional[requests.Session] = None,
) -> list[dict]:
    """
    Fetch match-winner (HOME_AWAY / FULL_TIME) decimal odds for one match.

    Parameters
    ----------
    event_id: The ``AA`` field from the live feed for this match.
    geo_code: ISO-3166-1 alpha-2 country code for bookmaker geo-filtering.
    geo_sub:  ISO-3166-2 subdivision code (optional; needed for some regions).
    session:  Optional requests.Session.

    Returns
    -------
    list[dict] — one entry per bookmaker — with keys:
        bookmaker_id, bookmaker_name, p1_odds, p2_odds, p1_opening, p2_opening
    Returns an empty list if no odds are available (match is live / finished /
    not yet indexed by the odds service).
    """
    s = session or requests.Session()

    params: dict[str, str] = {
        "_hash":                "oce",
        "eventId":              event_id,
        "projectId":            "2",
        "geoIpCode":            geo_code,
        "geoIpSubdivisionCode": geo_sub,
    }
    if not geo_sub:
        del params["geoIpSubdivisionCode"]

    try:
        resp = s.get(FS_ODDS_URL, params=params, headers=FS_HEADERS, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError) as exc:
        log.warning("Odds request failed for %s: %s", event_id, exc)
        return []

    event_odds = data.get("data", {}).get("findOddsByEventId") or {}
    if not event_odds:
        return []

    # Build bookmaker id → name map
    bk_map: dict[str, str] = {}
    for bk_entry in event_odds.get("settings", {}).get("bookmakers", []):
        inner = bk_entry.get("bookmaker", {})
        bk_id = str(inner.get("id", ""))
        if bk_id:
            bk_map[bk_id] = inner.get("name", f"bk{bk_id}")

    results: list[dict] = []
    for item in event_odds.get("odds", []):
        if item.get("bettingType") != "HOME_AWAY":
            continue
        if item.get("bettingScope") != "FULL_TIME":
            continue
        odds_items = item.get("odds", [])
        if len(odds_items) < 2:
            continue

        bk_id = str(item.get("bookmakerId", ""))
        try:
            # odds[0] = HOME  = p1 (AE in live feed)
            # odds[1] = AWAY  = p2 (AF in live feed)
            p1_odds    = float(odds_items[0]["value"])
            p2_odds    = float(odds_items[1]["value"])
            p1_opening = float(odds_items[0].get("opening") or p1_odds)
            p2_opening = float(odds_items[1].get("opening") or p2_odds)
        except (TypeError, ValueError, KeyError):
            continue

        results.append({
            "bookmaker_id":   int(bk_id) if bk_id.isdigit() else 0,
            "bookmaker_name": bk_map.get(bk_id, f"bk{bk_id}"),
            "p1_odds":        p1_odds,
            "p2_odds":        p2_odds,
            "p1_opening":     p1_opening,
            "p2_opening":     p2_opening,
        })

    return results


def fetch_flashscore_odds(
    *,
    tourney_filter:  Optional[str] = None,
    max_matches:     Optional[int] = None,
    include_live:    bool = True,
    geo_code:        str = GEO_IP_CODE,
    geo_sub:         str = GEO_IP_SUB,
    request_delay_s: float = 0.3,
) -> list[dict]:
    """
    Fetch Flashscore pre-match odds for all upcoming ATP/WTA singles.

    Wraps fetch_live_singles_matches() + fetch_match_odds() in a single call.

    Parameters
    ----------
    tourney_filter:  Optional substring filter on the tournament name
                     (case-insensitive). E.g. ``"monte carlo"`` or ``"atp"``.
    max_matches:     Cap on the number of matches to query odds for.
    include_live:    If False, skip matches already in progress.
    geo_code:        ISO country code for bookmakers (e.g. ``"GB"``, ``"DE"``, ``"US"``).
    geo_sub:         ISO subdivision code (e.g. ``"USNH"`` for New Hampshire, US).
    request_delay_s: Pause between individual odds API calls to avoid throttling.

    Returns
    -------
    list[dict] — one row per (match × bookmaker) — with keys:
        event_id, p1, p2, wu, wv, tourney, surface, ts_start,
        bookmaker_id, bookmaker_name, p1_odds, p2_odds, p1_opening, p2_opening
    """
    session = requests.Session()

    matches = fetch_live_singles_matches(session=session, include_live=include_live)
    log.info("fetch_flashscore_odds: %d singles matches in live feed", len(matches))

    if tourney_filter:
        tf      = tourney_filter.lower()
        matches = [m for m in matches if tf in m["tourney"].lower()]
        log.info("  after tourney_filter=%r: %d matches", tourney_filter, len(matches))

    if max_matches is not None:
        matches = matches[:max_matches]

    rows: list[dict] = []
    for i, match in enumerate(matches):
        if i and request_delay_s > 0:
            time.sleep(request_delay_s)

        odds_list = fetch_match_odds(
            match["aa"], geo_code=geo_code, geo_sub=geo_sub, session=session
        )
        if not odds_list:
            log.debug("  no odds: %s vs %s", match["p1"], match["p2"])
            continue

        for odds in odds_list:
            rows.append({
                "event_id":       match["aa"],
                "p1":             match["p1"],
                "p2":             match["p2"],
                "wu":             match["wu"],
                "wv":             match["wv"],
                "tourney":        match["tourney"],
                "surface":        match["surface"],
                "ts_start":       match["ts_start"],
                "bookmaker_id":   odds["bookmaker_id"],
                "bookmaker_name": odds["bookmaker_name"],
                "p1_odds":        odds["p1_odds"],
                "p2_odds":        odds["p2_odds"],
                "p1_opening":     odds["p1_opening"],
                "p2_opening":     odds["p2_opening"],
            })

    log.info("fetch_flashscore_odds: returning %d rows", len(rows))
    return rows


# ── CLI smoke-test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import datetime
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
        stream=sys.stdout,
    )

    print("Fetching Flashscore odds for upcoming ATP/WTA singles …")
    rows = fetch_flashscore_odds(max_matches=20, request_delay_s=0.2)

    print(f"\nTotal rows returned: {len(rows)}")

    if not rows:
        print("No rows returned — check the log output above.")
        sys.exit(1)

    # Group by match for clean display
    seen: dict[str, bool] = {}
    for row in rows:
        key = row["event_id"]
        if key not in seen:
            seen[key] = True
            ts = datetime.datetime.fromtimestamp(
                row["ts_start"], tz=datetime.timezone.utc
            ).strftime("%Y-%m-%d %H:%M UTC")
            print(f"\n  [{ts}]  {row['p1']}  vs  {row['p2']}")
            print(f"  {row['tourney']}")
        print(
            f"    {row['bookmaker_name']:28s}: "
            f"{row['p1']:20s} {row['p1_odds']:.2f}  |  "
            f"{row['p2']:20s} {row['p2_odds']:.2f}"
        )
