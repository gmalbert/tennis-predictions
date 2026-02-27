"""
scraper_atp.py
--------------
Scrape completed ATP Tour match results for a given year from the official
ATP Tour website (atptour.com).

The ATP site is a React application – all data is loaded via XHR after the
initial HTML shell.  We use Playwright (async) to:
  1. Navigate to the results-archive page and intercept the JSON payloads.
  2. For each tournament, follow through to the individual match scores.
  3. Cache every raw JSON response to disk so we never hit the site twice.

Output: data_files/atp_results_<year>.csv  (Sackmann-schema columns)

Usage:
    python scraper_atp.py              # scrapes 2025 by default
    python scraper_atp.py --year 2024  # or specify year
    python scraper_atp.py --clear-cache --year 2025
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
BASE_DIR   = Path(__file__).parent
CACHE_DIR  = BASE_DIR / "cache" / "atp"
OUT_DIR    = BASE_DIR / "data_files"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

ATP_BASE = "https://www.atptour.com"

# archive page – loads tournament list via XHR
ARCHIVE_URL = ATP_BASE + "/en/scores/results-archive"

# ATP internal JSON endpoints (intercepted from browser devtools)
TOURNAMENTS_API = ATP_BASE + "/en/-/tournaments/archived-results/{year}"
DRAW_API        = ATP_BASE + "/en/-/ajax/livescores/draws/{year}/{tourney_id}"
RESULTS_API     = ATP_BASE + "/en/scores/ajax/results-archive"

# Sackmann column order
SACKMANN_COLS = [
    "tourney_id", "tourney_name", "surface", "draw_size", "tourney_level",
    "tourney_date", "match_num",
    "winner_id", "winner_seed", "winner_entry",
    "winner_name", "winner_hand", "winner_ht", "winner_ioc", "winner_age",
    "loser_id",  "loser_seed",  "loser_entry",
    "loser_name",  "loser_hand",  "loser_ht",  "loser_ioc",  "loser_age",
    "score", "best_of", "round", "minutes",
    "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon",
    "w_SvGms", "w_bpSaved", "w_bpFaced",
    "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon",
    "l_SvGms", "l_bpSaved", "l_bpFaced",
    "winner_rank", "winner_rank_points",
    "loser_rank",  "loser_rank_points",
]

# ATP tourney_level codes (maps ATP category string → Sackmann single char)
LEVEL_MAP = {
    "Grand Slam": "G",
    "ATP Finals": "F",
    "ATP Masters 1000": "M",
    "ATP 500": "A",
    "ATP 250": "A",
    "Davis Cup": "D",
    "Olympics": "O",
}

SURFACE_MAP = {
    "Hard":  "Hard",
    "Clay":  "Clay",
    "Grass": "Grass",
    "Carpet": "Carpet",
}


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_path(key: str) -> Path:
    h = hashlib.md5(key.encode()).hexdigest()
    return CACHE_DIR / f"{h}.json"


def load_cache(key: str) -> Any | None:
    p = _cache_path(key)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            p.unlink(missing_ok=True)
    return None


def save_cache(key: str, data: Any) -> None:
    _cache_path(key).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Playwright helpers
# ---------------------------------------------------------------------------

async def _fetch_json_playwright(url: str, cache_key: str, wait_ms: int = 2000) -> dict | list | None:
    """
    Navigate to `url` with Playwright, wait for network idle, return the
    JSON body.  Results are cached; a cached hit skips the browser entirely.
    """
    cached = load_cache(cache_key)
    if cached is not None:
        return cached

    try:
        from playwright.async_api import async_playwright, TimeoutError as PWTimeout
    except ImportError:
        raise SystemExit(
            "Playwright not installed.  Run:  pip install playwright && playwright install chromium"
        )

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 900},
        )

        captured: list[dict] = []

        async def _handle_response(response):
            if url.split("?")[0] in response.url and "json" in response.headers.get("content-type", ""):
                try:
                    captured.append(await response.json())
                except Exception:
                    pass

        page = await context.new_page()
        page.on("response", _handle_response)

        try:
            await page.goto(url, wait_until="networkidle", timeout=30_000)
            await page.wait_for_timeout(wait_ms)
        except PWTimeout:
            print(f"  [ATP] Timeout loading {url} – using partial data")
        finally:
            await browser.close()

        data = captured[0] if captured else None
        if data:
            save_cache(cache_key, data)
        return data


async def _intercepted_fetch(base_url: str, cache_key: str) -> dict | list | None:
    """
    Fetch a JSON endpoint directly with Playwright (bypasses Cloudflare
    more reliably than raw requests for the ATP site).
    """
    cached = load_cache(cache_key)
    if cached is not None:
        return cached

    try:
        from playwright.async_api import async_playwright
    except ImportError:
        raise SystemExit("pip install playwright && playwright install chromium")

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        )
        page = await context.new_page()

        result = None
        try:
            resp = await page.goto(base_url, wait_until="domcontentloaded", timeout=20_000)
            if resp and resp.ok:
                try:
                    result = await resp.json()
                except Exception:
                    text = await resp.text()
                    try:
                        result = json.loads(text)
                    except Exception:
                        pass
        finally:
            await browser.close()

        if result is not None:
            save_cache(cache_key, result)
        return result


# ---------------------------------------------------------------------------
# ATP data fetching
# ---------------------------------------------------------------------------

async def fetch_tournament_list(year: int) -> list[dict]:
    """Return the list of ATP tournaments for `year`."""
    url = TOURNAMENTS_API.format(year=year)
    key = f"atp_tournaments_{year}"
    data = await _intercepted_fetch(url, key)

    if data is None:
        # Fallback: scrape the archive page and parse the embedded JSON
        archive = await _fetch_json_playwright(
            f"{ARCHIVE_URL}?year={year}", f"atp_archive_page_{year}"
        )
        data = archive

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # Try common wrapper keys
        for k in ("data", "tournaments", "items", "results"):
            if k in data and isinstance(data[k], list):
                return data[k]
    return []


async def fetch_tournament_results(year: int, atp_tourney_id: str, slug: str) -> list[dict]:
    """Return match records for a single tournament."""
    url = f"{ATP_BASE}/en/scores/results/{slug}/{year}/{atp_tourney_id}/results"
    key = f"atp_results_{year}_{atp_tourney_id}"
    data = await _fetch_json_playwright(url, key, wait_ms=3000)
    if data is None:
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for k in ("matches", "data", "draws", "results"):
            if k in data and isinstance(data[k], list):
                return data[k]
    return []


# ---------------------------------------------------------------------------
# Schema normalisation
# ---------------------------------------------------------------------------

def _safe(d: dict, *keys, default=None):
    for k in keys:
        if isinstance(d, dict):
            d = d.get(k)
        else:
            return default
        if d is None:
            return default
    return d


def _level(category: str | None) -> str:
    if not category:
        return "A"
    for k, v in LEVEL_MAP.items():
        if k.lower() in (category or "").lower():
            return v
    return "A"


def _surface(s: str | None) -> str:
    if not s:
        return "Hard"
    s = s.strip().title()
    return SURFACE_MAP.get(s, s)


def _tourney_date(d: str | None) -> str:
    """Normalise to YYYYMMDD."""
    if not d:
        return ""
    d = re.sub(r"[^\d]", "", d)
    return d[:8] if len(d) >= 8 else d.ljust(8, "0")


def _stat(d: dict, key: str):
    """Extract a numeric stat, return None if missing."""
    v = d.get(key)
    if v in (None, "", "N/A", "-"):
        return None
    try:
        return int(v)
    except (ValueError, TypeError):
        try:
            return float(v)
        except (ValueError, TypeError):
            return None


def normalise_atp_match(
    raw: dict,
    tourney_id: str,
    tourney_name: str,
    surface: str,
    draw_size: int,
    tourney_level: str,
    tourney_date: str,
    match_num: int,
) -> dict:
    """
    Convert one raw ATP JSON match record to a Sackmann-schema dict.
    Field names vary across ATP API versions; we try multiple keys.
    """
    w = _safe(raw, "winner") or raw
    l = _safe(raw, "loser") or {}

    # If the raw record doesn't have winner/loser split, try top-level
    winner_name = (
        _safe(w, "displayName") or _safe(w, "fullName") or
        _safe(raw, "winnerName") or ""
    )
    loser_name = (
        _safe(l, "displayName") or _safe(l, "fullName") or
        _safe(raw, "loserName") or ""
    )

    stats_w = _safe(raw, "winnerStats") or {}
    stats_l = _safe(raw, "loserStats") or {}

    row: dict[str, Any] = {
        "tourney_id":           tourney_id,
        "tourney_name":         tourney_name,
        "surface":              surface,
        "draw_size":            draw_size,
        "tourney_level":        tourney_level,
        "tourney_date":         tourney_date,
        "match_num":            match_num,
        "winner_id":            _safe(w, "playerId") or _safe(raw, "winnerId") or "",
        "winner_seed":          _safe(raw, "winnerSeed") or _safe(w, "seed") or "",
        "winner_entry":         _safe(raw, "winnerEntry") or _safe(w, "entry") or "",
        "winner_name":          winner_name,
        "winner_hand":          _safe(w, "hand") or "",
        "winner_ht":            _safe(w, "height"),
        "winner_ioc":           _safe(w, "countryCode") or _safe(w, "ioc") or "",
        "winner_age":           _safe(w, "age"),
        "loser_id":             _safe(l, "playerId") or _safe(raw, "loserId") or "",
        "loser_seed":           _safe(raw, "loserSeed") or _safe(l, "seed") or "",
        "loser_entry":          _safe(raw, "loserEntry") or _safe(l, "entry") or "",
        "loser_name":           loser_name,
        "loser_hand":           _safe(l, "hand") or "",
        "loser_ht":             _safe(l, "height"),
        "loser_ioc":            _safe(l, "countryCode") or _safe(l, "ioc") or "",
        "loser_age":            _safe(l, "age"),
        "score":                _safe(raw, "score") or _safe(raw, "result") or "",
        "best_of":              _safe(raw, "bestOf") or _safe(raw, "best_of") or 3,
        "round":                _safe(raw, "round") or _safe(raw, "roundName") or "",
        "minutes":              _safe(raw, "minutes") or _safe(raw, "duration"),
        # Serve stats – winner
        "w_ace":    _stat(stats_w, "aces"),
        "w_df":     _stat(stats_w, "doubleFaults"),
        "w_svpt":   _stat(stats_w, "servicePointsPlayed"),
        "w_1stIn":  _stat(stats_w, "firstServeIn"),
        "w_1stWon": _stat(stats_w, "firstServePointsWon"),
        "w_2ndWon": _stat(stats_w, "secondServePointsWon"),
        "w_SvGms":  _stat(stats_w, "serviceGamesPlayed"),
        "w_bpSaved":_stat(stats_w, "breakPointsSaved"),
        "w_bpFaced":_stat(stats_w, "breakPointsFaced"),
        # Serve stats – loser
        "l_ace":    _stat(stats_l, "aces"),
        "l_df":     _stat(stats_l, "doubleFaults"),
        "l_svpt":   _stat(stats_l, "servicePointsPlayed"),
        "l_1stIn":  _stat(stats_l, "firstServeIn"),
        "l_1stWon": _stat(stats_l, "firstServePointsWon"),
        "l_2ndWon": _stat(stats_l, "secondServePointsWon"),
        "l_SvGms":  _stat(stats_l, "serviceGamesPlayed"),
        "l_bpSaved":_stat(stats_l, "breakPointsSaved"),
        "l_bpFaced":_stat(stats_l, "breakPointsFaced"),
        # Rankings
        "winner_rank":         _safe(w, "rank") or _safe(raw, "winnerRank"),
        "winner_rank_points":  _safe(w, "rankingPoints") or _safe(raw, "winnerRankPoints"),
        "loser_rank":          _safe(l, "rank") or _safe(raw, "loserRank"),
        "loser_rank_points":   _safe(l, "rankingPoints") or _safe(raw, "loserRankPoints"),
    }
    return row


# ---------------------------------------------------------------------------
# Main scrape loop
# ---------------------------------------------------------------------------

async def scrape_year(year: int) -> pd.DataFrame:
    print(f"[ATP] Fetching tournament list for {year}…")
    tournaments = await fetch_tournament_list(year)

    if not tournaments:
        print(f"[ATP] No tournament data returned for {year}.  "
              f"Check the ATP endpoint or your network connection.")
        return pd.DataFrame(columns=SACKMANN_COLS)

    print(f"[ATP] Found {len(tournaments)} tournaments")

    all_rows: list[dict] = []

    for t in tournaments:
        t_name  = _safe(t, "name") or _safe(t, "tournamentName") or "Unknown"
        t_id    = str(_safe(t, "id") or _safe(t, "tournamentId") or "")
        slug    = _safe(t, "slug") or t_id
        surface = _surface(_safe(t, "surface") or _safe(t, "courtSurface"))
        level   = _level(_safe(t, "category") or _safe(t, "type"))
        t_date  = _tourney_date(_safe(t, "startDate") or _safe(t, "date") or "")
        try:
            draw_size = int(_safe(t, "drawSize") or 0)
        except (ValueError, TypeError):
            draw_size = 0

        # Compose a Sackmann-style tourney_id  e.g. "2025-0339"
        sackmann_id = f"{year}-{t_id}"

        print(f"  [ATP] {t_name} ({surface}) …", end="", flush=True)
        matches_raw = await fetch_tournament_results(year, t_id, slug)

        for i, m in enumerate(matches_raw, 1):
            # Skip walkovers / retirements (no winner clearly determined)
            score = str(_safe(m, "score") or _safe(m, "result") or "")
            if any(x in score.upper() for x in ("W/O", "RET", "DEF")):
                continue
            row = normalise_atp_match(
                m, sackmann_id, t_name, surface, draw_size, level, t_date, i
            )
            all_rows.append(row)

        print(f" {len(matches_raw)} matches")
        await asyncio.sleep(0.5)   # polite delay between tournaments

    df = pd.DataFrame(all_rows, columns=SACKMANN_COLS)
    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Scrape ATP Tour results")
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--clear-cache", action="store_true",
                        help="Delete cached responses before scraping")
    args = parser.parse_args()

    if args.clear_cache:
        for f in CACHE_DIR.glob("*.json"):
            f.unlink()
        print(f"[ATP] Cache cleared ({CACHE_DIR})")

    df = asyncio.run(scrape_year(args.year))

    if df.empty:
        print("[ATP] No data scraped – output not written.")
        return

    out_path = OUT_DIR / f"atp_results_{args.year}.csv"
    df.to_csv(out_path, index=False)
    print(f"\n[ATP] Saved {len(df)} matches → {out_path}")


if __name__ == "__main__":
    main()
