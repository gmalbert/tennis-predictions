"""
scraper_itf.py
--------------
Scrape completed match results from the ITF (International Tennis
Federation) website for a given year.

Covers: ITF Pro Circuit (Men's / Women's), Davis Cup, Billie Jean King Cup.
ATP Challengers and main-tour events are better covered by scraper_atp.py
and scraper_flashscore.py.

The ITF website uses server-side rendering with some JavaScript hydration.
We try two approaches in order:
  1. Direct HTTP/JSON requests against ITF's internal API endpoints
     (intercepted from browser devtools — no Playwright needed).
  2. Playwright fallback for pages that require JS execution.

Raw JSON is cached to disk.

Output: data_files/itf_results_<year>.csv  (Sackmann-schema columns)

Usage:
    python scraper_itf.py               # year=2025
    python scraper_itf.py --year 2024
    python scraper_itf.py --tour M      # M=Men, W=Women, B=Boys, G=Girls
    python scraper_itf.py --clear-cache
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
BASE_DIR  = Path(__file__).parent
CACHE_DIR = BASE_DIR / "cache" / "itf"
OUT_DIR   = BASE_DIR / "data_files"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

ITF_BASE = "https://www.itftennis.com"

# Main internal JSON API (discovered via browser devtools)
ITF_API_BASE        = "https://api.itftennis.com"

# Tournament list endpoint
TOURNEY_SEARCH_URL  = ITF_API_BASE + "/api/tournament/search"

# Individual tournament draws  (results are embedded in draws)
TOURNEY_DRAWS_URL   = ITF_API_BASE + "/api/draws?tournamentId={tid}&year={year}"
TOURNEY_RESULTS_URL = ITF_API_BASE + "/api/results?tournamentId={tid}&year={year}"

# Fallback: ITF website search
ITF_WEB_SEARCH      = ITF_BASE + "/en/competition/tournaments/tennis/men/{year}/"

REQUEST_DELAY = 1.2   # seconds

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept":         "application/json, text/plain, */*",
    "Accept-Language":"en-US,en;q=0.9",
    "Origin":         "https://www.itftennis.com",
    "Referer":        "https://www.itftennis.com/",
}

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

SURFACE_MAP = {
    "hard":   "Hard",
    "clay":   "Clay",
    "grass":  "Grass",
    "carpet": "Carpet",
    "acrylic":"Hard",
    "indoor hard": "Hard",
}

LEVEL_MAP_ITF = {
    "grand slam":    "G",
    "masters":       "M",
    "atp 500":       "A",
    "atp 250":       "A",
    "challenger":    "C",
    "itf":           "I",
    "futures":       "I",
    "pro circuit":   "I",
    "davis cup":     "D",
    "billie jean":   "D",
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
            pass
    return None


def save_cache(key: str, data: Any) -> None:
    _cache_path(key).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

_session = requests.Session()
_session.headers.update(HEADERS)


def fetch_json(url: str, params: dict | None = None, cache_key: str | None = None) -> Any:
    key = cache_key or url + json.dumps(params or {})
    cached = load_cache(key)
    if cached is not None:
        return cached

    try:
        resp = _session.get(url, params=params, timeout=20)
        if not resp.ok:
            print(f"    [ITF] HTTP {resp.status_code} for {url}")
            return None
        data = resp.json()
        save_cache(key, data)
        time.sleep(REQUEST_DELAY)
        return data
    except (requests.RequestException, ValueError) as e:
        print(f"    [ITF] Request error: {e}")
        return None


async def fetch_json_playwright(url: str, cache_key: str) -> Any:
    """Playwright-based JSON fetch for pages that block direct requests."""
    cached = load_cache(cache_key)
    if cached is not None:
        return cached

    try:
        from playwright.async_api import async_playwright
    except ImportError:
        raise SystemExit("pip install playwright && playwright install chromium")

    captured = []

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent=HEADERS["User-Agent"],
            extra_http_headers={"Accept-Language": HEADERS["Accept-Language"]},
        )

        async def on_response(response):
            if "application/json" in response.headers.get("content-type", ""):
                try:
                    captured.append(await response.json())
                except Exception:
                    pass

        page = await context.new_page()
        page.on("response", on_response)
        try:
            await page.goto(url, wait_until="networkidle", timeout=30_000)
            await page.wait_for_timeout(2000)
        except Exception:
            pass
        finally:
            await browser.close()

    data = captured[0] if captured else None
    if data:
        save_cache(cache_key, data)
    return data


# ---------------------------------------------------------------------------
# Tournament list
# ---------------------------------------------------------------------------

def fetch_tournament_list(year: int, tour: str = "M") -> list[dict]:
    """
    Fetch the ITF tournament list for `year` and gender `tour` (M/W/B/G).
    Tries the internal API first, then the web page via Playwright.
    """
    params = {
        "fields":      "id,name,startDate,endDate,surface,category,drawSize,venue",
        "year":        year,
        "tourType":    tour,
        "pageSize":    500,
        "page":        1,
    }
    key = f"itf_tournaments_{year}_{tour}"
    data = fetch_json(TOURNEY_SEARCH_URL, params=params, cache_key=key)

    if data is None:
        # Fallback JSON attempt with a slightly different endpoint structure
        alt_url = f"{ITF_API_BASE}/api/tournament?year={year}&tourType={tour}&pageSize=500"
        data = fetch_json(alt_url, cache_key=f"itf_tournaments_alt_{year}_{tour}")

    if data is None:
        return []

    # Normalise – API can return list or {"data": [...], "total": N}
    if isinstance(data, list):
        return data
    for k in ("data", "items", "tournaments", "results"):
        if k in data and isinstance(data[k], list):
            return data[k]
    return []


# ---------------------------------------------------------------------------
# Match / draw results for a single tournament
# ---------------------------------------------------------------------------

def fetch_tournament_matches(tid: str, year: int) -> list[dict]:
    """Return raw match records from the results or draws endpoint."""
    # Try results endpoint
    key_r = f"itf_results_{year}_{tid}"
    data = fetch_json(
        TOURNEY_RESULTS_URL.format(tid=tid, year=year),
        cache_key=key_r,
    )
    if data:
        records = _extract_list(data)
        if records:
            return records

    # Try draws endpoint
    key_d = f"itf_draws_{year}_{tid}"
    data = fetch_json(
        TOURNEY_DRAWS_URL.format(tid=tid, year=year),
        cache_key=key_d,
    )
    if data:
        return _extract_list(data)

    return []


def _extract_list(data: Any) -> list[dict]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for k in ("matches", "results", "data", "draws", "items"):
            if k in data and isinstance(data[k], list):
                return data[k]
    return []


# ---------------------------------------------------------------------------
# Data helpers
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


def _surface(s: str | None) -> str:
    if not s:
        return "Hard"
    return SURFACE_MAP.get(s.lower().strip(), s.title())


def _level(cat: str | None) -> str:
    if not cat:
        return "I"
    c = cat.lower()
    for k, v in LEVEL_MAP_ITF.items():
        if k in c:
            return v
    return "I"


def _norm_date(s: str | None) -> str:
    if not s:
        return ""
    clean = re.sub(r"[^\d]", "", s)
    return clean[:8] if len(clean) >= 8 else clean.ljust(8, "0")


def _player_name(d: dict | None) -> str:
    if not d:
        return ""
    return (
        _safe(d, "fullName") or
        _safe(d, "displayName") or
        f"{_safe(d, 'firstName', default='')} {_safe(d, 'lastName', default='')}".strip()
    )


# ---------------------------------------------------------------------------
# Normalise one raw ITF match record
# ---------------------------------------------------------------------------

def normalise_itf_match(
    raw: dict,
    tourney_id: str,
    tourney_name: str,
    surface: str,
    draw_size: int,
    tourney_level: str,
    tourney_date: str,
    match_num: int,
) -> dict | None:
    """
    Convert an ITF API match record to Sackmann-schema dict.
    Returns None if the match cannot be determined (no winner info).
    """
    winner_raw = _safe(raw, "winner") or {}
    loser_raw  = _safe(raw, "loser")  or {}

    # Some endpoints use side1/side2 with a winnerId
    if not winner_raw:
        side1    = _safe(raw, "side1") or {}
        side2    = _safe(raw, "side2") or {}
        winner_id_raw = str(_safe(raw, "winnerId") or "")
        s1_id    = str(_safe(side1, "id") or _safe(side1, "playerId") or "")
        if winner_id_raw and winner_id_raw == s1_id:
            winner_raw, loser_raw = side1, side2
        elif winner_id_raw:
            winner_raw, loser_raw = side2, side1
        else:
            return None   # can't determine

    winner_name = _player_name(winner_raw)
    loser_name  = _player_name(loser_raw)
    if not winner_name:
        return None

    score = _safe(raw, "score") or _safe(raw, "result") or ""

    # Skip walkovers
    if any(x in str(score).upper() for x in ("W/O", "WALKOVER", "RET", "DEF")):
        return None

    return {
        "tourney_id":          tourney_id,
        "tourney_name":        tourney_name,
        "surface":             surface,
        "draw_size":           draw_size,
        "tourney_level":       tourney_level,
        "tourney_date":        tourney_date,
        "match_num":           match_num,
        "winner_id":           str(_safe(winner_raw, "id") or _safe(winner_raw, "playerId") or ""),
        "winner_seed":         _safe(raw, "winnerSeed") or _safe(winner_raw, "seed") or "",
        "winner_entry":        _safe(raw, "winnerEntry") or _safe(winner_raw, "entry") or "",
        "winner_name":         winner_name,
        "winner_hand":         _safe(winner_raw, "hand") or "",
        "winner_ht":           _safe(winner_raw, "height"),
        "winner_ioc":          _safe(winner_raw, "nationality") or _safe(winner_raw, "ioc") or "",
        "winner_age":          _safe(winner_raw, "age"),
        "loser_id":            str(_safe(loser_raw, "id") or _safe(loser_raw, "playerId") or ""),
        "loser_seed":          _safe(raw, "loserSeed") or _safe(loser_raw, "seed") or "",
        "loser_entry":         _safe(raw, "loserEntry") or _safe(loser_raw, "entry") or "",
        "loser_name":          loser_name,
        "loser_hand":          _safe(loser_raw, "hand") or "",
        "loser_ht":            _safe(loser_raw, "height"),
        "loser_ioc":           _safe(loser_raw, "nationality") or _safe(loser_raw, "ioc") or "",
        "loser_age":           _safe(loser_raw, "age"),
        "score":               score,
        "best_of":             _safe(raw, "bestOf") or 3,
        "round":               _safe(raw, "round") or _safe(raw, "roundName") or "",
        "minutes":             _safe(raw, "minutes") or _safe(raw, "duration"),
        "w_ace": None, "w_df": None, "w_svpt": None,
        "w_1stIn": None, "w_1stWon": None, "w_2ndWon": None,
        "w_SvGms": None, "w_bpSaved": None, "w_bpFaced": None,
        "l_ace": None, "l_df": None, "l_svpt": None,
        "l_1stIn": None, "l_1stWon": None, "l_2ndWon": None,
        "l_SvGms": None, "l_bpSaved": None, "l_bpFaced": None,
        "winner_rank":         _safe(winner_raw, "rank"),
        "winner_rank_points":  _safe(winner_raw, "rankingPoints"),
        "loser_rank":          _safe(loser_raw,  "rank"),
        "loser_rank_points":   _safe(loser_raw,  "rankingPoints"),
    }


# ---------------------------------------------------------------------------
# Main scrape loop
# ---------------------------------------------------------------------------

def scrape_year(year: int, tour: str = "M") -> pd.DataFrame:
    print(f"[ITF] Fetching tournament list for {year} (tour={tour})…")
    tournaments = fetch_tournament_list(year, tour)

    if not tournaments:
        print(f"[ITF] No tournaments returned for year={year} tour={tour}. "
              f"The ITF API endpoint may have changed – check TOURNEY_SEARCH_URL.")
        return pd.DataFrame(columns=SACKMANN_COLS)

    print(f"[ITF] Found {len(tournaments)} tournaments")
    all_rows: list[dict] = []

    for t in tournaments:
        t_name  = _safe(t, "name") or "Unknown"
        t_id    = str(_safe(t, "id") or _safe(t, "tournamentId") or "")
        surface = _surface(_safe(t, "surface"))
        level   = _level(_safe(t, "category") or _safe(t, "type"))
        t_date  = _norm_date(_safe(t, "startDate") or _safe(t, "date") or "")
        try:
            draw_size = int(_safe(t, "drawSize") or 0)
        except (ValueError, TypeError):
            draw_size = 0

        sackmann_id = f"{year}-ITF{t_id}"

        print(f"  [ITF] {t_name} ({surface}) …", end="", flush=True)
        matches_raw = fetch_tournament_matches(t_id, year)

        added = 0
        for i, m in enumerate(matches_raw, 1):
            row = normalise_itf_match(m, sackmann_id, t_name, surface, draw_size, level, t_date, i)
            if row:
                all_rows.append(row)
                added += 1

        print(f" {added} matches")

    df = pd.DataFrame(all_rows, columns=SACKMANN_COLS)
    df = df.sort_values("tourney_date").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape ITF match results")
    parser.add_argument("--year",        type=int, default=2025)
    parser.add_argument("--tour",        type=str, default="M",
                        choices=["M", "W", "B", "G"],
                        help="M=Men, W=Women, B=Boys, G=Girls")
    parser.add_argument("--clear-cache", action="store_true")
    args = parser.parse_args()

    if args.clear_cache:
        for f in CACHE_DIR.glob("*.json"):
            f.unlink()
        print(f"[ITF] Cache cleared ({CACHE_DIR})")

    df = scrape_year(args.year, tour=args.tour)

    if df.empty:
        print("[ITF] No data scraped.")
        return

    out_path = OUT_DIR / f"itf_results_{args.year}_{args.tour}.csv"
    df.to_csv(out_path, index=False)
    print(f"\n[ITF] Saved {len(df)} matches → {out_path}")


if __name__ == "__main__":
    main()
