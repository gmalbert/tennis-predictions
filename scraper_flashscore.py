"""
scraper_flashscore.py
---------------------
Scrape **completed** ATP tennis match results from Flashscore for a given
year.

Flashscore has heavy bot-protection, so we use Playwright to:
  1. Navigate to the tennis results archive page for each calendar week.
  2. Intercept the internal XHR/fetch feed responses.
  3. Parse them using the same custom-delimiter logic as the live feed.
  4. Cache every raw feed payload to disk.

The feed format is identical to the live-schedule feed documented in
05_scheduling.md (fields delimited by ¬ and ÷, sections by ~), with
the key difference that we target status code "3" (finished) matches.

Output: data_files/flashscore_results_<year>.csv  (Sackmann-schema)

Usage:
    python scraper_flashscore.py              # year=2025
    python scraper_flashscore.py --year 2024
    python scraper_flashscore.py --start-date 2025-01-01 --end-date 2025-03-01
    python scraper_flashscore.py --clear-cache
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
BASE_DIR  = Path(__file__).parent
CACHE_DIR = BASE_DIR / "cache" / "flashscore"
OUT_DIR   = BASE_DIR / "data_files"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Live feed URL (from 05_scheduling.md) – works for today's matches
FS_LIVE_URL = "https://www.flashscore.com/x/feed/f_2_0_2_en-gb_1"

# Historical/results-by-date feed – Flashscore's internal date feed
# The date parameter is a Unix timestamp for the desired day
FS_DATE_FEED = "https://www.flashscore.com/x/feed/d_su_1_en-gb_1_{ts}"

# Results archive page – Playwright navigates here per date
FS_RESULTS_BASE = "https://www.flashscore.com/tennis/#/results/"

FS_HEADERS = {
    "x-fsign":          "SW9D1eZo",
    "User-Agent":       "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36",
    "Referer":          "https://www.flashscore.com/tennis/",
    "Accept":           "*/*",
    "Accept-Language":  "en-GB,en;q=0.9",
    "Origin":           "https://www.flashscore.com",
}

PAGE_DELAY_MS = 3_000   # wait after page load before closing

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

LEVEL_MAP = {
    "grand slam":    "G",
    "masters 1000":  "M",
    "atp finals":    "F",
    "atp 500":       "A",
    "atp 250":       "A",
    "challenger":    "C",
    "itf":           "I",
    "davis cup":     "D",
}


# ---------------------------------------------------------------------------
# Cache helpers (raw feed text)
# ---------------------------------------------------------------------------

def _cache_path(key: str) -> Path:
    h = hashlib.md5(key.encode()).hexdigest()
    return CACHE_DIR / f"{h}.txt"


def load_cache(key: str) -> str | None:
    p = _cache_path(key)
    return p.read_text(encoding="utf-8", errors="replace") if p.exists() else None


def save_cache(key: str, text: str) -> None:
    _cache_path(key).write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Feed parsing logic (same delimiter format as live feed)
# ---------------------------------------------------------------------------

def _parse_fields(section: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for part in section.split("\xac"):   # ¬  (0xAC)
        if "\xf7" in part:               # ÷  (0xF7)
            k, _, v = part.partition("\xf7")
            fields[k] = v
    return fields


def _extract_surface(header: str) -> str:
    h = header.lower()
    if "clay"   in h: return "Clay"
    if "grass"  in h: return "Grass"
    if "carpet" in h: return "Carpet"
    return "Hard"


def _extract_tournament_name(header: str) -> str:
    if ": " in header:
        header = header.split(": ", 1)[1]
    if ", " in header:
        header = header.rsplit(", ", 1)[0]
    return header.strip()


def _extract_level(header: str) -> str:
    h = header.lower()
    for k, v in LEVEL_MAP.items():
        if k in h:
            return v
    return "A"


def _build_score(fields: dict) -> str:
    """Reconstruct set scores from BA/BB (set1 p1/p2), BC/BD, BE/BF."""
    sets = []
    for p1_key, p2_key in [("BA", "BB"), ("BC", "BD"), ("BE", "BF"), ("BG", "BH")]:
        s1 = fields.get(p1_key, "")
        s2 = fields.get(p2_key, "")
        if s1 and s2:
            sets.append(f"{s1}-{s2}")
    return " ".join(sets)


def _determine_winner(fields: dict) -> tuple[str, str]:
    """
    Return (winner_name, loser_name) based on set scores.
    Flashscore AE = player1 name, AF = player2 name.
    Player who won more sets wins.  Ties go to the higher-set-score player.
    """
    p1 = fields.get("AE", "")
    p2 = fields.get("AF", "")
    p1_sets = p2_sets = 0
    for p1_key, p2_key in [("BA", "BB"), ("BC", "BD"), ("BE", "BF"), ("BG", "BH")]:
        try:
            s1 = int(fields.get(p1_key, 0) or 0)
            s2 = int(fields.get(p2_key, 0) or 0)
            if s1 > s2: p1_sets += 1
            elif s2 > s1: p2_sets += 1
        except (ValueError, TypeError):
            pass
    if p2_sets > p1_sets:
        return p2, p1
    return p1, p2   # default to p1 if equal / undetermined


def parse_feed(raw: str, target_date_str: str = "") -> list[dict]:
    """
    Parse a Flashscore custom-delimited feed into a list of match dicts.
    Only includes finished matches (status code AB == "3").
    """
    current_tourney   = ""
    current_surface   = "Hard"
    current_level     = "A"
    current_date_norm = target_date_str.replace("-", "") if target_date_str else ""
    rows: list[dict] = []
    match_num = 0

    for section in raw.split("~"):
        if not section:
            continue
        fields = _parse_fields(section)

        # Tournament header section
        if "ZA" in fields:
            hdr = fields["ZA"]
            current_tourney   = _extract_tournament_name(hdr)
            current_surface   = _extract_surface(hdr)
            current_level     = _extract_level(hdr)
            current_date_norm = ""   # reset per tournament
            continue

        if "AA" not in fields:
            continue

        # Only finished matches
        status = fields.get("AB", "0")
        if status != "3":
            continue

        p1_name = fields.get("AE", "")
        p2_name = fields.get("AF", "")
        if not p1_name or not p2_name:
            continue

        # Start time → YYYYMMDD
        ts_raw = fields.get("AD", "")
        if ts_raw:
            try:
                dt = datetime.fromtimestamp(int(ts_raw), tz=timezone.utc)
                date_norm = dt.strftime("%Y%m%d")
            except (ValueError, TypeError):
                date_norm = current_date_norm
        else:
            date_norm = current_date_norm

        score   = _build_score(fields)
        winner, loser = _determine_winner(fields)

        # Skip walkover/retirement proxies (empty scores)
        if not score:
            continue

        match_num += 1
        slug = re.sub(r"[^\w]", "", current_tourney or "UNKNOWN")
        tourney_id = f"{date_norm[:4]}-{slug[:10].upper()}" if date_norm else f"UNK-{slug[:10].upper()}"

        rows.append({
            "tourney_id":          tourney_id,
            "tourney_name":        current_tourney,
            "surface":             current_surface,
            "draw_size":           None,
            "tourney_level":       current_level,
            "tourney_date":        date_norm,
            "match_num":           match_num,
            "winner_id":           "",
            "winner_seed":         "",
            "winner_entry":        "",
            "winner_name":         winner,
            "winner_hand":         "",
            "winner_ht":           None,
            "winner_ioc":          "",
            "winner_age":          None,
            "loser_id":            "",
            "loser_seed":          "",
            "loser_entry":         "",
            "loser_name":          loser,
            "loser_hand":          "",
            "loser_ht":            None,
            "loser_ioc":           "",
            "loser_age":           None,
            "score":               score,
            "best_of":             3,
            "round":               fields.get("BZ", ""),
            "minutes":             None,
            "w_ace": None, "w_df": None, "w_svpt": None,
            "w_1stIn": None, "w_1stWon": None, "w_2ndWon": None,
            "w_SvGms": None, "w_bpSaved": None, "w_bpFaced": None,
            "l_ace": None, "l_df": None, "l_svpt": None,
            "l_1stIn": None, "l_1stWon": None, "l_2ndWon": None,
            "l_SvGms": None, "l_bpSaved": None, "l_bpFaced": None,
            "winner_rank": None, "winner_rank_points": None,
            "loser_rank":  None, "loser_rank_points":  None,
        })

    return rows


# ---------------------------------------------------------------------------
# Fetch a single day's completed results via Playwright
# ---------------------------------------------------------------------------

async def fetch_day_playwright(target: date) -> str:
    """
    Navigate to Flashscore tennis results for `target` date.
    Intercept the internal feed XHR and return the raw text payload.
    """
    ts = int(datetime(target.year, target.month, target.day, 0, 0, 0,
                      tzinfo=timezone.utc).timestamp())
    feed_url = FS_DATE_FEED.format(ts=ts)

    try:
        from playwright.async_api import async_playwright, TimeoutError as PWTimeout
    except ImportError:
        raise SystemExit("pip install playwright && playwright install chromium")

    captured_text = []

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent=FS_HEADERS["User-Agent"],
            extra_http_headers={
                "x-fsign":         FS_HEADERS["x-fsign"],
                "Referer":         FS_HEADERS["Referer"],
                "Accept-Language": FS_HEADERS["Accept-Language"],
            },
            viewport={"width": 1280, "height": 900},
        )

        async def on_response(response):
            url = response.url
            # Capture any feed response (contains our custom delimiter chars)
            if "flashscore.com/x/feed" in url or "x/feed" in url:
                try:
                    text = await response.text()
                    if "\xac" in text or "\xf7" in text:   # ¬ ÷
                        captured_text.append(text)
                except Exception:
                    pass

        page = await context.new_page()
        page.on("response", on_response)

        # Navigate to the results page for this date
        results_url = FS_RESULTS_BASE
        try:
            await page.goto(results_url, wait_until="networkidle", timeout=30_000)
        except PWTimeout:
            pass

        # Also try direct feed URL
        try:
            resp = await page.goto(feed_url, wait_until="domcontentloaded", timeout=15_000)
            if resp and resp.ok:
                text = await resp.text()
                if "\xac" in text or "\xf7" in text:
                    captured_text.append(text)
        except PWTimeout:
            pass

        await page.wait_for_timeout(PAGE_DELAY_MS)
        await browser.close()

    return "\n".join(captured_text)


async def fetch_day_requests(target: date) -> str:
    """
    Try to fetch the Flashscore feed for `target` date using requests
    (cheaper than Playwright; may fail if Cloudflare blocks).
    """
    import requests as _requests
    ts = int(datetime(target.year, target.month, target.day,
                      tzinfo=timezone.utc).timestamp())
    feed_url = FS_DATE_FEED.format(ts=ts)

    try:
        resp = _requests.get(feed_url, headers=FS_HEADERS, timeout=15)
        if resp.ok and ("\xac" in resp.text or "\xf7" in resp.text):
            return resp.text
    except Exception:
        pass
    return ""


# ---------------------------------------------------------------------------
# Main scrape loop
# ---------------------------------------------------------------------------

async def scrape_date_range(start: date, end: date) -> pd.DataFrame:
    """
    Iterate over each day in [start, end] and collect completed matches.
    Uses requests first (fast); falls back to Playwright if blocked.
    """
    all_rows: list[dict] = []
    seen_keys: set[str] = set()

    current = start
    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        cache_key = f"fs_{date_str}"
        print(f"  [FS] {date_str}", end="", flush=True)

        raw = load_cache(cache_key)
        if raw is None:
            raw = await fetch_day_requests(current)
            if not raw:
                print(" → requests blocked, trying Playwright…", end="", flush=True)
                raw = await fetch_day_playwright(current)
            if raw:
                save_cache(cache_key, raw)
            else:
                print(" — no data")
                current += timedelta(days=1)
                await asyncio.sleep(0.5)
                continue

        rows = parse_feed(raw, date_str)

        added = 0
        for r in rows:
            dup_key = (
                f"{r['tourney_date']}|"
                f"{min(r['winner_name'], r['loser_name'])}|"
                f"{max(r['winner_name'], r['loser_name'])}"
            )
            if dup_key in seen_keys:
                continue
            seen_keys.add(dup_key)
            all_rows.append(r)
            added += 1

        print(f" — {added} new matches")
        current += timedelta(days=1)
        await asyncio.sleep(0.3)

    df = pd.DataFrame(all_rows, columns=SACKMANN_COLS)
    df = df.sort_values("tourney_date").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape Flashscore completed results")
    parser.add_argument("--year",       type=int, default=2025)
    parser.add_argument("--start-date", type=str, default=None,
                        help="YYYY-MM-DD  (overrides --year start)")
    parser.add_argument("--end-date",   type=str, default=None,
                        help="YYYY-MM-DD  (defaults to yesterday)")
    parser.add_argument("--clear-cache", action="store_true")
    args = parser.parse_args()

    if args.clear_cache:
        for f in CACHE_DIR.glob("*.txt"):
            f.unlink()
        print(f"[FS] Cache cleared ({CACHE_DIR})")

    yesterday = date.today() - timedelta(days=1)
    start = date.fromisoformat(args.start_date) if args.start_date else date(args.year, 1, 1)
    end   = date.fromisoformat(args.end_date)   if args.end_date   else min(date(args.year, 12, 31), yesterday)

    print(f"[FS] Scraping {start} → {end}…")
    df = asyncio.run(scrape_date_range(start, end))

    if df.empty:
        print("[FS] No data scraped.")
        return

    out_path = OUT_DIR / f"flashscore_results_{args.year}.csv"
    df.to_csv(out_path, index=False)
    print(f"\n[FS] Saved {len(df)} matches → {out_path}")


if __name__ == "__main__":
    main()
