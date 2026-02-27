"""
scraper_tennis_abstract.py
--------------------------
Scrape completed ATP match results for a given year from Tennis Abstract
(tennisabstract.com).

Tennis Abstract is Jeff Sackmann's site and has player-level match
histories per season with surface, round, opponent and result.  The
pages are server-rendered HTML, so requests + BeautifulSoup works.

Strategy:
  1. Derive the list of player slugs from the local atp_players.csv
     (or fall back to a hardcoded top-150 list if the CSV isn't available).
  2. For each player, fetch their <year> match page.
  3. Parse every match row; skip duplicates (each match appears for both players).
  4. Cache raw HTML per player-year so we never re-fetch.

Output: data_files/ta_results_<year>.csv  (Sackmann-schema columns)

Usage:
    python scraper_tennis_abstract.py              # year=2025
    python scraper_tennis_abstract.py --year 2024
    python scraper_tennis_abstract.py --clear-cache
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
BASE_DIR  = Path(__file__).parent
CACHE_DIR = BASE_DIR / "cache" / "tennis_abstract"
OUT_DIR   = BASE_DIR / "data_files"
ATP_DIR   = BASE_DIR / "tennis_atp"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

TA_BASE        = "https://www.tennisabstract.com"
PLAYER_URL     = TA_BASE + "/cgi-bin/player.cgi"
PLAYER_LIST_URL= TA_BASE + "/cgi-bin/leaders.cgi?f=B0s1&stype=rank&p=0"

REQUEST_DELAY  = 3.5   # seconds between requests – TA rate-limits aggressively

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": TA_BASE,
    "Accept-Language": "en-US,en;q=0.9",
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

# Tennis Abstract tournament level strings → Sackmann code
LEVEL_MAP = {
    "Grand Slam":   "G",
    "Masters":      "M",
    "ATP Finals":   "F",
    "500":          "A",
    "250":          "A",
    "Davis Cup":    "D",
    "Olympics":     "O",
}

SURFACE_NORM = {"hard": "Hard", "clay": "Clay", "grass": "Grass", "carpet": "Carpet"}


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_path(key: str) -> Path:
    h = hashlib.md5(key.encode()).hexdigest()
    return CACHE_DIR / f"{h}.html"


def load_cache(key: str) -> str | None:
    p = _cache_path(key)
    return p.read_text(encoding="utf-8", errors="replace") if p.exists() else None


def save_cache(key: str, html: str) -> None:
    _cache_path(key).write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

_session = requests.Session()
_session.headers.update(HEADERS)


def fetch_html(url: str, params: dict | None = None, cache_key: str | None = None) -> str | None:
    key = cache_key or url
    cached = load_cache(key)
    if cached is not None:
        return cached

    try:
        resp = _session.get(url, params=params, timeout=20)
        resp.raise_for_status()
        html = resp.text
        save_cache(key, html)
        time.sleep(REQUEST_DELAY)
        return html
    except requests.RequestException as e:
        print(f"  [TA] Request failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Player list
# ---------------------------------------------------------------------------

def load_player_names_from_csv(atp_dir: Path) -> list[str]:
    """
    Return a list of player names in the "FirstLast" Tennis Abstract slug
    format, derived from the local atp_players.csv.
    Sorted by player_id DESCENDING so recent/active players come first.
    """
    p = atp_dir / "atp_players.csv"
    if not p.exists():
        return []
    df = pd.read_csv(p, encoding="utf-8", on_bad_lines="skip", low_memory=False)
    # Sort newest (highest player_id) first – recent players have higher IDs
    if "player_id" in df.columns:
        df = df.sort_values("player_id", ascending=False)
    names = []
    for _, row in df.iterrows():
        first = str(row.get("name_first", "") or "").strip()
        last  = str(row.get("name_last",  "") or "").strip()
        if first and last:
            names.append(f"{first}{last}".replace(" ", ""))
    return names


def fetch_ranked_player_names() -> list[str]:
    """
    Scrape Tennis Abstract's current rankings leaderboard to get the
    active player slug list.
    """
    html = fetch_html(PLAYER_LIST_URL, cache_key="ta_leaders_current")
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    names = []
    for a in soup.select("table a[href*='player.cgi']"):
        href = a.get("href", "")
        m = re.search(r"[?&]p=([^&]+)", href)
        if m:
            names.append(m.group(1))
    return list(dict.fromkeys(names))   # deduplicate, preserve order


# ---------------------------------------------------------------------------
# Player match page parser
# ---------------------------------------------------------------------------

def _level_from_str(s: str) -> str:
    s_lower = s.lower()
    for k, v in LEVEL_MAP.items():
        if k.lower() in s_lower:
            return v
    return "A"


def _surface_from_str(s: str) -> str:
    return SURFACE_NORM.get(s.lower().strip(), "Hard")


def _parse_date(s: str) -> str:
    """Convert various date strings → YYYYMMDD."""
    s = s.strip()
    # Try YYYY-MM-DD
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        return m.group(1) + m.group(2) + m.group(3)
    # Try Mon DD YYYY
    try:
        from datetime import datetime
        for fmt in ("%b %d %Y", "%B %d, %Y", "%d/%m/%Y", "%m/%d/%Y"):
            try:
                return datetime.strptime(s, fmt).strftime("%Y%m%d")
            except ValueError:
                pass
    except Exception:
        pass
    return re.sub(r"[^\d]", "", s)[:8]


def parse_player_matches(html: str, player_name: str, year: int) -> list[dict]:
    """
    Parse the match-result table from a Tennis Abstract player page.

    The table has approximately these columns (may vary):
      Date | Tournament | Surface | Round | Opponent | Score | Rank
    Each row represents one match from the player's perspective.
    """
    soup  = BeautifulSoup(html, "html.parser")
    rows: list[dict] = []

    # Tennis Abstract uses several table ids / classes – try common ones
    table = (
        soup.find("table", {"id": "maintable"}) or
        soup.find("table", {"id": "matchlog"}) or
        soup.find("table", class_=re.compile(r"sortable|matches|results", re.I)) or
        soup.find("table")
    )

    if not table:
        return []

    headers_el = table.find("tr")
    if not headers_el:
        return []

    raw_headers = [th.get_text(strip=True).lower() for th in headers_el.find_all(["th", "td"])]

    def col(row_cells: list, *names: str) -> str:
        for name in names:
            for i, h in enumerate(raw_headers):
                if name in h and i < len(row_cells):
                    return row_cells[i].get_text(strip=True)
        return ""

    def col_idx(row_cells: list, *names: str) -> str:
        for name in names:
            for i, h in enumerate(raw_headers):
                if name in h:
                    if i < len(row_cells):
                        return row_cells[i].get_text(separator=" ", strip=True)
        return ""

    for tr in table.find_all("tr")[1:]:
        cells = tr.find_all(["td", "th"])
        if len(cells) < 4:
            continue

        date_str  = col(cells, "date")
        if not date_str:
            continue
        date_norm = _parse_date(date_str)
        if not date_norm.startswith(str(year)):
            continue

        tourney_name = col(cells, "tournament", "tourney", "event")
        surface_raw  = col(cells, "surface", "court")
        surface      = _surface_from_str(surface_raw)
        round_name   = col(cells, "round")
        opponent     = col(cells, "opponent", "opp")
        score        = col(cells, "score", "result")
        rank_self    = col(cells, "rank", "atp")
        winner_flag  = col(cells, "w/l", "result", "wl", "won")

        # Determine who won
        won = winner_flag.strip().upper() in ("W", "WIN", "1")
        if won:
            winner_name = player_name
            loser_name  = opponent
        else:
            winner_name = opponent
            loser_name  = player_name

        # Build a synthetic tourney_id similar to Sackmann's format
        slug = re.sub(r"[^\w]", "", tourney_name or "UNKNOWN")
        tourney_id = f"{year}-{slug[:10].upper()}"

        try:
            winner_rank = int(rank_self) if won else None
            loser_rank  = None if won else int(rank_self)
        except (ValueError, TypeError):
            winner_rank = loser_rank = None

        rows.append({
            "tourney_id":          tourney_id,
            "tourney_name":        tourney_name,
            "surface":             surface,
            "draw_size":           None,
            "tourney_level":       _level_from_str(tourney_name or ""),
            "tourney_date":        date_norm,
            "match_num":           None,
            "winner_id":           "",
            "winner_seed":         "",
            "winner_entry":        "",
            "winner_name":         winner_name,
            "winner_hand":         "",
            "winner_ht":           None,
            "winner_ioc":          "",
            "winner_age":          None,
            "loser_id":            "",
            "loser_seed":          "",
            "loser_entry":         "",
            "loser_name":          loser_name,
            "loser_hand":          "",
            "loser_ht":            None,
            "loser_ioc":           "",
            "loser_age":           None,
            "score":               score,
            "best_of":             3,
            "round":               round_name,
            "minutes":             None,
            # Serve stats not available on summary page
            "w_ace": None, "w_df": None, "w_svpt": None,
            "w_1stIn": None, "w_1stWon": None, "w_2ndWon": None,
            "w_SvGms": None, "w_bpSaved": None, "w_bpFaced": None,
            "l_ace": None, "l_df": None, "l_svpt": None,
            "l_1stIn": None, "l_1stWon": None, "l_2ndWon": None,
            "l_SvGms": None, "l_bpSaved": None, "l_bpFaced": None,
            "winner_rank":        winner_rank,
            "winner_rank_points": None,
            "loser_rank":         loser_rank,
            "loser_rank_points":  None,
        })

    return rows


# ---------------------------------------------------------------------------
# Main scrape logic
# ---------------------------------------------------------------------------

def scrape_year(year: int, max_players: int = 200) -> pd.DataFrame:
    # 1. Build player list
    player_names: list[str] = []
    if ATP_DIR.exists():
        player_names = load_player_names_from_csv(ATP_DIR)
        print(f"[TA] Loaded {len(player_names)} names from atp_players.csv")
    if not player_names:
        player_names = fetch_ranked_player_names()
        print(f"[TA] Fetched {len(player_names)} names from leaderboard")

    if max_players:
        player_names = player_names[:max_players]

    print(f"[TA] Scraping {len(player_names)} players for {year}…")

    seen_matches: set[str] = set()   # deduplicate (match appears for both players)
    all_rows: list[dict] = []

    for i, name in enumerate(player_names, 1):
        print(f"  [{i}/{len(player_names)}] {name}", end="", flush=True)
        url    = PLAYER_URL
        params = {"p": name, "year": year}
        key    = f"ta_{name}_{year}"

        html = fetch_html(url, params=params, cache_key=key)
        if not html:
            print(" — skipped")
            continue

        matches = parse_player_matches(html, name, year)
        added = 0
        for m in matches:
            # Dedup key: opponents sorted + date
            side_a = min(m["winner_name"], m["loser_name"])
            side_b = max(m["winner_name"], m["loser_name"])
            dup_key = f"{m['tourney_date']}|{side_a}|{side_b}|{m['round']}"
            if dup_key in seen_matches:
                continue
            seen_matches.add(dup_key)
            all_rows.append(m)
            added += 1

        print(f" — {added} new matches")

    df = pd.DataFrame(all_rows, columns=SACKMANN_COLS)
    df = df.sort_values("tourney_date").reset_index(drop=True)
    # Assign sequential match numbers per tournament
    df["match_num"] = df.groupby("tourney_id").cumcount() + 1
    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape Tennis Abstract match results")
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--max-players", type=int, default=200,
                        help="How many players to iterate (sorted by ATP file order)")
    parser.add_argument("--clear-cache", action="store_true",
                        help="Delete cached HTML before running")
    args = parser.parse_args()

    if args.clear_cache:
        for f in CACHE_DIR.glob("*.html"):
            f.unlink()
        print(f"[TA] Cache cleared ({CACHE_DIR})")

    df = scrape_year(args.year, max_players=args.max_players)

    if df.empty:
        print("[TA] No data scraped.")
        return

    out_path = OUT_DIR / f"ta_results_{args.year}.csv"
    df.to_csv(out_path, index=False)
    print(f"\n[TA] Saved {len(df)} matches → {out_path}")


if __name__ == "__main__":
    main()
