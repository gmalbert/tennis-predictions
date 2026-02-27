"""
matchstat_api.py
Wrapper around the Matchstat RapidAPI tennis endpoint.

Budget: 500 requests / month (BASIC tier, free)
Strategy:
  - Fixtures for today + tomorrow are fetched at most once per calendar day (cached to disk).
  - No H2H or player-stat calls are made unless there are real upcoming singles matches.
  - H2H responses are cached permanently per player-pair (historical data only grows).
  - A lightweight usage log (cache/matchstat/_usage.json) tracks monthly call spend.
"""

import json
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import requests

# ── Constants ──────────────────────────────────────────────────────────────────
_HOST    = "tennis-api-atp-wta-itf.p.rapidapi.com"
_BASE    = f"https://{_HOST}/tennis/v2"
_CACHE   = Path("cache/matchstat")
_USAGE   = _CACHE / "_usage.json"
_BUDGET  = 500  # calls / month hard limit

# ── Key loading ────────────────────────────────────────────────────────────────

def _load_key() -> str:
    """Load the RapidAPI key from .env first, then .streamlit/secrets.toml."""
    # 1. Environment variable (set by python-dotenv or the shell)
    key = os.environ.get("RAPIDAPI_KEY", "")
    if key:
        return key

    # 2. .env file next to this script
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("RAPIDAPI_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
                if key:
                    return key

    # 3. Streamlit secrets (only when running inside Streamlit)
    try:
        import streamlit as st
        key = st.secrets.get("RAPIDAPI_KEY", "")
        if key:
            return key
    except Exception:
        pass

    raise RuntimeError(
        "RAPIDAPI_KEY not found. Add it to .env or .streamlit/secrets.toml."
    )


# ── Cache helpers ──────────────────────────────────────────────────────────────

def _cache_path(name: str) -> Path:
    _CACHE.mkdir(parents=True, exist_ok=True)
    return _CACHE / name


def _load_json(path: Path):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return None


def _save_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ── Usage tracking ─────────────────────────────────────────────────────────────

def _usage() -> dict:
    data = _load_json(_USAGE) or {}
    month = date.today().strftime("%Y-%m")
    return {"month": month, "calls": data.get("calls", 0) if data.get("month") == month else 0}


def _increment_usage() -> int:
    u = _usage()
    u["calls"] += 1
    _CACHE.mkdir(parents=True, exist_ok=True)
    _save_json(_USAGE, u)
    return u["calls"]


def calls_used_this_month() -> int:
    """Return how many API calls have been made in the current calendar month."""
    return _usage()["calls"]


def calls_remaining() -> int:
    return max(0, _BUDGET - calls_used_this_month())


# ── Core HTTP call ─────────────────────────────────────────────────────────────

def _get(endpoint: str, params: dict | None = None) -> list | dict:
    """
    Make one authenticated GET request. Raises if budget exhausted or HTTP error.
    Returns parsed JSON body.
    """
    if calls_remaining() == 0:
        raise RuntimeError(f"Monthly API budget of {_BUDGET} calls exhausted.")

    headers = {
        "x-rapidapi-host": _HOST,
        "x-rapidapi-key":  _load_key(),
    }
    url = f"{_BASE}/{endpoint.lstrip('/')}"
    resp = requests.get(url, headers=headers, params=params or {}, timeout=15)
    resp.raise_for_status()
    used = _increment_usage()
    print(f"[matchstat] {resp.status_code} {url}  (calls this month: {used}/{_BUDGET})")
    return resp.json()


# ── Fixtures ───────────────────────────────────────────────────────────────────

_FIXTURE_INCLUDES = "round,tournament,tournament.court,tournament.country,odds"


def get_fixtures(date_str: str, tour: str = "atp", force: bool = False) -> list:
    """
    Fetch all fixtures for a single date (YYYY-MM-DD).
    Result is cached to disk for the calendar day; subsequent calls return the cache.
    Set force=True to bypass cache and re-fetch.
    """
    cache_file = _cache_path(f"fixtures_{tour}_{date_str}.json")

    if not force and cache_file.exists():
        data = _load_json(cache_file)
        if data is not None:
            return data

    raw = _get(f"{tour}/fixtures/{date_str}", params={"include": _FIXTURE_INCLUDES})
    fixtures = raw if isinstance(raw, list) else (
        raw.get("data") or raw.get("fixtures") or raw.get("results") or []
    )
    _save_json(cache_file, fixtures)
    return fixtures


def get_upcoming_fixtures(days_ahead: int = 1, tour: str = "atp", force: bool = False) -> list:
    """
    Return today's + the next `days_ahead` days' fixtures combined.
    Each day is cached independently so only new dates consume API calls.
    """
    today = date.today()
    all_fixtures = []
    for offset in range(days_ahead + 1):          # 0 = today, 1 = tomorrow, …
        d = (today + timedelta(days=offset)).isoformat()
        all_fixtures.extend(get_fixtures(d, tour=tour, force=force))
    return all_fixtures


def _singles_only(fixtures: list) -> list:
    """Filter out doubles matches (player names that contain '/')."""
    return [
        f for f in fixtures
        if "/" not in f.get("player1", {}).get("name", "")
        and "/" not in f.get("player2", {}).get("name", "")
    ]


def has_upcoming_matches(tour: str = "atp", days_ahead: int = 1) -> bool:
    """
    Return True only if there are real singles matches scheduled in the next
    `days_ahead` days. Uses the cached fixture data — no extra API call if
    fixtures for those days are already cached.
    """
    today = date.today()
    for offset in range(days_ahead + 1):
        d = (today + timedelta(days=offset)).isoformat()
        cache_file = _cache_path(f"fixtures_{tour}_{d}.json")
        if cache_file.exists():
            fixtures = _load_json(cache_file) or []
        else:
            # Fetch and cache (costs 1 call per uncached day)
            fixtures = get_fixtures(d, tour=tour)
        if _singles_only(fixtures):
            return True
    return False


# ── Today's odds convenience function ─────────────────────────────────────────

def get_today_odds(tour: str = "atp") -> list[dict]:
    """
    Return a flat list of today's upcoming singles matches with odds.
    Each item is a dict with keys:
        fixture_id, date, tournament, surface, round,
        player1_id, player1_name, player1_country,
        player2_id, player2_name, player2_country,
        odds_p1, odds_p2,              # match-winner decimal odds
        total_games, over_odds, under_odds,  # totals market
        handicap, hcp_p1_odds, hcp_p2_odds  # spread market
    Only matches that have odds data are included.
    """
    today = date.today().isoformat()
    fixtures = get_fixtures(today, tour=tour)
    singles = _singles_only(fixtures)

    results = []
    for f in singles:
        odds = f.get("odds")
        if not odds:
            continue
        tournament = f.get("tournament") or {}
        court      = tournament.get("court") or {}
        round_     = f.get("round") or {}
        p1         = f.get("player1") or {}
        p2         = f.get("player2") or {}

        results.append({
            "fixture_id":      f["id"],
            "date":            f["date"],
            "tournament":      tournament.get("name"),
            "surface":         court.get("name"),
            "round":           round_.get("name"),
            "player1_id":      p1.get("id"),
            "player1_name":    p1.get("name"),
            "player1_country": p1.get("countryAcr"),
            "player2_id":      p2.get("id"),
            "player2_name":    p2.get("name"),
            "player2_country": p2.get("countryAcr"),
            # Match winner
            "odds_p1":         odds.get("k1"),
            "odds_p2":         odds.get("k2"),
            # Totals
            "total_games":     odds.get("total"),
            "over_odds":       odds.get("ktm"),
            "under_odds":      odds.get("ktb"),
            # Handicap (spread)
            "handicap":        odds.get("f1"),        # e.g. +1.5 games for p1
            "hcp_p1_odds":     odds.get("kf1"),
            "hcp_p2_odds":     odds.get("kf2"),
        })
    return results


# ── H2H ───────────────────────────────────────────────────────────────────────

def get_h2h(player1_id: int, player2_id: int) -> list:
    """
    Fetch head-to-head fixture history between two players.
    Cached permanently per ordered pair (lower id first).
    NOTE: Cache is intentionally permanent — call invalidate_h2h_cache()
    after the players next meet to refresh it.
    """
    p_lo, p_hi = sorted([player1_id, player2_id])
    cache_file = _cache_path(f"h2h_{p_lo}_{p_hi}.json")

    cached = _load_json(cache_file)
    if cached is not None:
        return cached

    raw = _get(f"atp/fixtures/h2h/{player1_id}/{player2_id}")
    data = raw if isinstance(raw, list) else (
        raw.get("data") or raw.get("fixtures") or raw.get("results") or []
    )
    _save_json(cache_file, data)
    return data


def invalidate_h2h_cache(player1_id: int, player2_id: int) -> None:
    """Delete the cached H2H file so the next call re-fetches it."""
    p_lo, p_hi = sorted([player1_id, player2_id])
    cache_file = _cache_path(f"h2h_{p_lo}_{p_hi}.json")
    if cache_file.exists():
        cache_file.unlink()
        print(f"[matchstat] H2H cache cleared for {p_lo} vs {p_hi}")


# ── CLI quick-test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Calls remaining this month: {calls_remaining()}/{_BUDGET}\n")

    if not has_upcoming_matches():
        print("No upcoming singles matches found — skipping further API calls.")
    else:
        print("Today's singles matches with odds:")
        for m in get_today_odds():
            p1 = m["player1_name"]
            p2 = m["player2_name"]
            o1 = m["odds_p1"]
            o2 = m["odds_p2"]
            tour = m["tournament"]
            rnd  = m["round"]
            print(f"  {tour} [{rnd}]  {p1} ({o1}) vs {p2} ({o2})")

    print(f"\nCalls used this month: {calls_used_this_month()}/{_BUDGET}")
