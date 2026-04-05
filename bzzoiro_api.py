"""
bzzoiro_api.py
--------------
Client for the Bzzoiro Tennis API (https://tennis.bzzoiro.com/api/).

Free, unlimited, no rate limits.
Auth: Authorization: Token <BZZOIRO_KEY>

Key public functions
--------------------
  get_scheduled_matches(date_str, circuit, days_ahead)  → list[dict]
  get_live_matches(circuit)                             → list[dict]
  get_predictions(date_from, date_to, upcoming_only)    → list[dict]
  get_rankings(circuit, top, as_of)                     → list[dict]
  get_players(search, gender, country)                  → list[dict]
  get_tournaments(circuit, category)                    → list[dict]
  get_today_matches_with_predictions(circuit)           → list[dict]

Caching
-------
  - Scheduled matches: cached per (date, circuit), TTL 2 min
  - Live matches:       no disk cache (TTL 30 s — always fresh)
  - Predictions:        cached per date-range, TTL 2 min
  - Rankings:           cached per (circuit, date), TTL 5 min
  - Players/Tournaments: cached, TTL 5 min
"""

from __future__ import annotations

import json
import os
import time
from datetime import date, timedelta
from pathlib import Path

import requests

# ── Constants ──────────────────────────────────────────────────────────────────
_BASE  = "https://tennis.bzzoiro.com/api"
_CACHE = Path("cache/bzzoiro")

# TTLs in seconds (match the server-side cache so we never serve staler data)
_TTL = {
    "live":        30,
    "matches":    120,
    "predictions":120,
    "rankings":   300,
    "players":    300,
    "tournaments":300,
}


# ── Key loading ────────────────────────────────────────────────────────────────

def _load_key() -> str:
    """Load BZZOIRO_KEY from environment, .env file, or Streamlit secrets."""
    key = os.environ.get("BZZOIRO_KEY", "")
    if key:
        return key

    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("BZZOIRO_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
                if key:
                    return key

    try:
        import streamlit as st
        key = st.secrets.get("BZZOIRO_KEY", "")
        if key:
            return key
    except Exception:
        pass

    raise RuntimeError(
        "BZZOIRO_KEY not found. Add it to .env or .streamlit/secrets.toml."
    )


# ── Cache helpers ──────────────────────────────────────────────────────────────

def _cache_path(name: str) -> Path:
    _CACHE.mkdir(parents=True, exist_ok=True)
    return _CACHE / name


def _load_cached(path: Path, ttl: int):
    """Return cached data if the file exists and is fresher than `ttl` seconds."""
    if not path.exists():
        return None
    age = time.time() - path.stat().st_mtime
    if age > ttl:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _save_cache(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ── Core HTTP ──────────────────────────────────────────────────────────────────

def _get(endpoint: str, params: dict | None = None) -> dict:
    """Single authenticated GET; raises on HTTP errors with friendly messages."""
    headers = {"Authorization": f"Token {_load_key()}"}
    url = f"{_BASE}/{endpoint.lstrip('/')}/"
    resp = requests.get(url, headers=headers, params=params or {}, timeout=15)

    if resp.status_code >= 500:
        raise RuntimeError(
            f"Bzzoiro API temporarily unavailable (HTTP {resp.status_code}). "
            "Try again in a few minutes."
        )
    if resp.status_code in (401, 403):
        raise RuntimeError(
            f"Bzzoiro API authentication failed (HTTP {resp.status_code}). "
            "Check that BZZOIRO_KEY is valid."
        )
    resp.raise_for_status()
    return resp.json()


def _get_all_pages(endpoint: str, params: dict | None = None) -> list:
    """Fetch all pages of a paginated endpoint and return a flat results list."""
    params = dict(params or {})
    results: list = []
    page = 1
    while True:
        params["page"] = page
        data = _get(endpoint, params)
        results.extend(data.get("results", []))
        if not data.get("next"):
            break
        page += 1
    return results


# ── Public API ─────────────────────────────────────────────────────────────────

def get_scheduled_matches(
    date_str: str | None = None,
    circuit: str = "ATP",
    days_ahead: int = 0,
) -> list[dict]:
    """
    Return scheduled matches for a date or date range.

    Parameters
    ----------
    date_str   : YYYY-MM-DD (default: today)
    circuit    : "ATP" or "WTA"
    days_ahead : include this many extra days beyond date_str (0 = single day)
    """
    d_from = date_str or date.today().isoformat()
    d_to   = (date.fromisoformat(d_from) + timedelta(days=days_ahead)).isoformat()

    cache_key = f"matches_{circuit}_{d_from}_{d_to}.json"
    cached    = _load_cached(_cache_path(cache_key), _TTL["matches"])
    if cached is not None:
        return cached

    params = {"date_from": d_from, "date_to": d_to, "status": "scheduled"}
    results = _get_all_pages("matches", params)
    # Filter by circuit via nested tournament.circuit field
    filtered = [
        m for m in results
        if (m.get("tournament") or {}).get("circuit", "").upper() == circuit.upper()
    ]
    _save_cache(_cache_path(cache_key), filtered)
    return filtered


def get_live_matches(circuit: str | None = None) -> list[dict]:
    """
    Return all currently live matches (no disk cache — always fresh, 30 s TTL).

    Each match includes current_set, current_game_score, serving_player,
    sets_detail, and running set scores.
    """
    results = _get_all_pages("live")
    if circuit:
        results = [
            m for m in results
            if (m.get("tournament") or {}).get("circuit", "").upper() == circuit.upper()
        ]
    return results


def get_predictions(
    date_from: str | None = None,
    date_to: str | None = None,
    upcoming_only: bool = True,
) -> list[dict]:
    """
    Return Bzzoiro ML predictions.

    Each item includes:
        match                              – nested match object
        prob_player1_wins / prob_player2_wins  – win % (0–100)
        predicted_winner                   – 1 or 2
        confidence                         – model confidence (0–100)
        expected_total_sets
        expected_total_games
        prob_over_22_5_games
    """
    d_from = date_from or date.today().isoformat()
    d_to   = date_to   or date.today().isoformat()

    cache_key = f"predictions_{d_from}_{d_to}.json"
    cached    = _load_cached(_cache_path(cache_key), _TTL["predictions"])
    if cached is not None:
        return cached

    params: dict = {"upcoming": str(upcoming_only).lower()}
    if date_from:
        params["date_from"] = date_from
    if date_to:
        params["date_to"] = date_to

    results = _get_all_pages("predictions", params)
    _save_cache(_cache_path(cache_key), results)
    return results


def get_rankings(circuit: str = "ATP", top: int = 200, as_of: str | None = None) -> list[dict]:
    """
    Return ATP or WTA rankings.

    Parameters
    ----------
    circuit : "ATP" or "WTA"
    top     : limit to top N players (max 500)
    as_of   : historical snapshot date (YYYY-MM-DD). Omit for latest.

    Returns list of {ranking, points, player: {id, name, country, ...}}
    """
    cache_key = f"rankings_{circuit}_{as_of or 'latest'}_{top}.json"
    cached    = _load_cached(_cache_path(cache_key), _TTL["rankings"])
    if cached is not None:
        return cached

    params: dict = {"type": circuit, "top": top}
    if as_of:
        params["date"] = as_of

    results = _get_all_pages("rankings", params)
    _save_cache(_cache_path(cache_key), results)
    return results


def get_players(
    search: str | None = None,
    gender: str | None = None,
    country: str | None = None,
) -> list[dict]:
    """
    Search the 4,700+ player database.

    Parameters
    ----------
    search  : name substring (e.g. "Djokovic")
    gender  : "M" or "F"
    country : ISO country code (e.g. "ES", "US")
    """
    cache_key = f"players_{search or ''}_{gender or ''}_{country or ''}.json"
    cached    = _load_cached(_cache_path(cache_key), _TTL["players"])
    if cached is not None:
        return cached

    params: dict = {}
    if search:
        params["search"] = search
    if gender:
        params["gender"] = gender
    if country:
        params["country"] = country

    results = _get_all_pages("players", params)
    _save_cache(_cache_path(cache_key), results)
    return results


def get_tournaments(circuit: str | None = None, category: str | None = None) -> list[dict]:
    """
    Return tournament catalog.

    Parameters
    ----------
    circuit  : "ATP" or "WTA"
    category : "grand_slam", "masters_1000", "atp_500", "atp_250", "wta_1000", etc.
    """
    cache_key = f"tournaments_{circuit or 'all'}_{category or 'all'}.json"
    cached    = _load_cached(_cache_path(cache_key), _TTL["tournaments"])
    if cached is not None:
        return cached

    params: dict = {}
    if circuit:
        params["circuit"] = circuit
    if category:
        params["category"] = category

    results = _get_all_pages("tournaments", params)
    _save_cache(_cache_path(cache_key), results)
    return results


# ── Convenience helper for predictions.py ─────────────────────────────────────

def get_today_matches_with_predictions(circuit: str = "ATP") -> list[dict]:
    """
    Merge today's scheduled matches with Bzzoiro ML predictions.

    Returns a flat list of dicts with keys compatible with the existing
    Matchstat schema so both sources can feed the same Streamlit rendering:
        player1_name, player2_name, tournament, surface, round,
        odds_p1, odds_p2,
        bzz_prob_p1, bzz_prob_p2, bzz_confidence,
        bzz_predicted_winner, bzz_expected_sets,
        bzz_expected_games, bzz_prob_over_22_5

    Doubles matches (names containing '/') are excluded.
    """
    today       = date.today().isoformat()
    matches     = get_scheduled_matches(today, circuit=circuit)
    predictions = get_predictions(date_from=today, date_to=today)

    # Build lookup: match_id → prediction
    pred_by_match: dict[int, dict] = {}
    for p in predictions:
        m = p.get("match") or {}
        if m.get("id"):
            pred_by_match[m["id"]] = p

    results = []
    for m in matches:
        p1_obj = m.get("player1_obj") or m.get("player1") or {}
        p2_obj = m.get("player2_obj") or m.get("player2") or {}
        p1_str = p1_obj.get("name", "") if isinstance(p1_obj, dict) else str(p1_obj)
        p2_str = p2_obj.get("name", "") if isinstance(p2_obj, dict) else str(p2_obj)

        # Skip doubles
        if "/" in p1_str or "/" in p2_str:
            continue

        tournament = m.get("tournament") or {}
        pred       = pred_by_match.get(m.get("id"))

        row: dict = {
            "match_id":       m.get("id"),
            "date":           m.get("match_date"),   # API field is match_date
            "tournament":     tournament.get("name") if isinstance(tournament, dict) else str(tournament),
            "surface":        tournament.get("surface") if isinstance(tournament, dict) else None,
            "round":          m.get("round_name") or "",  # API field is round_name (flat string)
            "player1_name":   p1_str,
            "player1_ranking":p1_obj.get("ranking") if isinstance(p1_obj, dict) else None,
            "player2_name":   p2_str,
            "player2_ranking":p2_obj.get("ranking") if isinstance(p2_obj, dict) else None,
            # Odds keys match Matchstat schema
            "odds_p1":        m.get("odds_player1"),
            "odds_p2":        m.get("odds_player2"),
        }

        if pred:
            row.update({
                "bzz_prob_p1":          pred.get("prob_player1_wins"),   # 0–100
                "bzz_prob_p2":          pred.get("prob_player2_wins"),   # 0–100
                "bzz_confidence":       pred.get("confidence"),           # 0–100
                "bzz_predicted_winner": pred.get("predicted_winner"),
                "bzz_expected_sets":    pred.get("expected_total_sets"),
                "bzz_expected_games":   pred.get("expected_total_games"),
                "bzz_prob_over_22_5":   pred.get("prob_over_22_5_games"),
            })

        results.append(row)

    return results
