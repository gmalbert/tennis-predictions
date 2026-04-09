# Bzzoiro Tennis API — Integration Analysis & Implementation

> Status: Completed. Bzzoiro integration guidance matches the existing `bzzoiro_api.py` client in the repo.
>
## Overview

[Bzzoiro Sports Data](https://tennis.bzzoiro.com/) provides a **free, unlimited** REST API
for tennis data. The `BZZOIRO_KEY` is already in `.env`. This document covers:

1. What the API offers  
2. How it compares with and supplements the existing stack  
3. Specific gaps it fills  
4. Recommended features to build  
5. Complete implementation code (`bzzoiro_api.py`)

---

## 1. Endpoint Reference

| Endpoint | Cache TTL | Key Notes |
|---|---|---|
| `GET /api/tournaments/` | 5 min | Filter by `circuit` (ATP/WTA) and `category` |
| `GET /api/players/` | 5 min | 4,700+ players; filter by gender, country, name search |
| `GET /api/matches/` | 2 min | Date range, tournament, player, status (`scheduled`/`live`/`finished`) |
| `GET /api/live/` | **30 sec** | Live scores with current set, game score, serving player |
| `GET /api/predictions/` | 2 min | ML win probs, confidence, expected sets/games, over/under games |
| `GET /api/rankings/` | 5 min | ATP/WTA rankings; historical by date; limit with `?top=N` |

**Base URL:** `https://tennis.bzzoiro.com/api/`  
**Auth:** `Authorization: Token BZZOIRO_KEY` header  
**Pagination:** 50 results/page; follow `next` URL or use `?page=N`  
**Price:** 100% free, unlimited requests, no credit card

---

## 2. Current API Stack vs. Bzzoiro

| Capability | Current Stack | Bzzoiro | Verdict |
|---|---|---|---|
| Today's ATP schedule | Matchstat RapidAPI (500 calls/month) | `/api/matches/?status=scheduled` (unlimited) | **Replace with Bzzoiro as primary** |
| Match odds | Matchstat RapidAPI, The Odds API | `/api/matches/` includes `odds_player1/2` | Bzzoiro as fallback when Matchstat is down |
| Live scores (game level) | Not available | `/api/live/` — game score, serving player, 30 sec TTL | **New capability** |
| WTA matches | Not implemented | `/api/matches/` covers ATP + WTA | **New capability** |
| External ML predictions | None | `/api/predictions/` — win %, confidence, expected games | **New capability** (calibration) |
| Current rankings | Only our ELO | `/api/rankings/?type=ATP&top=200` | **Supplement** |
| Historical rankings | Not available | `/api/rankings/?date=YYYY-MM-DD` | **New capability** |
| Player profiles (searchable) | Sackmann `atp_players.csv` (static) | `/api/players/?search=...` (live) | **Supplement** |
| Tournament catalog | None | `/api/tournaments/` | **New capability** |
| Match stats (aces, DFs) | Sackmann files (historical only) | `/api/matches/` per-match serve stats | Limited coverage, but useful for recent |

---

## 3. Gaps Filled

### 3a. Live scores — currently missing entirely
`/api/live/` returns all currently live matches with:
- Current set number, current game score (`"40-30"`), tiebreak score  
- Serving player indicator (1 or 2)  
- Running set scores  
- 30-second server cache — safe to poll

This enables a **real-time live scores ticker** in the Today's Matches tab. The existing tab
shows all matches as "upcoming" because Matchstat only returns upcoming fixtures.

### 3b. WTA coverage
Nothing in the current stack tracks WTA. Bzzoiro's `matches`, `live`, `rankings`, and
`predictions` endpoints all cover WTA. The `circuit` query param on tournaments lets you
filter `WTA` specifically.

### 3c. Budget pressure on Matchstat (500 calls/month)
With Bzzoiro unlimited, the smart strategy is:
- Use **Bzzoiro as the primary schedule/fixture source** (unlimited)
- Reserve Matchstat calls for its H2H endpoint (not available in Bzzoiro)
- Fall back to Matchstat odds only when Bzzoiro odds are absent

### 3d. External prediction as a cross-check
`/api/predictions/` returns Bzzoiro's own ML confidence, win probabilities, expected total
sets and games, and over/under lines. This is useful for:
- Comparing our ELO-based model against an independent signal  
- Displaying a "consensus" probability (average of our model + Bzzoiro)  
- Identifying high-divergence matches (our model disagrees strongly with Bzzoiro → flag for review)

### 3e. Live ATP/WTA rankings
`/api/rankings/?type=ATP&top=200` gives the current official ATP ranking (points + rank)
which can replace or verify `atp_rankings_current.csv` from the Sackmann repo (updated only
when git pull runs).

---

## 4. Recommended Features

| Priority | Feature | Bzzoiro Endpoint | Notes |
|---|---|---|---|
| 🔴 High | Live score ticker in Today's Matches tab | `/api/live/` | Real-time; 30 s TTL; new UI |
| 🔴 High | Replace Matchstat as schedule source | `/api/matches/?status=scheduled` | Eliminates 500/month budget risk |
| 🟠 Medium | WTA tab in Streamlit app | `/api/matches/`, `/api/live/` | Parallel to existing ATP view |
| 🟠 Medium | Bzzoiro predictions overlay | `/api/predictions/` | Second-opinion panel per match |
| 🟠 Medium | Live ATP rankings sidebar / ELO Rankings tab | `/api/rankings/` | Replace static CSV with live data |
| 🟡 Low | Player search in Player Analysis tab | `/api/players/?search=` | Live lookup for players not in parquet |
| 🟡 Low | Tournament browser | `/api/tournaments/` | Navigate by surface/category |
| 🟡 Low | Historical ranking lookup | `/api/rankings/?date=` | Feature: rank at tournament date |

---

## 5. Implementation — `bzzoiro_api.py`

Drop this file into the project root alongside `matchstat_api.py`.

```python
"""
bzzoiro_api.py
--------------
Client for the Bzzoiro Tennis API (https://tennis.bzzoiro.com/api/).

Free, unlimited, no rate limits.
Auth: Authorization: Token <BZZOIRO_KEY>

Key public functions
--------------------
  get_scheduled_matches(date_str, circuit)  → list[dict]
  get_live_matches()                        → list[dict]
  get_predictions(date_from, date_to)       → list[dict]
  get_rankings(circuit, top)                → list[dict]
  get_players(search)                       → list[dict]
  get_tournaments(circuit, category)        → list[dict]

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
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterator

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
    Return scheduled (and recently finished) matches for a date or date range.

    Parameters
    ----------
    date_str   : YYYY-MM-DD (default: today)
    circuit    : "ATP" or "WTA"
    days_ahead : include this many extra days beyond date_str (0 = single day)

    Each match dict includes player names/rankings, tournament, surface, round,
    score, and odds_player1 / odds_player2 where available.
    """
    d_from = date_str or date.today().isoformat()
    d_to   = (
        date.fromisoformat(d_from) + timedelta(days=days_ahead)
    ).isoformat()

    cache_key = f"matches_{circuit}_{d_from}_{d_to}.json"
    cached    = _load_cached(_cache_path(cache_key), _TTL["matches"])
    if cached is not None:
        return cached

    params = {
        "date_from": d_from,
        "date_to":   d_to,
        "status":    "scheduled",
    }
    results = _get_all_pages("matches", params)
    # Filter by circuit via tournament.circuit field
    filtered = [
        m for m in results
        if (m.get("tournament") or {}).get("circuit", "").upper() == circuit.upper()
    ]
    _save_cache(_cache_path(cache_key), filtered)
    return filtered


def get_live_matches(circuit: str | None = None) -> list[dict]:
    """
    Return all currently live matches (no disk cache — always fresh).

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
        match           – nested match object (player1/2 names, tournament, surface)
        prob_player1_wins / prob_player2_wins  – win % (0–100)
        predicted_winner  – 1 or 2
        confidence        – model confidence (0–100)
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

    Returns list of dicts: {ranking, points, player: {id, name, country, ...}}
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


# ── Convenience helpers for predictions.py ───────────────────────────────────

def get_today_matches_with_predictions(circuit: str = "ATP") -> list[dict]:
    """
    Merge today's scheduled matches with Bzzoiro predictions.

    Returns a flat list of dicts ready for the Streamlit UI:
        player1_name, player2_name, tournament, surface, round,
        odds_p1, odds_p2,
        bzz_prob_p1, bzz_prob_p2, bzz_confidence,
        bzz_expected_sets, bzz_expected_games, bzz_prob_over_22_5
    """
    today = date.today().isoformat()
    matches     = get_scheduled_matches(today, circuit=circuit)
    predictions = get_predictions(date_from=today, date_to=today)

    # Build a lookup: match_id → prediction
    pred_by_match: dict[int, dict] = {}
    for p in predictions:
        m = p.get("match") or {}
        if m.get("id"):
            pred_by_match[m["id"]] = p

    results = []
    for m in matches:
        # Skip doubles (player names contain '/')
        p1_name = (m.get("player1_obj") or m.get("player1") or {})
        p2_name = (m.get("player2_obj") or m.get("player2") or {})
        p1_str  = p1_name.get("name", "") if isinstance(p1_name, dict) else str(p1_name)
        p2_str  = p2_name.get("name", "") if isinstance(p2_name, dict) else str(p2_name)
        if "/" in p1_str or "/" in p2_str:
            continue

        tournament = m.get("tournament") or {}
        pred       = pred_by_match.get(m.get("id"))

        row = {
            "match_id":          m.get("id"),
            "date":              m.get("date"),
            "tournament":        tournament.get("name"),
            "surface":           tournament.get("surface"),
            "round":             (m.get("round") or {}).get("name") if isinstance(m.get("round"), dict) else m.get("round"),
            "player1_name":      p1_str,
            "player1_ranking":   p1_name.get("ranking") if isinstance(p1_name, dict) else None,
            "player2_name":      p2_str,
            "player2_ranking":   p2_name.get("ranking") if isinstance(p2_name, dict) else None,
            "odds_p1":           m.get("odds_player1"),
            "odds_p2":           m.get("odds_player2"),
        }

        if pred:
            row.update({
                "bzz_prob_p1":          pred.get("prob_player1_wins"),
                "bzz_prob_p2":          pred.get("prob_player2_wins"),
                "bzz_confidence":       pred.get("confidence"),
                "bzz_predicted_winner": pred.get("predicted_winner"),
                "bzz_expected_sets":    pred.get("expected_total_sets"),
                "bzz_expected_games":   pred.get("expected_total_games"),
                "bzz_prob_over_22_5":   pred.get("prob_over_22_5_games"),
            })

        results.append(row)

    return results
```

---

## 6. Streamlit Integration Snippets

### 6a. Live score ticker (add to Today's Matches tab in `predictions.py`)

```python
# ── Live score ticker ─────────────────────────────────────────────────────────
@st.cache_data(ttl=30)   # matches server TTL
def load_live_scores() -> list[dict]:
    try:
        from bzzoiro_api import get_live_matches
        return get_live_matches(circuit="ATP")
    except Exception as e:
        return []

live_scores = load_live_scores()

if live_scores:
    st.markdown("### 🔴 Live Now")
    for m in live_scores:
        p1     = (m.get("player1_obj") or {}).get("name", m.get("player1", "P1"))
        p2     = (m.get("player2_obj") or {}).get("name", m.get("player2", "P2"))
        s1     = m.get("player1_sets", 0)
        s2     = m.get("player2_sets", 0)
        game   = m.get("current_game_score", "")
        server = m.get("serving_player")  # 1 or 2
        srv_icon = "🎾" 
        label_p1 = f"{srv_icon} {p1}" if server == 1 else p1
        label_p2 = f"{srv_icon} {p2}" if server == 2 else p2
        st.markdown(
            f'<div class="match-card">'
            f'<span><strong>{label_p1}</strong> {s1} – {s2} <strong>{label_p2}</strong></span>'
            f'<span style="color:#6b7280">{game}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
```

### 6b. Bzzoiro predictions as second opinion (add to Today's Matches match cards)

```python
# Inside the per-match card rendering loop, after computing p1_prob:
from bzzoiro_api import get_predictions
from datetime import date as _date

@st.cache_data(ttl=120)
def _bzz_preds_today():
    try:
        from bzzoiro_api import get_predictions
        today = _date.today().isoformat()
        return {
            (p["match"]["player1"], p["match"]["player2"]): p
            for p in get_predictions(date_from=today, date_to=today)
            if p.get("match")
        }
    except Exception:
        return {}

bzz_preds = _bzz_preds_today()

# Then per match card:
bzz = bzz_preds.get((p1, p2)) or bzz_preds.get((p2, p1))
if bzz:
    bzz_p1_pct = bzz.get("prob_player1_wins", 0)
    bzz_conf   = bzz.get("confidence", 0)
    st.caption(f"Bzzoiro model: {p1} {bzz_p1_pct:.0f}% · confidence {bzz_conf:.0f}%")
```

### 6c. Live rankings tab (add to or replace ELO Rankings tab)

```python
@st.cache_data(ttl=300)
def load_atp_rankings(top: int = 100) -> pd.DataFrame:
    try:
        from bzzoiro_api import get_rankings
        rows = get_rankings(circuit="ATP", top=top)
        return pd.DataFrame([
            {
                "Rank":    r["ranking"],
                "Player":  r["player"]["name"],
                "Country": r["player"].get("country", ""),
                "Points":  r["points"],
            }
            for r in rows
        ])
    except Exception:
        return pd.DataFrame()

with tab_elo:
    st.subheader("ATP Rankings (Live)")
    rank_df = load_atp_rankings(top=200)
    if not rank_df.empty:
        st.dataframe(rank_df, hide_index=True, use_container_width=True)
```

### 6d. Replace Matchstat as the schedule source (drop-in replacement in `predictions.py`)

```python
@st.cache_data(ttl=120)
def load_today_matches_bzzoiro() -> list[dict]:
    """
    Primary schedule source — unlimited Bzzoiro API.
    Falls back to Matchstat if Bzzoiro fails or returns nothing.
    """
    try:
        from bzzoiro_api import get_today_matches_with_predictions
        matches = get_today_matches_with_predictions(circuit="ATP")
        if matches:
            return matches
    except Exception as e:
        st.warning(f"Bzzoiro schedule unavailable: {e}")

    # Fallback: try Matchstat
    try:
        from matchstat_api import get_today_odds, has_upcoming_matches
        if not has_upcoming_matches():
            return []
        raw = get_today_odds()
        # Normalise to the same schema as Bzzoiro
        return [
            {**m, "player1_name": m["player1_name"], "player2_name": m["player2_name"]}
            for m in raw
        ]
    except Exception as e:
        st.warning(f"Matchstat schedule also unavailable: {e}")
        return []
```

---

## 7. `.streamlit/secrets.toml` addition

```toml
# Bzzoiro Tennis API (free, unlimited)
BZZOIRO_KEY = "75407d2e4d270d4442e52a5b6ac18259ed396b69"
```

---

## 8. Summary of Recommendations

1. [x] **Create `bzzoiro_api.py`** (code in §5) and commit it.  
2. [ ] **Switch schedule source**: replace `matchstat_api.get_today_odds()` with
   `bzzoiro_api.get_today_matches_with_predictions()` — eliminates the 500-call/month
   budget risk as the primary concern. Keep Matchstat for H2H.  
3. [ ] **Add live scores ticker** (§6a) — the `/api/live/` endpoint is the only source in
   the stack that provides real-time game scores with serving player.  
4. [x] **Show Bzzoiro predictions as a second-opinion overlay** (§6b) alongside our model —
   especially useful for calibration and identifying divergence.  
5. [x] **Add WTA tab** using `get_today_matches_with_predictions(circuit="WTA")`.  
6. [x] **Update ELO Rankings tab** to show live official rankings from `/api/rankings/` (§6c).  
7. [x] **Add `BZZOIRO_KEY` to `.streamlit/secrets.toml`** for Streamlit Cloud deployment (§7).
