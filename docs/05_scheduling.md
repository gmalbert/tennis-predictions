# Live Scheduling & Match Data

## Overview

The app fetches **today's matches** from two sources:
1. **Flashscore** (primary) — comprehensive coverage: ATP, WTA, Challengers, ITF, doubles.
2. **ESPN** (fallback) — ATP + WTA main tour only.

Results are cached for 2 minutes to avoid excessive requests and sorted by status: `live > starting_soon > upcoming > scheduled`.

---

## 1. Data Models

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class TennisMatch:
    """A scheduled or in-progress match."""
    player1_name: str
    player2_name: str
    tournament: str
    surface: str
    round: str
    start_time: datetime
    status: str         # "live", "starting_soon", "upcoming", "scheduled"
    score: Optional[str] = None


@dataclass
class CompletedMatch:
    """A finished match with result."""
    player1_name: str
    player2_name: str
    winner_name: str
    tournament: str
    surface: str
    score: str
    start_time: datetime
```

### Status Determination

```python
from datetime import timezone

STATUS_PRIORITY = {"live": 0, "starting_soon": 1, "upcoming": 2, "scheduled": 3}


def determine_status(start_time: datetime, is_live: bool) -> str:
    if is_live:
        return "live"
    now = datetime.now(timezone.utc)
    diff = (start_time - now).total_seconds()
    if diff <= 0:
        return "live"
    if diff <= 1800:        # 30 minutes
        return "starting_soon"
    if diff <= 43200:       # 12 hours
        return "upcoming"
    return "scheduled"
```

---

## 2. Flashscore Parser

Flashscore's live-feed uses a custom delimited text format. Each section is separated by `~`, fields within a section by `¬` (0xAC), and key-value pairs by `÷` (0xF7).

### Key Field Codes

| Code | Meaning | Example |
|------|---------|---------|
| `ZA` | Tournament header | `ATP - SINGLES: Dallas (USA), hard (indoor)` |
| `AA` | Match ID | `bRk3Hq27` |
| `AB` | Status code | `0`=not started, `2`=live, `3`=finished |
| `AD` | Start time (unix) | `1708963200` |
| `AE` | Player 1 name | `Sinner J.` |
| `AF` | Player 2 name | `Alcaraz C.` |
| `BA`/`BB` | Set 1 score (P1/P2) | `6`/`4` |
| `BC`/`BD` | Set 2 score | `7`/`5` |
| `BE`/`BF` | Set 3 score | `3`/`6` |

### Full Parser

```python
import requests
import time
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional

# ---- Cache ----
_cache: Dict[str, object] = {}
_cache_time: Dict[str, float] = {}
CACHE_TTL = 120  # seconds


def _get_cached(key: str):
    if key in _cache and (time.time() - _cache_time.get(key, 0)) < CACHE_TTL:
        return _cache[key]
    return None


def _set_cache(key: str, value):
    _cache[key] = value
    _cache_time[key] = time.time()


# ---- Flashscore constants ----
FLASHSCORE_URL = "https://www.flashscore.com/x/feed/f_2_0_2_en-gb_1"
FLASHSCORE_HEADERS = {
    "x-fsign": "SW9D1eZo",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "Referer": "https://www.flashscore.com/tennis/",
}


def _extract_surface(header: str) -> str:
    lower = header.lower()
    if "clay" in lower:
        return "Clay"
    if "grass" in lower:
        return "Grass"
    return "Hard"


def _extract_tournament_name(header: str) -> str:
    # "ATP - SINGLES: Buenos Aires (Argentina), clay" → "Buenos Aires"
    if ": " in header:
        header = header.split(": ", 1)[1]
    if ", " in header:
        header = header.rsplit(", ", 1)[0]
    return header.strip()


def _extract_category(header: str) -> str:
    if ": " in header:
        return header.split(": ", 1)[0].strip()
    return ""


def _fetch_flashscore_raw() -> str:
    cached = _get_cached("flashscore_raw")
    if cached is not None:
        return cached
    try:
        resp = requests.get(FLASHSCORE_URL, headers=FLASHSCORE_HEADERS, timeout=15)
        resp.raise_for_status()
        raw = resp.text
    except (requests.RequestException, ValueError):
        return ""
    _set_cache("flashscore_raw", raw)
    return raw


def fetch_flashscore_matches() -> List[TennisMatch]:
    """Parse Flashscore feed into TennisMatch objects."""
    cached = _get_cached("flashscore_all")
    if cached is not None:
        return cached

    raw = _fetch_flashscore_raw()
    if not raw:
        return []

    matches: List[TennisMatch] = []
    current_tournament = ""
    current_surface = "Hard"
    current_category = ""

    for section in raw.split("~"):
        if not section:
            continue

        fields = {}
        for part in section.split("\xac"):          # ¬
            if "\xf7" in part:                       # ÷
                k, v = part.split("\xf7", 1)
                fields[k] = v

        # Tournament header
        if "ZA" in fields:
            current_tournament = _extract_tournament_name(fields["ZA"])
            current_surface = _extract_surface(fields["ZA"])
            current_category = _extract_category(fields["ZA"])
            continue

        if "AA" not in fields:
            continue

        status_code = fields.get("AB", "0")
        # Skip finished / cancelled / walkovers
        if status_code in ("3", "5", "9", "10", "12", "17"):
            continue

        p1 = fields.get("AE", "").strip()
        p2 = fields.get("AF", "").strip()
        if not p1 or not p2:
            continue

        ts_str = fields.get("AD", "0")
        try:
            start_time = datetime.fromtimestamp(int(ts_str), tz=timezone.utc)
        except (ValueError, OSError):
            start_time = datetime.now(timezone.utc)

        is_live = status_code == "2"
        status = determine_status(start_time, is_live)

        # Live score
        score = None
        if is_live:
            sets = []
            for k1, k2 in [("BA","BB"), ("BC","BD"), ("BE","BF"), ("BG","BH"), ("BI","BJ")]:
                s1, s2 = fields.get(k1, ""), fields.get(k2, "")
                if s1 and s2:
                    sets.append(f"{s1}-{s2}")
            if sets:
                score = " ".join(sets)

        display = current_tournament
        if current_category:
            display = f"{current_tournament} ({current_category})"

        matches.append(TennisMatch(
            player1_name=p1,
            player2_name=p2,
            tournament=display,
            surface=current_surface,
            round="",
            start_time=start_time,
            status=status,
            score=score,
        ))

    _set_cache("flashscore_all", matches)
    return matches
```

---

## 3. ESPN Fallback

```python
ESPN_ENDPOINTS = {
    "ATP": "https://site.api.espn.com/apis/site/v2/sports/tennis/atp/scoreboard",
    "WTA": "https://site.api.espn.com/apis/site/v2/sports/tennis/wta/scoreboard",
}


def fetch_espn_matches() -> List[TennisMatch]:
    matches: List[TennisMatch] = []

    for tour, url in ESPN_ENDPOINTS.items():
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError):
            continue

        for event in data.get("events", []):
            tournament_name = event.get("name", "")
            surface = _extract_surface(tournament_name)

            comps = []
            for grp in event.get("groupings", []):
                comps.extend(grp.get("competitions", []))
            if not comps:
                comps = event.get("competitions", [])

            for comp in comps:
                state = comp.get("status", {}).get("type", {}).get("state", "pre")
                if state == "post":
                    continue

                competitors = comp.get("competitors", [])
                if len(competitors) < 2:
                    continue

                p1 = competitors[0].get("athlete", {}).get("displayName", "")
                p2 = competitors[1].get("athlete", {}).get("displayName", "")
                if not p1 or not p2 or "TBD" in (p1, p2):
                    continue

                date_str = comp.get("startDate", comp.get("date", event.get("date", "")))
                try:
                    start_time = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                except (ValueError, AttributeError):
                    start_time = datetime.now(timezone.utc)

                is_live = state == "in"
                status = determine_status(start_time, is_live)

                matches.append(TennisMatch(
                    player1_name=p1,
                    player2_name=p2,
                    tournament=tournament_name,
                    surface=surface,
                    round="",
                    start_time=start_time,
                    status=status,
                    score=None,
                ))

    return matches
```

---

## 4. Combined Entry Point

```python
def get_todays_matches() -> List[TennisMatch]:
    """
    Primary: Flashscore. Fallback: ESPN.
    Returns sorted by status priority.
    """
    cached = _get_cached("all_matches")
    if cached is not None:
        return cached

    matches = fetch_flashscore_matches()
    if not matches:
        matches = fetch_espn_matches()

    matches.sort(key=lambda m: (STATUS_PRIORITY.get(m.status, 9), m.start_time))

    _set_cache("all_matches", matches)
    return matches
```

---

## 5. Completed Match Tracker

For the prediction backlog page, parse finished matches (Flashscore status `3`) and determine the winner from set scores.

```python
def get_completed_matches() -> List[CompletedMatch]:
    """Fetch today's finished matches from Flashscore."""
    cached = _get_cached("completed_matches")
    if cached is not None:
        return cached

    raw = _fetch_flashscore_raw()
    if not raw:
        return []

    completed: List[CompletedMatch] = []
    current_tournament = ""
    current_surface = "Hard"
    current_category = ""

    for section in raw.split("~"):
        if not section:
            continue

        fields = {}
        for part in section.split("\xac"):
            if "\xf7" in part:
                k, v = part.split("\xf7", 1)
                fields[k] = v

        if "ZA" in fields:
            current_tournament = _extract_tournament_name(fields["ZA"])
            current_surface = _extract_surface(fields["ZA"])
            current_category = _extract_category(fields["ZA"])
            continue

        if "AA" not in fields or fields.get("AB") != "3":
            continue

        p1 = fields.get("AE", "").strip()
        p2 = fields.get("AF", "").strip()
        if not p1 or not p2:
            continue

        ts_str = fields.get("AD", "0")
        try:
            start_time = datetime.fromtimestamp(int(ts_str), tz=timezone.utc)
        except (ValueError, OSError):
            start_time = datetime.now(timezone.utc)

        # Determine winner from set scores
        set_keys = [("BA","BB"),("BC","BD"),("BE","BF"),("BG","BH"),("BI","BJ")]
        sets, p1_won, p2_won = [], 0, 0
        for k1, k2 in set_keys:
            s1, s2 = fields.get(k1, ""), fields.get(k2, "")
            if s1 and s2:
                sets.append(f"{s1}-{s2}")
                try:
                    if int(s1) > int(s2): p1_won += 1
                    elif int(s2) > int(s1): p2_won += 1
                except ValueError:
                    pass

        if not sets:
            continue

        winner = p1 if p1_won > p2_won else p2

        display = current_tournament
        if current_category:
            display = f"{current_tournament} ({current_category})"

        completed.append(CompletedMatch(
            player1_name=p1,
            player2_name=p2,
            winner_name=winner,
            tournament=display,
            surface=current_surface,
            score=" ".join(sets),
            start_time=start_time,
        ))

    completed.sort(key=lambda m: m.start_time, reverse=True)
    _set_cache("completed_matches", completed)
    return completed
```

---

## 6. Player Name Matching

Flashscore uses `"LastName F."` format while our DB uses `"First Last"`. This module handles fuzzy matching.

```python
import re
from typing import Dict, Optional

# Pre-built lookup indices
_lookup_norm: Dict[str, int] = {}
_lookup_last: Dict[str, Optional[int]] = {}
_lookup_norms_by_pid: Dict[int, str] = {}
_lookup_built_for: Optional[int] = None


def _normalize(name: str) -> str:
    name = re.sub(r"\s+[A-Z]\.\s*$", "", name)       # strip trailing initial
    return re.sub(r"[^a-z ]", "", name.lower()).strip()


def _last_name(name: str) -> str:
    clean = re.sub(r"\s+[A-Z]\.\s*$", "", name).strip()
    parts = clean.split()
    return max(parts, key=len).lower() if parts else ""


def ensure_lookups(player_names: Dict[int, str]):
    """Build O(1) lookup dicts. Only rebuilds if dict identity changes."""
    global _lookup_norm, _lookup_last, _lookup_norms_by_pid, _lookup_built_for

    if _lookup_built_for == id(player_names):
        return

    norm_map, last_counts, norms_by_pid = {}, {}, {}
    for pid, db_name in player_names.items():
        if not isinstance(db_name, str):
            continue
        normed = _normalize(db_name)
        if normed:
            norm_map[normed] = pid
            norms_by_pid[pid] = normed
        last = _last_name(db_name)
        if last and len(last) > 2:
            last_counts.setdefault(last, []).append(pid)

    _lookup_norm = norm_map
    _lookup_last = {k: v[0] if len(v) == 1 else None for k, v in last_counts.items()}
    _lookup_norms_by_pid = norms_by_pid
    _lookup_built_for = id(player_names)


def match_player_to_database(api_name: str, player_names: Dict[int, str]) -> Optional[int]:
    """
    Match a schedule API name to a player_id in our database.

    Strategy:
    1. Exact normalized match (O(1))
    2. Unique last-name match (O(1))
    3. Substring fallback (O(n))
    """
    if not api_name or "/" in api_name:
        return None

    api_norm = _normalize(api_name)
    api_last = _last_name(api_name)
    if not api_norm:
        return None

    ensure_lookups(player_names)

    # 1. Exact match
    pid = _lookup_norm.get(api_norm)
    if pid is not None:
        return pid

    # 2. Last-name match (unique only)
    if api_last and len(api_last) > 2:
        pid = _lookup_last.get(api_last)
        if pid is not None:
            return pid

    # 3. Substring fallback
    for pid, db_norm in _lookup_norms_by_pid.items():
        if len(api_norm) > 3 and (api_norm in db_norm or db_norm in api_norm):
            return pid

    return None
```

---

## 7. Streamlit Caching

In the main app, wrap the fetch functions with Streamlit's TTL cache:

```python
import streamlit as st

@st.cache_data(ttl=120)
def cached_todays_matches():
    return get_todays_matches()

@st.cache_data(ttl=120)
def cached_completed_matches():
    return get_completed_matches()
```
