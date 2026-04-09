"""
fetch_odds_api.py
-----------------
Daily snapshot of The Odds API (https://api.the-odds-api.com) live tennis odds.

Writes data_files/odds_api_today.json — one record per match, keeping the
single bookmaker with the tightest overround (closest to fair).  The file is
stamped with today's UTC date so the app can detect stale data and skip API
calls when already populated.

Usage:
    python fetch_odds_api.py              # fetch and write
    python fetch_odds_api.py --dry-run    # print, do not write
    python fetch_odds_api.py --verbose    # show per-match detail
"""

import argparse
import json
import os
from datetime import datetime, timezone

import requests

DATA_DIR = "data_files"
OUT_FILE = os.path.join(DATA_DIR, "odds_api_today.json")
BASE_URL = "https://api.the-odds-api.com/v4"

# Tennis sport keys supported by The Odds API.
# Only keys that are *active* (returned by /sports) consume a request.
TENNIS_SPORTS = [
    "tennis_atp_aus_open_singles",
    "tennis_atp_french_open",
    "tennis_atp_wimbledon",
    "tennis_atp_us_open",
    "tennis_atp_indian_wells",
    "tennis_atp_miami_open",
    "tennis_atp_monte_carlo_masters",
    "tennis_atp_madrid_open",
    "tennis_atp_italian_open",
    "tennis_atp_canadian_open",
    "tennis_atp_cincinnati_open",
    "tennis_atp_shanghai_masters",
    "tennis_atp_paris_masters",
    "tennis_atp_dubai",
    "tennis_atp_qatar_open",
    "tennis_atp_china_open",
    "tennis_wta_aus_open_singles",
    "tennis_wta_french_open",
    "tennis_wta_wimbledon",
    "tennis_wta_us_open",
    "tennis_wta_indian_wells_masters",
    "tennis_wta_miami_open",
    "tennis_wta_madrid_open",
    "tennis_wta_italian_open",
    "tennis_wta_canadian_open",
    "tennis_wta_cincinnati_open",
]


def _get_api_key() -> str:
    """Load ODDS_API_KEY from env, .env file, or Streamlit secrets."""
    key = os.environ.get("ODDS_API_KEY", "")
    if not key:
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
        if os.path.exists(env_path):
            with open(env_path) as fh:
                for line in fh:
                    line = line.strip()
                    if line.startswith("ODDS_API_KEY="):
                        key = line.split("=", 1)[1].strip().strip("\"'")
                        break
    if not key:
        try:
            import streamlit as st
            key = st.secrets.get("ODDS_API_KEY", "")
        except Exception:
            pass
    return key


def _devig(o1: float, o2: float) -> tuple[float, float]:
    """Multiplicative devig — returns (p1, p2) fair probs summing to 1."""
    r1, r2 = 1.0 / o1, 1.0 / o2
    total = r1 + r2
    return r1 / total, r2 / total


def fetch_active_tennis_odds(api_key: str, verbose: bool = False) -> list[dict]:
    """
    Fetch h2h odds for all active tennis events from The Odds API.

    Consumes:
      1 request  — /sports (to find which tournament keys are currently active)
      N requests — /sports/{key}/odds  (one per active tournament, typically 1-2)

    Returns one record per match, keeping the bookmaker with the tightest
    overround (lowest vig) so we get best-quality implied probabilities.
    """
    # Discover active sport keys — avoids wasting quota on inactive tournaments
    active_keys: set[str] = set()
    remaining: str | None = None
    try:
        resp = requests.get(
            f"{BASE_URL}/sports",
            params={"apiKey": api_key, "all": "false"},
            timeout=15,
        )
        resp.raise_for_status()
        remaining = resp.headers.get("x-requests-remaining")
        active_keys = {s["key"] for s in resp.json() if s.get("group") == "Tennis"}
        if verbose:
            print(f"  Active tennis sport keys: {sorted(active_keys)}")
    except Exception as e:
        print(f"[warn] Could not list active sports: {e}  (will attempt all known keys)")
        active_keys = set(TENNIS_SPORTS)

    all_matches: dict[tuple, dict] = {}  # (p1_lower, p2_lower) → best row

    for sport_key in TENNIS_SPORTS:
        if sport_key not in active_keys:
            if verbose:
                print(f"  skip (inactive): {sport_key}")
            continue

        try:
            resp = requests.get(
                f"{BASE_URL}/sports/{sport_key}/odds",
                params={
                    "apiKey":     api_key,
                    "regions":    "us,uk,eu",
                    "markets":    "h2h",
                    "oddsFormat": "decimal",
                },
                timeout=15,
            )
            remaining = resp.headers.get("x-requests-remaining")
            if resp.status_code == 404:
                continue
            resp.raise_for_status()
        except Exception as e:
            print(f"[warn] {sport_key}: {e}")
            continue

        for event in resp.json():
            p1 = event.get("home_team", "")
            p2 = event.get("away_team", "")
            ct = event.get("commence_time", "")
            best_bk: dict | None = None
            best_overround = 9999.0

            for bk in event.get("bookmakers", []):
                for mkt in bk.get("markets", []):
                    if mkt.get("key") != "h2h":
                        continue
                    outcomes = mkt.get("outcomes", [])
                    if len(outcomes) != 2:
                        continue
                    o_map = {o["name"]: float(o["price"]) for o in outcomes}
                    o1_odds = o_map.get(p1)
                    o2_odds = o_map.get(p2)
                    if not (o1_odds and o2_odds and o1_odds > 1 and o2_odds > 1):
                        continue
                    overround = 1.0 / o1_odds + 1.0 / o2_odds
                    if overround < best_overround:
                        best_overround = overround
                        dp1, dp2 = _devig(o1_odds, o2_odds)
                        best_bk = {
                            "player1":        p1,
                            "player2":        p2,
                            "player1_odds":   o1_odds,
                            "player2_odds":   o2_odds,
                            "bookmaker":      bk.get("title", "Unknown"),
                            "p1_prob":        round(dp1, 4),
                            "p2_prob":        round(dp2, 4),
                            "overround":      round(best_overround, 4),
                            "commence_time":  ct,
                            "sport_key":      sport_key,
                        }

            if best_bk:
                key = (p1.lower(), p2.lower())
                all_matches[key] = best_bk
                if verbose:
                    print(f"  {p1:25s} {best_bk['player1_odds']:.2f}  "
                          f"vs  {p2:25s} {best_bk['player2_odds']:.2f}"
                          f"  ({best_bk['bookmaker']})")

    if remaining is not None:
        print(f"  Requests remaining this month: {remaining}")

    return list(all_matches.values())


def main(dry_run: bool = False, verbose: bool = False) -> int:
    """Fetch odds and write to OUT_FILE. Returns number of matches written."""
    api_key = _get_api_key()
    if not api_key:
        print("[error] ODDS_API_KEY not set. "
              "Add it to .env or .streamlit/secrets.toml.")
        return 0

    now_utc  = datetime.now(timezone.utc)
    date_str = now_utc.strftime("%Y-%m-%d")

    # Idempotency: skip fetch if today's file is already present
    if not dry_run and os.path.exists(OUT_FILE):
        try:
            with open(OUT_FILE) as fh:
                existing = json.load(fh)
            if existing.get("date_fetched") == date_str:
                n = len(existing.get("matches", []))
                print(f"Today's Odds API cache already present "
                      f"({n} matches) — skipping API call.")
                return n
        except Exception:
            pass  # malformed file — re-fetch

    print(f"Fetching The Odds API tennis odds for {date_str} ...")
    matches = fetch_active_tennis_odds(api_key, verbose=verbose)
    print(f"  Found {len(matches)} matches with odds")

    if dry_run:
        for m in matches:
            print(f"  {m['player1']:25s} {m['player1_odds']:.2f}  vs  "
                  f"{m['player2']:25s} {m['player2_odds']:.2f}"
                  f"  ({m['bookmaker']})")
        return len(matches)

    os.makedirs(DATA_DIR, exist_ok=True)
    payload = {
        "date_fetched": date_str,
        "fetched_at":   now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "matches":      matches,
    }
    with open(OUT_FILE, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"  Written {len(matches)} matches -> {OUT_FILE}")
    return len(matches)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch today's tennis odds from The Odds API and cache locally."
    )
    parser.add_argument("--dry-run",  action="store_true", help="Print only, do not write")
    parser.add_argument("--verbose",  action="store_true", help="Show per-match detail")
    args = parser.parse_args()
    main(dry_run=args.dry_run, verbose=args.verbose)
