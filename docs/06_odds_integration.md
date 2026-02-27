# Odds Integration & Value Betting

## Overview

Live bookmaker odds are fetched from **The Odds API**. By comparing our model's win probability to the bookmaker's implied probability, we identify **value bets** — situations where our edge exceeds the bookmaker's margin.

---

## 1. The Odds API Client

| Detail | Value |
|---|---|
| Website | https://the-odds-api.com |
| Free tier | 500 requests/month |
| Base URL | `https://api.the-odds-api.com/v4` |
| Auth | API key as query param |
| Response | JSON, decimal odds |

### Data Model

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class TennisOdds:
    """Odds for a tennis match from a single bookmaker."""
    player1: str
    player2: str
    player1_odds: float     # decimal odds (e.g. 1.50)
    player2_odds: float     # decimal odds (e.g. 2.80)
    bookmaker: str
    commence_time: Optional[datetime] = None
    tournament: str = ""

    @property
    def player1_implied_prob(self) -> float:
        return 1 / self.player1_odds if self.player1_odds > 0 else 0

    @property
    def player2_implied_prob(self) -> float:
        return 1 / self.player2_odds if self.player2_odds > 0 else 0

    @property
    def overround(self) -> float:
        """Bookmaker margin (vig). Always > 1."""
        return self.player1_implied_prob + self.player2_implied_prob
```

### Full Client

```python
import os
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple


# Supported tournament keys
TENNIS_SPORTS = {
    "Australian Open":   "tennis_atp_aus_open_singles",
    "French Open":       "tennis_atp_french_open",
    "Wimbledon":         "tennis_atp_wimbledon",
    "US Open":           "tennis_atp_us_open",
    "Indian Wells":      "tennis_atp_indian_wells",
    "Miami Open":        "tennis_atp_miami_open",
    "Monte Carlo":       "tennis_atp_monte_carlo_masters",
    "Madrid Open":       "tennis_atp_madrid_open",
    "Italian Open":      "tennis_atp_italian_open",
    "Canadian Open":     "tennis_atp_canadian_open",
    "Cincinnati":        "tennis_atp_cincinnati_open",
    "Shanghai Masters":  "tennis_atp_shanghai_masters",
    "Paris Masters":     "tennis_atp_paris_masters",
    "Dubai":             "tennis_atp_dubai",
    "Qatar Open":        "tennis_atp_qatar_open",
    "China Open":        "tennis_atp_china_open",
}


class OddsAPIClient:
    BASE_URL = "https://api.the-odds-api.com/v4"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._cache: Dict[str, List[TennisOdds]] = {}
        self._cache_expiry: Dict[str, datetime] = {}
        self._remaining_requests = None
        self._used_requests = None

    # ---- Caching ----
    def _cache_key(self, sport: str, regions: str) -> str:
        return f"{sport}_{regions}"

    def _is_cached(self, key: str) -> bool:
        return key in self._cache_expiry and datetime.now() < self._cache_expiry[key]

    # ---- Public methods ----
    def get_active_tournaments(self) -> List[str]:
        """Which tennis tournaments currently have odds available."""
        try:
            resp = requests.get(
                f"{self.BASE_URL}/sports",
                params={"apiKey": self.api_key, "all": "true"},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException:
            return []

        active = []
        for sport in data:
            if sport.get("group") == "Tennis" and sport.get("active"):
                for name, key in TENNIS_SPORTS.items():
                    if key == sport.get("key"):
                        active.append(name)
                        break
        return active

    def get_tennis_odds(
        self,
        tournament: Optional[str] = None,
        regions: str = "uk,eu",
        markets: str = "h2h",
    ) -> List[TennisOdds]:
        """
        Fetch odds for a tournament (or scan all supported tournaments).
        """
        if tournament:
            sport_key = TENNIS_SPORTS.get(tournament)
            if not sport_key:
                print(f"Unknown tournament: {tournament}")
                return []
            return self._fetch(sport_key, regions, markets)

        # Scan all
        for name, key in TENNIS_SPORTS.items():
            odds = self._fetch(key, regions, markets)
            if odds:
                return odds
        return []

    def _fetch(self, sport_key: str, regions: str, markets: str) -> List[TennisOdds]:
        ck = self._cache_key(sport_key, regions)
        if self._is_cached(ck):
            return self._cache[ck]

        try:
            resp = requests.get(
                f"{self.BASE_URL}/sports/{sport_key}/odds",
                params={
                    "apiKey": self.api_key,
                    "regions": regions,
                    "markets": markets,
                    "oddsFormat": "decimal",
                },
                timeout=10,
            )
            self._remaining_requests = resp.headers.get("x-requests-remaining")
            self._used_requests = resp.headers.get("x-requests-used")
            if resp.status_code == 404:
                return []
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            print(f"Odds API error: {e}")
            return []

        odds_list: List[TennisOdds] = []
        for event in data:
            p1 = event.get("home_team", "")
            p2 = event.get("away_team", "")
            ct = event.get("commence_time")

            for bk in event.get("bookmakers", []):
                for market in bk.get("markets", []):
                    if market.get("key") != "h2h":
                        continue
                    outcomes = market.get("outcomes", [])
                    if len(outcomes) != 2:
                        continue

                    p1_odds = p2_odds = None
                    for o in outcomes:
                        if o["name"] == p1:
                            p1_odds = o["price"]
                        elif o["name"] == p2:
                            p2_odds = o["price"]

                    if p1_odds and p2_odds:
                        odds_list.append(TennisOdds(
                            player1=p1,
                            player2=p2,
                            player1_odds=p1_odds,
                            player2_odds=p2_odds,
                            bookmaker=bk.get("title", "Unknown"),
                            tournament=sport_key,
                            commence_time=(
                                datetime.fromisoformat(ct.replace("Z", "+00:00"))
                                if ct else None
                            ),
                        ))

        self._cache[ck] = odds_list
        self._cache_expiry[ck] = datetime.now() + timedelta(minutes=5)
        return odds_list

    def get_best_odds(
        self, player1: str, player2: str, tournament: Optional[str] = None
    ) -> Optional[Tuple[TennisOdds, TennisOdds]]:
        """Find the best available odds for a specific match across bookmakers."""
        all_odds = self.get_tennis_odds(tournament)
        if not all_odds:
            return None

        p1_low, p2_low = player1.lower(), player2.lower()
        best_p1 = best_p2 = None

        for odds in all_odds:
            o1, o2 = odds.player1.lower(), odds.player2.lower()

            # Fuzzy last-name matching
            match_p1 = p1_low in o1 or o1 in p1_low or p1_low.split()[-1] in o1
            match_p2 = p2_low in o2 or o2 in p2_low or p2_low.split()[-1] in o2

            if not (match_p1 and match_p2):
                # Try swapped
                match_p1 = p1_low in o2 or o2 in p1_low or p1_low.split()[-1] in o2
                match_p2 = p2_low in o1 or o1 in p2_low or p2_low.split()[-1] in o1
                if match_p1 and match_p2:
                    odds = TennisOdds(
                        player1=odds.player2, player2=odds.player1,
                        player1_odds=odds.player2_odds, player2_odds=odds.player1_odds,
                        bookmaker=odds.bookmaker, tournament=odds.tournament,
                        commence_time=odds.commence_time,
                    )
                else:
                    continue

            if best_p1 is None or odds.player1_odds > best_p1.player1_odds:
                best_p1 = odds
            if best_p2 is None or odds.player2_odds > best_p2.player2_odds:
                best_p2 = odds

        return (best_p1, best_p2) if (best_p1 or best_p2) else None

    @property
    def api_usage(self) -> Dict:
        return {"remaining": self._remaining_requests, "used": self._used_requests}


def create_odds_client(api_key: Optional[str] = None) -> OddsAPIClient:
    if api_key is None:
        api_key = os.environ.get("ODDS_API_KEY", "")
    return OddsAPIClient(api_key)
```

---

## 2. Value Bet Calculator

A bet has **positive expected value** when:

$$EV = P_{model} \times Odds_{decimal} - 1 > 0$$

The **edge** is the difference between our probability and the bookmaker's implied probability:

$$Edge = P_{model} - P_{implied}$$

where $P_{implied} = \frac{1}{Odds_{decimal}}$.

```python
class ValueBetCalculator:
    """Compare model predictions to bookmaker odds to find value."""

    def __init__(self, odds_client: OddsAPIClient):
        self.client = odds_client

    def analyze(
        self,
        player1: str,
        player2: str,
        model_prob_p1: float,
        tournament: Optional[str] = None,
    ) -> Dict:
        model_prob_p2 = 1 - model_prob_p1
        best = self.client.get_best_odds(player1, player2, tournament)

        result = {
            "player1": player1,
            "player2": player2,
            "model_prob_p1": model_prob_p1,
            "model_prob_p2": model_prob_p2,
            "live_odds_available": False,
            "p1_analysis": None,
            "p2_analysis": None,
        }

        if best:
            bp1, bp2 = best
            result["live_odds_available"] = True

            if bp1:
                ev = model_prob_p1 * bp1.player1_odds - 1
                result["p1_analysis"] = {
                    "odds": bp1.player1_odds,
                    "bookmaker": bp1.bookmaker,
                    "implied_prob": bp1.player1_implied_prob,
                    "expected_value": ev,
                    "edge": model_prob_p1 - bp1.player1_implied_prob,
                    "is_value_bet": ev > 0,
                }

            if bp2:
                ev = model_prob_p2 * bp2.player2_odds - 1
                result["p2_analysis"] = {
                    "odds": bp2.player2_odds,
                    "bookmaker": bp2.bookmaker,
                    "implied_prob": bp2.player2_implied_prob,
                    "expected_value": ev,
                    "edge": model_prob_p2 - bp2.player2_implied_prob,
                    "is_value_bet": ev > 0,
                }
        else:
            # No live odds — show fair odds from model
            result["p1_analysis"] = {
                "fair_odds": 1 / model_prob_p1 if model_prob_p1 > 0 else 999,
                "note": "No live odds available",
            }
            result["p2_analysis"] = {
                "fair_odds": 1 / model_prob_p2 if model_prob_p2 > 0 else 999,
                "note": "No live odds available",
            }

        return result
```

---

## 3. Confidence Thresholds

| Level | Threshold | Action |
|---|---|---|
| **HIGH** | ≥ 75% | Show on bookie slip, green badge |
| **MEDIUM** | 65–75% | Show prediction, orange badge |
| **LOW** | < 65% | Show "skip" or gray badge |

```python
HIGH_CONFIDENCE = 0.75
MEDIUM_CONFIDENCE = 0.65
```

---

## 4. Bookie Slip (Auto-Generated)

For the next 12 hours of upcoming matches, filter to high-confidence picks and render as a styled card:

```python
from datetime import timedelta, timezone

def generate_bookie_slip(matches, predictions, player_names, elo):
    """
    Build a list of high-confidence picks for the next 12 hours.
    """
    cutoff = datetime.now(timezone.utc) + timedelta(hours=12)
    picks = []

    for match in matches:
        if match.status == "live":
            continue
        if match.start_time > cutoff:
            continue

        pred = predictions.get(id(match))
        if pred and pred[1] >= HIGH_CONFIDENCE:
            picks.append({
                "match": match,
                "winner": pred[0],
                "confidence": pred[1],
            })

    picks.sort(key=lambda p: -p["confidence"])
    return picks
```

---

## 5. API Key Setup

### Environment Variable

```bash
export ODDS_API_KEY="your_key_here"
```

### Streamlit Secrets (for Streamlit Cloud)

Create `.streamlit/secrets.toml`:

```toml
ODDS_API_KEY = "your_key_here"
```

Then access in code:

```python
import streamlit as st

api_key = st.secrets.get("ODDS_API_KEY", os.environ.get("ODDS_API_KEY", ""))
```

---

## 6. Rate Limiting & Best Practices

- **Free tier**: 500 requests/month → ~16/day
- **Cache aggressively**: 5-minute TTL per sport/region combo
- **Track usage**: Check `x-requests-remaining` header after each call
- **Batch wisely**: Fetch odds per tournament (not per match)
- **Off-season**: No requests consumed when tournaments aren't active
