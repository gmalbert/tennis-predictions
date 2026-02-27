# Data Acquisition

## Overview

The primary historical data source is **Jeff Sackmann's `tennis_atp` repository** — the most comprehensive open tennis dataset available (1968–present, updated weekly). Live schedule data comes from **Flashscore** (primary) and **ESPN** (fallback). Betting odds come from **The Odds API**.

---

## 1. Historical Match Data — Jeff Sackmann

**Repository:** https://github.com/JeffSackmann/tennis_atp

### What's Included

| File Pattern | Contents |
|---|---|
| `atp_matches_YYYY.csv` | Match-level results per year (winner/loser, score, stats) |
| `atp_players.csv` | Player metadata (name, DOB, height, hand, country) |
| `atp_rankings_YYYY.csv` | Weekly ATP rankings |
| `atp_matches_qual_chall_YYYY.csv` | Qualifier & Challenger results |
| `atp_matches_futures_YYYY.csv` | Futures results |

### Key Columns in Match Files

```
tourney_id, tourney_name, surface, draw_size, tourney_level, tourney_date,
match_num, winner_id, winner_seed, winner_entry, winner_name, winner_hand,
winner_ht, winner_ioc, winner_age, winner_rank, winner_rank_points,
loser_id, loser_seed, loser_entry, loser_name, loser_hand, loser_ht,
loser_ioc, loser_age, loser_rank, loser_rank_points,
score, best_of, round, minutes,
w_ace, w_df, w_svpt, w_1stIn, w_1stWon, w_2ndWon, w_SvGms, w_bpSaved, w_bpFaced,
l_ace, l_df, l_svpt, l_1stIn, l_1stWon, l_2ndWon, l_SvGms, l_bpSaved, l_bpFaced
```

### Auto-Download Script (`setup_data.py`)

Place this at the project root. Streamlit Cloud will call it on first deploy.

```python
"""
setup_data.py
Downloads ATP data if not present.
Used by Streamlit Cloud on first deploy and for local setup.
"""
import os
import subprocess

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tennis_atp")


def setup():
    """Clone ATP data repo if it doesn't exist."""
    if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
        print("Downloading ATP match data...")
        subprocess.run(
            [
                "git", "clone", "--depth", "1",
                "https://github.com/JeffSackmann/tennis_atp.git",
                DATA_DIR,
            ],
            check=True,
        )
        print("ATP data downloaded successfully.")
    else:
        print("ATP data already present.")


if __name__ == "__main__":
    setup()
```

### Loading & Cleaning Code

```python
import pandas as pd
import numpy as np


def load_atp_data(
    data_dir: str,
    start_year: int = 1991,
    end_year: int = 2025,
) -> pd.DataFrame:
    """
    Load and combine ATP match data from multiple years.

    Args:
        data_dir:   Path to the tennis_atp directory.
        start_year: First year to include.
        end_year:   Last year to include.

    Returns:
        Combined DataFrame of all matches.
    """
    all_matches = []

    for year in range(start_year, end_year + 1):
        try:
            df = pd.read_csv(
                f"{data_dir}/atp_matches_{year}.csv",
                low_memory=False,
            )
            df["year"] = year
            all_matches.append(df)
            print(f"Loaded {year}: {len(df)} matches")
        except FileNotFoundError:
            print(f"Warning: No data file for {year}")

    combined = pd.concat(all_matches, ignore_index=True)
    print(f"\nTotal matches loaded: {len(combined)}")
    return combined


def load_player_data(data_dir: str) -> pd.DataFrame:
    """Load player information (height, hand, DOB, etc.)"""
    return pd.read_csv(f"{data_dir}/atp_players.csv")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove incomplete / problematic records.
    """
    print("\n--- Data Cleaning ---")
    initial_count = len(df)

    # Remove walkovers and retirements
    df = df[
        ~df["score"].str.contains("W/O|RET|DEF|Walkover", case=False, na=False)
    ]
    print(f"After removing walkovers/retirements: {len(df)}")

    # Drop rows missing critical IDs
    df = df.dropna(subset=["winner_id", "loser_id"])
    print(f"After removing missing player IDs: {len(df)}")

    # Drop rows missing surface
    df = df.dropna(subset=["surface"])
    print(f"After removing missing surface: {len(df)}")

    # Type conversions
    df["winner_id"] = df["winner_id"].astype(int)
    df["loser_id"] = df["loser_id"].astype(int)
    df["tourney_date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d")

    # Sort chronologically (required for correct ELO calculation)
    df = df.sort_values("tourney_date").reset_index(drop=True)

    removed = initial_count - len(df)
    print(f"\nRemoved {removed} matches ({removed / initial_count * 100:.1f}%)")
    print(f"Final dataset: {len(df)} matches")
    return df
```

---

## 2. Live Schedule Data

### Option A — Flashscore (Primary)

Flashscore exposes an internal live-feed endpoint that returns **all** professional tennis matches (ATP, WTA, Challengers, ITF, doubles).

| Detail | Value |
|---|---|
| URL | `https://www.flashscore.com/x/feed/f_2_0_2_en-gb_1` |
| Method | GET |
| Required Header | `x-fsign: SW9D1eZo` |
| Response | Custom delimited text (see [scheduling.md](05_scheduling.md)) |
| Coverage | Comprehensive — every professional match |
| Rate Limit | Informal; cache results ≥ 2 min |

### Option B — ESPN (Fallback)

ESPN has a free public JSON API for ATP/WTA main-tour scoreboard data.

| Tour | Endpoint |
|---|---|
| ATP | `https://site.api.espn.com/apis/site/v2/sports/tennis/atp/scoreboard` |
| WTA | `https://site.api.espn.com/apis/site/v2/sports/tennis/wta/scoreboard` |

Pros: Clean JSON. Cons: Main tour only (no Challengers/ITF).

---

## 3. Betting Odds Data

### The Odds API

| Detail | Value |
|---|---|
| Website | https://the-odds-api.com |
| Free tier | 500 requests/month |
| Endpoint | `https://api.the-odds-api.com/v4/sports/{sport_key}/odds` |
| Format | JSON with decimal odds per bookmaker |
| Supported tournaments | All Grand Slams, Masters 1000, 500s |

See [odds_integration.md](06_odds_integration.md) for full client code.

### Supported Tournament Keys

```python
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
```

---

## 4. Update Pipeline

### Weekly Retrain Script (`update_data.sh`)

```bash
#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "=== Updating ATP data ==="
cd tennis_atp
git pull origin master
cd ..

echo "=== Retraining model ==="
python tennis_predictor.py

echo "=== Done ==="
```

### Recommended Cron Schedule

```
# Retrain every Monday at 03:00 UTC
0 3 * * 1 /path/to/update_data.sh >> /path/to/retrain.log 2>&1
```

---

## 5. Data Summary

| Source | Type | Format | Refresh |
|---|---|---|---|
| Jeff Sackmann `tennis_atp` | Historical matches + players | CSV | Weekly (git pull) |
| Flashscore live feed | Today's schedule & live scores | Custom text | Every 2 min (cached) |
| ESPN scoreboard API | Today's ATP/WTA schedule | JSON | Every 2 min (cached) |
| The Odds API | Bookmaker odds | JSON | Every 5 min (cached) |
