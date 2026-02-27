# Project Roadmap

## Tennis Analytics & Betting Platform

A Streamlit-based web application for tennis match predictions, ELO rankings, live schedules, and value-bet identification.

---

## Phase 1 — Data Foundation

| Task | Description | Doc |
|------|-------------|-----|
| Historical match data | Clone Jeff Sackmann's `tennis_atp` repo (1968–present) | [data_acquisition.md](data_acquisition.md) |
| Player metadata | Heights, DOB, handedness, nationality from same repo | [data_acquisition.md](data_acquisition.md) |
| Data cleaning | Remove walkovers, retirements, missing fields | [data_acquisition.md](data_acquisition.md) |

## Phase 2 — Feature Engineering & ELO

| Task | Description | Doc |
|------|-------------|-----|
| ELO rating system | Overall + surface-specific ratings (Hard, Clay, Grass) | [features.md](features.md) |
| Head-to-head tracking | Win/loss record between any two players | [features.md](features.md) |
| Recent form | Win rate in last 50 matches | [features.md](features.md) |
| Serve statistics | Ace rate, DF rate, 1st serve %, break-point save % | [features.md](features.md) |
| Match context | Ranking diff, height diff, age diff, tournament level, best-of format | [features.md](features.md) |

## Phase 3 — Modeling

| Task | Description | Doc |
|------|-------------|-----|
| Balanced dataset construction | Randomly swap winner/loser perspective so model learns both sides | [modeling.md](modeling.md) |
| Baseline (ELO-only) | Predict higher-rated player wins (~64%) | [modeling.md](modeling.md) |
| Decision Tree / Random Forest | Interpretable tree models (~66-70%) | [modeling.md](modeling.md) |
| XGBoost | Gradient boosting for best accuracy (~70%) | [modeling.md](modeling.md) |
| Neural Network | MLP classifier (~69%) | [modeling.md](modeling.md) |
| Model serialization | Pickle best model + ELO state for serving | [modeling.md](modeling.md) |

## Phase 4 — Live Data & Scheduling

| Task | Description | Doc |
|------|-------------|-----|
| Flashscore scraping | Parse Flashscore live feed for ATP/WTA/Challengers/ITF schedules | [scheduling.md](scheduling.md) |
| ESPN fallback | ESPN public API for ATP/WTA main-tour schedules | [scheduling.md](scheduling.md) |
| Player name matching | Fuzzy-match API names to our database IDs | [scheduling.md](scheduling.md) |
| Completed match tracking | Parse finished matches for backlog accuracy stats | [scheduling.md](scheduling.md) |

## Phase 5 — Odds & Value Bets

| Task | Description | Doc |
|------|-------------|-----|
| The Odds API integration | Fetch live bookmaker odds (decimal format) | [odds_integration.md](odds_integration.md) |
| Value-bet calculator | Compare model probability to implied probability | [odds_integration.md](odds_integration.md) |
| Bookie slip generator | Auto-generate high-confidence picks | [odds_integration.md](odds_integration.md) |

## Phase 6 — Streamlit UI

| Task | Description | Doc |
|------|-------------|-----|
| Today's Matches page | Live/upcoming matches with predictions | [layout.md](layout.md) |
| Match Prediction page | Head-to-head predictor with confidence meter | [layout.md](layout.md) |
| ELO Rankings page | Sortable rankings with surface breakdown | [layout.md](layout.md) |
| Tournament Simulator | Bracket simulation | [layout.md](layout.md) |
| Player Analysis page | Per-player ELO breakdown + H2H tool | [layout.md](layout.md) |
| Prediction Backlog | Accuracy tracking of past predictions | [layout.md](layout.md) |

## Phase 7 — Deployment & Maintenance

| Task | Description | Doc |
|------|-------------|-----|
| Streamlit Cloud deployment | Auto-setup via `setup_data.py` | [general_considerations.md](general_considerations.md) |
| Weekly model retraining | Cron/scheduled script to update ELOs and retrain | [general_considerations.md](general_considerations.md) |
| Performance & caching | `@st.cache_resource` / `@st.cache_data` | [general_considerations.md](general_considerations.md) |
| Legal & ethical considerations | Responsible gambling, data licensing | [general_considerations.md](general_considerations.md) |

---

## File Structure (Target)

```
tennis-predictions/
├── predictions.py              # Main Streamlit app (entry point)
├── requirements.txt
├── setup_data.py               # Auto-download ATP data on first run
├── tennis_predictor.py         # Model training pipeline
├── tennis_schedule.py          # Live match schedule scraping
├── odds_api.py                 # Odds API client + value bet calculator
├── .streamlit/
│   └── config.toml             # Streamlit theme config
├── data_files/
│   └── logo.png
├── tennis_atp/                 # Cloned at runtime (gitignored)
│   ├── atp_matches_YYYY.csv
│   └── atp_players.csv
├── docs/
│   ├── 01_roadmap.md
│   ├── 02_data_acquisition.md
│   ├── 03_features.md
│   ├── 04_modeling.md
│   ├── 05_scheduling.md
│   ├── 06_odds_integration.md
│   ├── 07_layout.md
│   └── 08_general_considerations.md
└── models/                     # Serialized .pkl files
    └── tennis_predictor.pkl
```
