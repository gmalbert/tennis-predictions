# Tennis Predictions — GitHub Copilot Instructions

## Project Overview

**App name:** Tennis Predictions
**Purpose:** Streamlit app predicting ATP/WTA match outcomes and surfacing betting value.
**Entry point:** `streamlit run predictions.py`
**Part of:** Betting Oracle suite

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit (single-page, tabbed) |
| ML | scikit-learn (match winner probability) |
| Data | pandas, Parquet feature store, tennis-data.co.uk |
| Odds | The Odds API, flashscore |
| Config | python-dotenv (`.env` file) |
| Python | 3.9+ |

---

## File Conventions

### Key files
- `predictions.py` — entry point; sets `st.set_page_config` ONCE.
- `predict.py` — model loading and prediction logic.
- `features.py` — feature engineering for match predictions.
- `fetch_today_odds.py` — fetch today's live odds from The Odds API.
- `ingest_tennis_data_co_uk.py` — ingest historical ATP/WTA match data.
- `scraper_atp.py` — ATP tour data scraper.
- `footer.py` — `add_betting_oracle_footer()` must be called at page bottom.
- `scripts/export_best_bets.py` — exports `data_files/best_bets_today.json` for Sports Picks Grid.

### Data files
- `data_files/features_2020_present.parquet` — pre-computed feature store (all matches 2020+)
- `data_files/tennis_predictor.pkl` — trained scikit-learn model
- `data_files/logo.png` — app logo
- `data_files/best_bets_today.json` — unified schema for Sports Picks Grid aggregator

---

## Tennis Domain Knowledge

### Player stats (key features)
- Recent win rate (last 10, last 20 matches)
- Surface-specific win rates (hard, clay, grass, indoor)
- Head-to-head record
- Recent form (last 5 matches)
- Serve/return stats from ATP/WTA API

### Bet types
- `moneyline` — match winner
- `set_spread` — handicap sets
- `total_sets` — over/under total sets played

### Surface encoding
- `0` = hard, `1` = clay, `2` = grass, `3` = indoor hard

---

## Coding Conventions

- `st.set_page_config()` called ONCE in `predictions.py` only
- Use `width='stretch'` for dataframes/charts
- `DATA_DIR = "data_files"` — reference all files relative to this
- Wrap model loading and prediction in try/except
- API keys via `python-dotenv`; never hardcode; `.env` is gitignored
- Guard against missing feature columns before calling model predict

---

## Export for Sports Picks Grid

`scripts/export_best_bets.py` loads today's matches with odds, runs predictions, and writes `data_files/best_bets_today.json`.

Run: `python scripts/export_best_bets.py`
