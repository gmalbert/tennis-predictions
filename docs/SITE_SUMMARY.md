> **AI Onboarding Guide** — See also the project docs folder for detailed architecture documentation.

# Tennis Predictions — Site Summary

## What This App Does

Comprehensive ATP/WTA tennis match prediction system. Uses 1968–present historical data (TennisMyLife), builds Elo ratings and rolling surface-form features, trains an XGBoost classifier, and fetches live pre-match odds via Matchstat RapidAPI for edge calculations. The Streamlit UI shows today's matches, a match explorer, and Elo rankings.

## Quick Start

```bash
# 1. Activate virtual environment
.\.venv\Scripts\Activate.ps1        # Windows
source .venv/bin/activate           # macOS/Linux

# 2. (One-time or periodic) Update historical data
python update_tml_data.py           # Download current-year TennisMyLife CSV files

# 3. Build features and train model
python features.py                  # Build feature matrix from 1968+ matches
python train.py                     # Train XGBoost → tennis_predictor.pkl

# 4. Run the app
streamlit run predictions.py
```

GitHub Actions handles Steps 2–4 nightly via `update_data.yml`.

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit ≥1.51 (3-tab layout: Today, Match Explorer, Elo Rankings) |
| ML | XGBoost 1.5+ classifier |
| Historical data | TennisMyLife (1968–present, MIT-licensed) |
| Live odds | Matchstat RapidAPI (500 calls/month budget) |
| Scraping | BeautifulSoup4, Playwright (headless browser) |
| Data storage | Parquet (feature matrix), PKL (model) |

## Key Files

| File | Purpose |
|---|---|
| `predictions.py` | Entry point — 3 tabs: Today's Matches, Match Explorer, Elo Rankings |
| `features.py` | Build feature matrix: Elo diff, surface form, H2H, rolling stats from 1968+ matches |
| `train.py` | Train XGBoost, evaluate (AUC, Brier, log loss), save to `tennis_predictor.pkl` |
| `predict.py` | Load trained model → generate live predictions for upcoming matches |
| `update_tml_data.py` | Download current-year TennisMyLife CSV files |
| `matchstat_api.py` | Matchstat RapidAPI client — live pre-match odds (500 calls/month hard limit) |
| `fetch_today_odds.py` | Fetch and cache today's live odds (per-day cache file) |
| `enrich_with_odds.py` | Merge odds with predictions, calculate market probabilities and edge |
| `scripts/export_best_bets.py` | Extract elite/strong bets → `best_bets_today.json` |

## Data Flow

1. **Historical data**: TennisMyLife CSVs (1968–present) + tennis-data.co.uk odds (xlsx) → `features.py`
2. **Feature engineering**: Elo seeding, rolling form per surface, H2H stats → `feature_matrix.parquet`
3. **Training**: `train.py` → XGBoost → `tennis_predictor.pkl`
4. **Live odds** (daily): `fetch_today_odds.py` → Matchstat RapidAPI (cached per day) → `predict.py`
5. **Edge calculation**: `enrich_with_odds.py` → model probability vs bookmaker implied probability
6. **Export**: `scripts/export_best_bets.py` → `best_bets_today.json` (consumed by sports-picks-grid)

## Environment Variables

| Variable | Purpose | Required |
|---|---|---|
| `MATCHSTAT_API_KEY` | Matchstat via RapidAPI — live pre-match odds | Required for live odds |
| `ODDS_API_KEY` | The Odds API — historical odds joins | Optional |

## Critical API Budget Rule

**Matchstat API = 500 calls/month.** The per-day cache in `fetch_today_odds.py` is mandatory — never bypass it. If the monthly budget is exhausted, the app falls back to model-only predictions without live odds.

## Critical Conventions

- Feature engineering uses winner/loser perspective — randomize perspective for downstream models to prevent bias
- Name normalization between TennisMyLife, Matchstat, and tennis-data.co.uk has ~81.5% success rate — mismatches silently drop matches
- Always check whether `feature_matrix.parquet` is fresh before running predictions (the nightly action regenerates it)
- `scraper_atp.py` and `scraper_itf.py` handle data collection from Flashscore / Tennis Abstract

## Common Gotchas

- Playwright requires separate browser install: `playwright install chromium`
- If Matchstat API returns 429 (rate exceeded), the per-day cache prevents repeated calls in the same day, but the monthly cap will still be hit if not monitored
- ELO ratings are basic (no rating deviation / volatility); new players and returning-from-injury players may have stale ratings
- `ingest_tennis_data_co_uk.py` downloads xlsx files from tennis-data.co.uk — these have historical odds for calibration, not live odds
