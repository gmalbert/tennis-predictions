# Tennis Predictions — Architecture

## Overview
Streamlit app predicting ATP/WTA match outcomes and surfacing betting value. A scikit-learn model is trained on a Parquet feature store built from tennis-data.co.uk historical data. Odds are fetched from The Odds API.

## Data Flow
```
tennis-data.co.uk (historical CSV — ATP/WTA)
        ↓
ingest_tennis_data_co_uk.py
        ↓
data_files/features_2020_present.parquet (pre-computed feature store)
        ↓
features.py → feature engineering (surfaces, form, H2H, serve/return stats)
        ↓
data_files/tennis_predictor.pkl (trained scikit-learn model)
        ↓
predict.py → match winner probability
        ↓
fetch_today_odds.py → The Odds API (ATP/WTA lines)
        ↓
predictions.py (Streamlit entry) → tabbed UI
        ↓
scripts/export_best_bets.py → data_files/best_bets_today.json
```

## ML Model
- **Algorithm**: scikit-learn classifier (match winner probability, 0–1)
- **Feature store**: `data_files/features_2020_present.parquet`
- **Artifact**: `data_files/tennis_predictor.pkl`
- **Surface encoding**: 0=hard, 1=clay, 2=grass, 3=indoor hard

### Key Features
| Feature | Description |
|---------|-------------|
| `win_rate_l10`, `win_rate_l20` | Recent win rate (last 10/20 matches) |
| `surface_win_rate_{surface}` | Surface-specific win rates |
| `h2h_record` | Head-to-head record vs opponent |
| `form_l5` | Last 5 match form points |
| `serve_stats`, `return_stats` | ATP/WTA serve/return averages |

## API Integrations
| Source | Purpose | Key |
|--------|---------|-----|
| tennis-data.co.uk | Historical ATP/WTA CSVs | None (public) |
| The Odds API | ATP/WTA betting odds | `ODDS_API_KEY` |
| ATP/WTA API | Live serve/return stats | None (scraper) |

## Key Components
- `predictions.py` — entry, `st.set_page_config`
- `predict.py` — model loading and prediction logic
- `features.py` — feature engineering from parquet store
- `fetch_today_odds.py` — fetch today's odds from The Odds API
- `ingest_tennis_data_co_uk.py` — ingest historical match data
- `scraper_atp.py` — ATP tour data scraper
- `footer.py` — `add_betting_oracle_footer()`
- `scripts/export_best_bets.py` — exports `best_bets_today.json`

## Edge Calculation
- `edge` = model implied probability - market implied probability
- EV_THRESHOLD: flag when edge > 3%

## Storage
- `data_files/features_2020_present.parquet` — feature store (all matches 2020+)
- `data_files/tennis_predictor.pkl` — trained model
- `data_files/logo.png` — app logo
- `data_files/best_bets_today.json` — unified Sports Picks Grid schema
