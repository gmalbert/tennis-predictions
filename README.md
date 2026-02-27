# tennis-predictions
Comprehensive tennis match-prediction system powered by historical data,
bookmaker odds, and machine learning. Designed to run as a Streamlit web
application; data pipelines operate autonomously on GitHub Actions.

This repository originally took inspiration from
[LewisWJackson's tennis predictor](https://github.com/LewisWJackson/tennis-predictor),
but has evolved substantially with new data sources, caching layers, and a
modern UI.

## 📦 Key Features

- **Historical data (2020–present)** built from TennisMyLife and tennis-data.co.uk;
  odds matched via intelligent name normalisation (81.5% success rate).  
- **Live pre-match odds** fetched from Matchstat RapidAPI with per-day caching
  and a 500-call/month budget guard.  
- **Full feature engineering pipeline** generating ELO, serve stats, surface
  form, H2H counts, and market probabilities.  Built daily via GitHub Actions.  
- **Streamlit UI** with three tabs:
  1. *Today's Matches* (live odds, ELO, market value)
  2. *Match Explorer* (filterable historical dataset)
  3. *ELO Rankings* (overall and surface leaderboards)
- **Automated update workflow** (`.github/workflows/update_data.yml`) downloads
  latest matches and rebuilds features, committing changes back to `main`.
- **MIT-licensed data sources**: TennisMyLife (1968–present) and tennis-data.co.uk
  (odds 2020–2025).  All code is permissively licensed.

## 🚀 Getting Started
1. Clone this repo and create a Python 3.11 venv.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Populate keys:
   - `ODDS_API_KEY` for The Odds API (optional, project now uses Matchstat)
   - `RAPIDAPI_KEY` for Matchstat tennis API; place in `.env` or
     `.streamlit/secrets.toml`
4. Run initial data prep:
   ```bash
   python update_tml_data.py       # download current-year TML files
   python features.py              # build feature matrix (2020+)
   ```
5. Start the app:
   ```bash
   streamlit run predictions.py
   ```
6. Deploy to Streamlit Cloud by connecting this repo; the GitHub Action will
   keep data fresh each morning.

## 📁 Repository Structure
```
tennis-predictions/
├── data_files/                 # intermediate and output datasets
│   ├── features_2020_present.parquet  # feature matrix used by app
│   └── *.xlsx                   # raw tennis-data.co.uk downloads
├── docs/                       # design and reference documentation
├── tml-data/                   # TennisMyLife CSVs + enriched odds
├── matchstat_api.py            # client with caching & budget tracking
├── features.py                 # feature engineering pipeline
├── update_tml_data.py          # daily TML downloader
├── enrich_with_odds.py         # join tennis-data.co.uk odds onto TML
├── predictions.py              # Streamlit application
└── .github/workflows/update_data.yml  # scheduled data-refresh CI
```

## 🛠 CI / Automation
A GitHub Action (`update_data.yml`) runs daily at 05:00 UTC to:  
1. Refresh current-year TML files.  
2. Re-run `features.py` to rebuild `features_2020_present.parquet`.  
3. Commit and push any changed data.

## 📚 Documentation
See the `docs/` folder for deeper guides — data acquisition, feature
engineering, odds integration, and more.  Start with
[docs/01_roadmap.md](docs/01_roadmap.md).

---

Happy coding and may your nets be full of aces! 🎾