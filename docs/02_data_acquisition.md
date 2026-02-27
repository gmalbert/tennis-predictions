# Data Acquisition

## Overview

Data comes from three sources, each serving a distinct role:

| Source | Role | Update frequency |
|---|---|---|
| **TennisMyLife (TML)** | Historical match results + serve/return stats | Daily (automated) |
| **tennis-data.co.uk** | Historical betting odds (2020–2025) | Annual (manual, one-time) |
| **Matchstat RapidAPI** | Live upcoming fixtures + pre-match odds | Daily at runtime (cached) |

All data is scoped to **2020 onward** for modelling. Older TML files are kept on disk but excluded from feature computation and training.

---

## 1. TennisMyLife (TML)

**URL:** https://stats.tennismylife.org  
**License:** MIT  
**Local path:** `tml-data/`

### What's included

| File pattern | Contents |
|---|---|
| `YYYY.csv` | ATP main-tour results per year (1968–2026) |
| `YYYY_challenger.csv` | Challenger results per year (1978–2026) |
| `ongoing_tourneys.csv` | Live in-progress matches, updated daily |
| `ATP_Database.csv` | Full historical union file |

### Schema (50 columns, Sackmann-compatible)

```
tourney_id, tourney_name, surface, draw_size, tourney_level, indoor,
tourney_date, match_num,
winner_id, winner_seed, winner_entry, winner_name, winner_hand,
winner_ht, winner_ioc, winner_age, winner_rank, winner_rank_points,
loser_id,  loser_seed,  loser_entry,  loser_name,  loser_hand,
loser_ht,  loser_ioc,  loser_age,  loser_rank,  loser_rank_points,
score, best_of, round, minutes,
w_ace, w_df, w_svpt, w_1stIn, w_1stWon, w_2ndWon, w_SvGms, w_bpSaved, w_bpFaced,
l_ace, l_df, l_svpt, l_1stIn, l_1stWon, l_2ndWon, l_SvGms, l_bpSaved, l_bpFaced
```

### Automated update

`update_tml_data.py` re-downloads only the files that change daily:

- Current year main tour (`2026.csv`)
- Current year challengers (`2026_challenger.csv`)
- Live matches (`ongoing_tourneys.csv`)

Run manually: `python update_tml_data.py`  
Runs in CI: `.github/workflows/update_data.yml` (daily at 05:00 UTC)

---

## 2. tennis-data.co.uk Betting Odds (historical, 2020–2025)

**URL:** https://www.tennis-data.co.uk  
**Local path:** `data_files/` (raw xlsx), `tml-data/YYYY_with_odds.csv` (joined)  
**Combined file:** `tml-data/atp_2020_2025_with_odds.csv`

### What's included

Decimal odds from multiple bookmakers, captured just before match start:

| Column | Bookmaker |
|---|---|
| `B365W` / `B365L` | Bet365 |
| `PSW` / `PSL` | Pinnacle |
| `MaxW` / `MaxL` | Oddsportal best available |
| `AvgW` / `AvgL` | Oddsportal market average |
| `BFEW` / `BFEL` | Betfair Exchange |

### Join methodology

`enrich_with_odds.py` joins tennis-data.co.uk rows onto TML rows using:

1. **Name normalisation** — `_name_keys()` generates 1–2 lookup keys per name handling initials, hyphens, apostrophes, compound surnames (e.g. "Carreno Busta P." and "Pablo Carreno Busta" map to the same key)
2. **Date proximity** — when two files differ by up to 3 days for the same tournament, the closest date wins
3. **Adjacent-year lookup** — January tournaments that appear in the previous year's xlsx are handled automatically

### Match rate

| Year | Rows | Matched | Rate |
|---|---|---|---|
| 2020 | 1,466 | 1,147 | 78.2% |
| 2021 | 2,735 | 2,264 | 82.8% |
| 2022 | 2,918 | 2,415 | 82.8% |
| 2023 | 2,995 | 2,413 | 80.6% |
| 2024 | 3,076 | 2,504 | 81.4% |
| 2025 | 2,944 | 2,406 | 81.7% |
| **Total** | **16,134** | **13,149** | **81.5%** |

### Why 18.5% is unmatched (structural ceiling, not a matching failure)

- **~40 rows** — United Cup, Laver Cup, Next Gen Finals: team/exhibition events never published by tennis-data.co.uk
- **~2,500 rows** — Qualifying rounds at Slams and Masters 1000s: not covered by tennis-data.co.uk

These cannot be recovered by improving name matching. They are excluded from odds-dependent model features.

### How to re-run enrichment (manual, once per year)

1. Download the new year's xlsx manually from `https://www.tennis-data.co.uk/YYYY/YYYY.xlsx` (browser only — SSL blocks automated download)
2. Place in `data_files/`
3. Run: `python enrich_with_odds.py --year YYYY`
4. Run: `python features.py` to rebuild the feature matrix

This is intentionally **not automated in CI** because tennis-data.co.uk only publishes the full-year file after the season ends.

---

## 3. Matchstat RapidAPI (live, 2026 onward)

**Host:** `tennis-api-atp-wta-itf.p.rapidapi.com`  
**Plan:** BASIC ($0/month, 500 requests/month)  
**Key storage:** `.env` → `RAPIDAPI_KEY` and `.streamlit/secrets.toml` → `RAPIDAPI_KEY`  
**Client:** `matchstat_api.py`

### What's included

- Today's and tomorrow's fixtures (ATP, WTA, ITF, Challengers)
- Pre-match decimal odds: `k1`/`k2` (moneyline), `total`/`ktm`/`ktb` (totals), `f1`/`kf1`/`kf2` (handicap)
- H2H match history between any two players
- Player profiles, rankings, current-event stats

### Budget management (500 calls/month)

| Use | Calls/day | Calls/month |
|---|---|---|
| Today's fixtures (cached per calendar day) | 1 | ~30 |
| Tomorrow's fixtures (cached per calendar day) | 1 | ~30 |
| H2H lookups (cached permanently per pair) | 0–10 | ~100 |
| **Reserve** | — | **~340** |

- No call is made if there are no upcoming singles matches (`has_upcoming_matches()` guard)
- Fixture responses cached to `cache/matchstat/fixtures_atp_YYYY-MM-DD.json`
- H2H responses cached to `cache/matchstat/h2h_{id_lo}_{id_hi}.json` until invalidated

### Why this replaces The Odds API for tennis

Matchstat provides richer pre-match odds (moneyline + totals + handicap + set score) at no cost, scoped to tennis only, and also covers Challengers and ITF. The Odds API key is preserved in `.env` for use in other projects.

---

## 4. Gaps & Limitations

| Gap | Impact | Mitigation |
|---|---|---|
| No odds before 2020 | Market-implied probability unavailable before 2020 | ELO and rank features cover full history |
| Challenger odds absent from td.co.uk | No historical odds for challenger training rows | Matchstat covers challengers live from 2026 |
| No 2026 historical odds in td.co.uk yet | 2026 training rows have no bookmaker odds column | Null feature; model handles missing values |
| tennis-data.co.uk SSL blocks automation | Cannot auto-download new xlsx in CI | Annual manual download process documented above |
| ~18.5% unmatched rows in historical odds | Those rows have null odds features | Excluded from odds-dependent model paths |

---

## 5. Automated Pipeline Timing

```
05:00 UTC daily  (GitHub Actions — .github/workflows/update_data.yml)
  │
  ├── update_tml_data.py      re-downloads 2026.csv, 2026_challenger.csv, ongoing_tourneys.csv
  ├── features.py             recomputes rolling ELO, serve stats, recent form
  │                           → data_files/features_2020_present.parquet
  └── git commit + push       only if any file changed

Streamlit app — on page load
  │
  └── matchstat_api.py        reads today's fixture cache (or fetches once if stale)
                              returns upcoming matches + pre-match odds
```

---

## 6. File Inventory

```
tml-data/
  1968.csv … 2026.csv                        main tour by year (59 files)
  1978_challenger.csv … 2026_challenger.csv  challengers by year (47 files)
  ongoing_tourneys.csv                       live in-progress matches
  ATP_Database.csv                           full historical union
  2020_with_odds.csv … 2025_with_odds.csv    TML + td.co.uk odds per year
  atp_2020_2025_with_odds.csv                combined training set (16,134 rows, 60 cols)

data_files/
  2020.xlsx … 2025.xlsx                      raw tennis-data.co.uk files
  features_2020_present.parquet              output of features.py; read by predictions.py

cache/                                       gitignored
  matchstat/
    fixtures_atp_YYYY-MM-DD.json             cached daily fixture + odds response
    h2h_{id_lo}_{id_hi}.json                 cached H2H history per player pair
    _usage.json                              monthly API call counter
```
