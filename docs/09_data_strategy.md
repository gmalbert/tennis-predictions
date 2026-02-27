# Data Pulling Strategy

## TL;DR

| Layer | Source | Years | Cost | Script |
|---|---|---|---|---|
| Main tour historical | tennis-data.co.uk Excel | 2000–2025 | Free | `ingest_tennis_data_co_uk.py` |
| Challengers/Futures historical | Jeff Sackmann ATP repo | 1968–2024 | Free | `git pull` |
| Serve statistics (historical) | Jeff Sackmann ATP repo | 1991–2024 | Free | `git pull` |
| Player metadata | Sackmann `atp_players.csv` | All time | Free | bundled |
| Live / upcoming schedule | Flashscore live feed | Today | Free | `tennis_schedule.py` |
| Pre-match betting odds | The Odds API | Rolling window | Freemium | `odds_api.py` |

---

## 1. Why Each Source Was Chosen

### tennis-data.co.uk (primary for main tour)

**What it has that nothing else free has:**
- Complete ATP main-tour results from 2000 to present season
- Betting odds from Bet365, Pinnacle, market Max, market Average, and Betfair Exchange
- Set-by-set scores, round, court type (indoor/outdoor), surface
- Rankings and ranking points at match time

**What it lacks:**
- Serve statistics (aces, DFs, break points)
- Player physical attributes (height, hand, IOC) — partially backfilled from Sackmann
- Challenger / Futures / ITF events
- Player IDs — backfilled via name matching against `atp_players.csv`

**URL pattern:**  
ATP men: `https://www.tennis-data.co.uk/{year}/{year}.xlsx`  
WTA women: `https://www.tennis-data.co.uk/{year}w/{year}.xlsx`

**Download note:**  
The site uses an older TLS stack that may conflict with Windows security policy.
If automated download fails, manually open the URL in a browser and save to `data_files/{year}.xlsx`.
The script detects local files automatically on re-run.

---

### Jeff Sackmann tennis_atp repo (primary for serve stats and Challengers)

**Location:** `tennis_atp/` (cloned from https://github.com/JeffSackmann/tennis_atp)

**What it has:**
- Full match results with serve statistics (aces, double faults, break points, etc.) from ~1991
- Challenger and Futures match results
- Player registry with IDs, hand, height, DOB, IOC code
- Draw sizes, seeds, entry types

**What it lacks:**
- Betting odds
- 2025 data — as of early 2026, the last commit is "2024 season". The 2025 file does not yet exist.
  Update schedule: Sackmann typically publishes the full year file ~January of the following year.

**Key files:**
```
tennis_atp/
  atp_matches_{year}.csv          # main tour, 1968–2024
  atp_matches_qual_chall_{year}.csv  # qualifiers + challengers, recent years
  atp_matches_futures_{year}.csv  # futures, recent years
  atp_players.csv                 # all-time player registry
  atp_rankings_current.csv        # current ATP live ranking
  atp_rankings_{decade}.csv       # historical rankings by decade
```

---

### The Odds API (live betting lines)

Used by `odds_api.py` to fetch upcoming match odds in real time. Requires `ODDS_API_KEY` env var.
Free tier: 500 requests/month. Paid tiers available.
Use for: model calibration, value-bet identification, live display in the Streamlit app.

---

### Flashscore live feed (today's schedule)

Used by `tennis_schedule.py`. Only reliable for today's schedule and live scores.
Historical date-feed URLs do not work — all days are blocked.

---

## 2. Best Cross-Section of Data for Model Training

### For match outcome prediction (win/loss):
Combine tennis-data.co.uk (2000–2025) with Sackmann metadata backfill:
```
data_files/td_atp_2000.csv ... data_files/td_atp_2025.csv
```
This gives ~60,000 main-tour matches with:
- Surface, round, tournament level
- Rankings and ranking points at match time
- Head-to-head deducible from name/ID
- Historical betting odds as a calibration signal

### For serve-quality / in-match metrics:
Use Sackmann main-tour CSVs (1991–2024):
```
tennis_atp/atp_matches_{year}.csv
```
This adds aces, DFs, 1stIn%, 1stWon%, 2ndWon%, BPSaved, BPFaced for both players.
Cross-join with tennis-data.co.uk on (winner_name + loser_name + approximate date) to get both
serve stats AND odds in the same row.

### Recommended merged training set:
1. Start with tennis-data.co.uk for 2000–2025 (result + odds + rankings + set scores)
2. Left-join Sackmann serve stats on matching match keys
3. Fill ~70% of rows with serve stats (availability depends on year and tier)
4. For 2025 specifically: td_atp_2025.csv exists; Sackmann 2025 will be released ~Jan 2026

---

## 3. Data Pulling Playbook

### Initial setup (one-time)

```bash
# 1. Clone Sackmann repo (already done)
git clone https://github.com/JeffSackmann/tennis_atp.git tennis_atp

# 2. Download and convert tennis-data.co.uk for the years you want
#    If automated download fails, manually save each file as data_files/{year}.xlsx
python ingest_tennis_data_co_uk.py --year-range 2010 2025

# 3. Verify output
ls data_files/td_atp_*.csv
```

### Weekly update (during the season)

```bash
# 1. Pull latest Sackmann data (Challengers + serve stats get updated mid-season occasionally)
cd tennis_atp && git pull origin master && cd ..

# 2. Re-download current year Excel from tennis-data.co.uk
#    Delete the old local file first so download is triggered
del data_files\2025.xlsx
python ingest_tennis_data_co_uk.py --year 2025

# 3. (Optional) Refresh live odds via The Odds API — handled by the app on each load
```

### After Sackmann publishes 2025 (expected Jan 2026)

```bash
cd tennis_atp && git pull origin master && cd ..
# tennis_atp/atp_matches_2025.csv will now exist with full serve stats
# Run merge to combine with td_atp_2025.csv
python merge_2025_data.py
```

---

## 4. Scraper Status Summary

All four custom scrapers were built and tested. None are currently viable for production:

| Scraper | Status | Reason | Viable? |
|---|---|---|---|
| `scraper_atp.py` | Failed | ATP site uses React SPA; guessed XHR endpoints wrong | Fix with browser devtools |
| `scraper_tennis_abstract.py` | Partial | Rate-limited (429) after ~3 requests; player list sorted wrong (now fixed). Would take ~30 min for all active players | Viable but slow — run overnight |
| `scraper_flashscore.py` | Failed | Historical date-feed URL does not work; Cloudflare blocks all requests | Not viable for historical |
| `scraper_itf.py` | Failed | Guessed API host `api.itftennis.com` doesn't resolve | Needs browser devtools |

**Recommendation:** Don't invest more in custom scrapers until tennis-data.co.uk + Sackmann gap is actually a problem (i.e., when you need live 2025 Challenger results not yet in either source). At that point, Tennis Abstract scraper is the most fixable path.

---

## 5. The Only Genuine Data Gap

**2025 Challenger / Future results** — the one thing no free structured source covers right now:

- tennis-data.co.uk: main tour only
- Sackmann: 2025 file not published yet
- All scrapers: blocked or broken

Options:
1. **Accept the gap** — if your model only needs to predict main-tour matches, this isn't an issue
2. **Wait for Sackmann** — 2025 file expected ~Jan 2026; until then, 2024 challenger data is available
3. **Tennis Abstract scraper** — fix the rate-limiting (e.g., run with `--max-players 200` overnight with 4s delay)
4. **Sofascore API** (undocumented) — community projects like `sofascore-api` reverse-engineered their internal API; worth investigating if Challenger coverage is required

---

## 6. Feature Engineering Implications

Given the data cross-section above, these features are reliably available for modelling:

**Always available (tennis-data.co.uk):**
- `surface`, `tourney_level`, `round`, `best_of`
- `winner_rank`, `loser_rank`, `winner_rank_points`, `loser_rank_points`
- `rank_diff` = winner_rank − loser_rank (strong predictor)
- `rank_ratio` = winner_rank / loser_rank
- Betting market implied probability from B365/Pinnacle/Avg odds
- Set-count features from reconstructed score

**Available 1991–2024 via Sackmann (backfill onto td_ rows where matched):**
- First-serve percentage, break-point conversion rate
- Ace rate, double-fault rate
- `w_df`, `w_ace`, `l_df`, `l_ace` — serve aggressiveness signals

**Available via player registry backfill:**
- Playing hand (R/L/U)
- Height differential
- IOC / nationality

**Not available in free data (would require paid API):**
- Real-time form / last 5 match win rate (must compute from your own historical data)
- Head-to-head record on specific surface (compute from your CSVs)
- ATP live ranking (available in `atp_rankings_current.csv`)
