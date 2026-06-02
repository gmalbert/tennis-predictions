# Tennis Oracle — Next 5 Features to Implement

> **Based on:** Codebase gap analysis as of July 2025

---

## Feature 1: Surface-Specific XGBoost Models

**Why:** Clay, hard court, and grass require dramatically different skills — Djokovic on clay is very different from Djokovic on grass. A single model averages these differences away. Training separate classifiers per surface (or using surface as a strong interaction feature) would be the single biggest accuracy improvement available.

**How:**
1. Add `surface` column to the training dataset (already in the bet enrichment pipeline via `enrich_with_odds.py`)
2. Train three separate XGBoost classifiers in the model training script: `model_clay`, `model_hard`, `model_grass`
3. At prediction time, route each upcoming match to the correct surface model
4. Compare AUC per surface model vs the unified model in `pages/model_performance.py`

**Complexity:** Medium

---

## Feature 2: ATP/WTA Ranking Points Differential Feature

**Why:** The current Elo rating is derived from match results. Adding official ranking points differential as an additional feature (not a replacement) captures ATP/WTA consensus form in a way that Elo trails. The difference between #1 (15,000 pts) and #3 (6,000 pts) is far larger than the difference between #50 and #52.

**How:**
1. Fetch ATP/WTA rankings from `https://www.atptour.com/en/rankings` or the unofficial unofficial ATP rankings JSON endpoint
2. Compute `rank_pts_diff` = `home_rank_pts − away_rank_pts` per match
3. Add to feature vector in `utils.py` / model training script alongside existing Elo diff
4. Use `shift(1)` — apply rankings from the week before each match to prevent leakage

**Complexity:** Low

---

## Feature 3: Matchstat API Budget Tracker

**Why:** The Matchstat API is limited to 500 calls/month. There is currently no display of monthly usage. If the budget is exhausted, all live enrichment fails silently, degrading predictions without any user-facing warning.

**How:**
1. Create `utils/api_budget.py` with a simple SQLite (or JSON file) counter that increments on each Matchstat API call in `bzzoiro_api.py`
2. At startup, check calls used this month; if > 450, disable live enrichment and show `st.warning("API budget near limit")`
3. Display a progress bar in the sidebar: "Matchstat API: 347/500 calls used this month"
4. Add a monthly reset at the first day of each month (compare stored month vs `datetime.now().month`)

**Complexity:** Low

---

## Feature 4: Glicko-2 Rating Upgrade

**Why:** Glicko-2 adds rating deviation (RD) and volatility parameters to basic Elo. RD quantifies uncertainty about a player's true skill — a player returning from injury has high RD. This directly improves predictions for matches involving inactive or inconsistent players, which are common in tennis.

**How:**
1. Add the `glicko2` Python package to `requirements.txt`
2. Create `models/glicko2_ratings.py` with functions to build and update Glicko-2 ratings from match history
3. Replace the Elo computation in the pipeline with Glicko-2; keep Elo as a comparison baseline
4. Add `rating_deviation` and `volatility` as features in the match prediction model (high RD = uncertain player)

**Complexity:** High

---

## Feature 5: ITF Seeding for Debut ATP/WTA Players

**Why:** New ATP/WTA players have no historical data, causing cold-start prediction failures. `scraper_itf.py` already exists to fetch ITF circuit data. Using ITF results to seed initial Elo/Glicko ratings for debut players would eliminate the cold-start problem.

**How:**
1. Verify `scraper_itf.py` is functional and produces consistent player name matches
2. When a player appears for the first time in ATP/WTA data, look up their ITF rating/record
3. Compute an initial Elo seeding: convert ITF match record → starting Elo using a log-odds regression calibrated on ITF → ATP transition data
4. Store initial Elo in the player ratings table with `source = "itf"` flag

**Complexity:** Medium
