"""
features.py
-----------
Build the feature matrix used for model training and live prediction.

Input:  tml-data/YYYY.csv (all years, for ELO seeding)
        tml-data/atp_2020_2025_with_odds.csv (historical odds)
        tml-data/2026.csv (current year, no odds yet)
Output: data_files/features_2020_present.parquet  (and .csv fallback)

Run:    python features.py
        python features.py --csv   (also write CSV)

Design notes
------------
- ELO is computed from 1968 onward so that 2020 values are properly seeded,
  but only rows with tourney_date >= 2020-01-01 are written to the output.
- All rolling statistics look back only over completed past matches — no leakage.
- Each output row represents one match in winner/loser orientation.
  Downstream models should randomise perspective (swap w/l with 50% probability)
  to remove the "winner first" bias.
"""

import argparse
import os
import glob
from collections import defaultdict

import numpy as np
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_START  = pd.Timestamp("2020-01-01")
TML_DIR      = "tml-data"
OUT_DIR      = "data_files"
OUT_PARQUET  = os.path.join(OUT_DIR, "features_2020_present.parquet")
OUT_CSV      = os.path.join(OUT_DIR, "features_2020_present.csv")

ELO_K           = 32
ELO_INIT        = 1500
ROLLING_WINDOW  = 20    # matches for recent form
SURFACE_WINDOW  = 20    # matches per surface for surface form

SURFACES = ["Hard", "Clay", "Grass", "Carpet"]

# Feature columns used for model training (signed diffs — winner perspective)
FEATURE_COLS = [
    "elo_diff",
    "surface_elo_diff",
    "total_elo_diff",
    "h2h_diff",
    "form_diff",
    "rank_diff",
    "height_diff",
    "age_diff",
]


# ── 1. Load & clean all TML main-tour data ────────────────────────────────────

def _load_tml_all() -> pd.DataFrame:
    """Load all annual main-tour TML CSVs from 1968 onwards."""
    files = sorted(glob.glob(os.path.join(TML_DIR, "[0-9][0-9][0-9][0-9].csv")))
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
            dfs.append(df)
        except Exception as e:
            print(f"  [warn] could not read {f}: {e}")
    combined = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(combined):,} rows from {len(files)} TML main-tour files")
    return combined


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Remove walkovers, retirements, and rows missing critical fields."""
    n0 = len(df)

    # Remove abandoned/incomplete matches
    bad_score = df["score"].str.contains(
        r"W/O|RET|DEF|Walkover|ABN|UNF", case=False, na=False
    )
    df = df[~bad_score].copy()

    # Require player IDs, surface, tourney_date
    df = df.dropna(subset=["winner_id", "loser_id", "surface", "tourney_date"])

    # Coerce types — player IDs are alphanumeric strings (e.g. "CD85"), keep as str
    df["winner_id"] = df["winner_id"].astype(str).str.strip()
    df["loser_id"]  = df["loser_id"].astype(str).str.strip()
    # Drop rows whose IDs look like NaN after string conversion
    df = df[~df["winner_id"].isin(["nan", "", "None"]) &
            ~df["loser_id"].isin(["nan", "", "None"])]
    df["tourney_date"] = pd.to_datetime(df["tourney_date"].astype(str), format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["tourney_date"])
    # Normalise match_num to float for consistent join with odds file
    df["match_num"] = pd.to_numeric(df["match_num"], errors="coerce")

    # Sort chronologically – critical for correct rolling computation
    df = df.sort_values(["tourney_date", "tourney_id", "match_num"]).reset_index(drop=True)

    print(f"After cleaning: {len(df):,} rows (removed {n0-len(df):,})")
    return df


# ── 2. ELO ────────────────────────────────────────────────────────────────────

def _elo_expected(r_a: float, r_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400))


class EloRatingSystem:
    """
    ELO rating system with overall and surface-specific ratings.
    Player IDs are treated as strings (e.g. "CD85").
    """

    def __init__(self, k_factor: int = ELO_K, initial_rating: int = ELO_INIT):
        self.k_factor       = k_factor
        self.initial_rating = initial_rating
        self.ratings: dict[str, float] = {}
        self.surface_ratings: dict[str, dict[str, float]] = {s: {} for s in SURFACES}
        self.rating_history: dict[str, list[tuple[str, float]]] = {}

    def expected_score(self, player: float, opponent: float) -> float:
        return 1.0 / (1.0 + 10 ** ((opponent - player) / 400))

    def update_ratings(
        self, winner_id: str, loser_id: str, surface: str, date: str
    ) -> tuple[float, float, float, float]:
        """
        Update ratings after a match.
        Returns (winner_elo_before, loser_elo_before,
                 winner_surface_elo_before, loser_surface_elo_before).
        """
        w_elo = self.ratings.get(winner_id, self.initial_rating)
        l_elo = self.ratings.get(loser_id,  self.initial_rating)
        exp_w = self.expected_score(w_elo, l_elo)
        delta = self.k_factor * (1.0 - exp_w)
        self.ratings[winner_id] = w_elo + delta
        self.ratings[loser_id]  = l_elo - delta

        surf = surface if surface in self.surface_ratings else "Hard"
        w_surf = self.surface_ratings[surf].get(winner_id, self.initial_rating)
        l_surf = self.surface_ratings[surf].get(loser_id,  self.initial_rating)
        exp_ws = self.expected_score(w_surf, l_surf)
        delta_s = self.k_factor * (1.0 - exp_ws)
        self.surface_ratings[surf][winner_id] = w_surf + delta_s
        self.surface_ratings[surf][loser_id]  = l_surf - delta_s

        self.rating_history.setdefault(winner_id, []).append((date, self.ratings[winner_id]))
        self.rating_history.setdefault(loser_id,  []).append((date, self.ratings[loser_id]))
        return w_elo, l_elo, w_surf, l_surf

    def get_rating(self, player_id: str) -> float:
        return self.ratings.get(player_id, self.initial_rating)

    def get_surface_rating(self, player_id: str, surface: str) -> float:
        surf = surface if surface in self.surface_ratings else "Hard"
        return self.surface_ratings[surf].get(player_id, self.initial_rating)

    def get_top_players(self, n: int = 20) -> list[tuple[str, float]]:
        return sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)[:n]


class FeatureEngineer:
    """
    Stateful feature engineer for online / live inference.
    Call update() for each completed match (in chronological order),
    then get_prediction_features() for a future match.
    """

    def __init__(self):
        self.elo_system    = EloRatingSystem()
        self.h2h: dict[tuple[str, str], int]   = {}
        self.recent_matches: dict[str, list[int]] = {}
        self.recent_window = ROLLING_WINDOW

    def update(self, winner_id: str, loser_id: str, surface: str, date: str) -> dict:
        """Record one completed match; return its pre-match feature dict."""
        w_elo, l_elo, w_surf, l_surf = self.elo_system.update_ratings(
            winner_id, loser_id, surface, date
        )
        h2h_w = self.h2h.get((winner_id, loser_id), 0)
        h2h_l = self.h2h.get((loser_id, winner_id), 0)
        self.h2h[(winner_id, loser_id)] = h2h_w + 1

        w_recent = self.recent_matches.get(winner_id, [])[-self.recent_window:]
        l_recent = self.recent_matches.get(loser_id,  [])[-self.recent_window:]
        w_form   = sum(w_recent) / len(w_recent) if w_recent else 0.5
        l_form   = sum(l_recent) / len(l_recent) if l_recent else 0.5
        self.recent_matches.setdefault(winner_id, []).append(1)
        self.recent_matches.setdefault(loser_id,  []).append(0)

        return {
            "winner_elo": w_elo, "loser_elo": l_elo,
            "elo_diff": w_elo - l_elo,
            "winner_surface_elo": w_surf, "loser_surface_elo": l_surf,
            "surface_elo_diff": w_surf - l_surf,
            "total_elo_diff": (w_elo + w_surf) - (l_elo + l_surf),
            "h2h_winner": h2h_w, "h2h_loser": h2h_l,
            "h2h_diff": h2h_w - h2h_l,
            "winner_form": w_form, "loser_form": l_form,
            "form_diff": w_form - l_form,
        }

    def get_prediction_features(self, p1_id: str, p2_id: str, surface: str) -> dict:
        """Return feature vector for a *future* match (no outcome known yet)."""
        elo1  = self.elo_system.get_rating(p1_id)
        elo2  = self.elo_system.get_rating(p2_id)
        s1    = self.elo_system.get_surface_rating(p1_id, surface)
        s2    = self.elo_system.get_surface_rating(p2_id, surface)
        h2h1  = self.h2h.get((p1_id, p2_id), 0)
        h2h2  = self.h2h.get((p2_id, p1_id), 0)
        r1    = self.recent_matches.get(p1_id, [])[-self.recent_window:]
        r2    = self.recent_matches.get(p2_id, [])[-self.recent_window:]
        f1    = sum(r1) / len(r1) if r1 else 0.5
        f2    = sum(r2) / len(r2) if r2 else 0.5
        return {
            "elo_diff":         elo1 - elo2,
            "surface_elo_diff": s1   - s2,
            "total_elo_diff":   (elo1 + s1) - (elo2 + s2),
            "h2h_diff":         h2h1 - h2h2,
            "form_diff":        f1   - f2,
        }

    def get_elo_system(self) -> EloRatingSystem:
        return self.elo_system


def _compute_elo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute overall and surface-specific ELO for each row.
    Returns df with new columns:
        elo_pre_w, elo_pre_l, elo_surf_pre_w, elo_surf_pre_l
    """
    ratings: dict[str, float] = defaultdict(lambda: float(ELO_INIT))
    surf_ratings: dict[str, dict[str, float]] = {
        s: defaultdict(lambda: float(ELO_INIT)) for s in SURFACES
    }

    elo_pre_w, elo_pre_l = [], []
    elo_surf_pre_w, elo_surf_pre_l = [], []

    for row in df.itertuples(index=False):
        w, l = row.winner_id, row.loser_id
        surf = str(row.surface).strip().capitalize()
        if surf not in SURFACES:
            surf = "Hard"  # default

        # Record pre-match ratings
        rw = ratings[w]
        rl = ratings[l]
        rw_s = surf_ratings[surf][w]
        rl_s = surf_ratings[surf][l]

        elo_pre_w.append(rw)
        elo_pre_l.append(rl)
        elo_surf_pre_w.append(rw_s)
        elo_surf_pre_l.append(rl_s)

        # Update overall ELO
        exp_w = _elo_expected(rw, rl)
        delta = ELO_K * (1.0 - exp_w)
        ratings[w] = rw + delta
        ratings[l] = rl - delta

        # Update surface ELO
        exp_ws = _elo_expected(rw_s, rl_s)
        delta_s = ELO_K * (1.0 - exp_ws)
        surf_ratings[surf][w] = rw_s + delta_s
        surf_ratings[surf][l] = rl_s - delta_s

    df = df.copy()
    df["elo_pre_w"]      = elo_pre_w
    df["elo_pre_l"]      = elo_pre_l
    df["elo_surf_pre_w"] = elo_surf_pre_w
    df["elo_surf_pre_l"] = elo_surf_pre_l
    print("ELO computed.")
    return df


# ── 3. Rolling per-player stats ───────────────────────────────────────────────

def _safe_pct(num, denom):
    """Return num/denom as a percentage, or NaN if denom == 0."""
    if denom and denom > 0:
        return 100.0 * num / denom
    return np.nan


def _rolling_serve(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each match, compute the rolling average serve stats
    (1stIn%, 1stWon%, 2ndWon%, bpSaved%) over the last ROLLING_WINDOW matches
    for both winner and loser.
    """
    # We need per-player rolling stats.
    # Build a per-player match history as we iterate chronologically.
    player_recent: dict[str, list[dict]] = defaultdict(list)

    def _avg(pid: int, key: str) -> float:
        vals = [m[key] for m in player_recent[pid] if m[key] is not np.nan]
        return float(np.mean(vals)) if vals else np.nan

    cols = {
        "serve_1stIn_w":   [], "serve_1stIn_l":   [],
        "serve_1stWon_w":  [], "serve_1stWon_l":  [],
        "serve_2ndWon_w":  [], "serve_2ndWon_l":  [],
        "serve_bpSaved_w": [], "serve_bpSaved_l": [],
        "recent_win_rate_w": [], "recent_win_rate_l": [],
    }

    for row in df.itertuples(index=False):
        w, l = row.winner_id, row.loser_id

        # Snapshot averages BEFORE this match
        for suffix, pid in [("w", w), ("l", l)]:
            history = player_recent[pid][-ROLLING_WINDOW:]
            if history:
                wins   = sum(1 for m in history if m["won"])
                wrate  = wins / len(history)
                vals_1in  = [m["1stIn"]  for m in history if m["1stIn"]  is not np.nan and not np.isnan(m["1stIn"])]
                vals_1won = [m["1stWon"] for m in history if m["1stWon"] is not np.nan and not np.isnan(m["1stWon"])]
                vals_2won = [m["2ndWon"] for m in history if m["2ndWon"] is not np.nan and not np.isnan(m["2ndWon"])]
                vals_bp   = [m["bpSaved"] for m in history if m["bpSaved"] is not np.nan and not np.isnan(m["bpSaved"])]
                avg1in  = float(np.mean(vals_1in))  if vals_1in  else np.nan
                avg1won = float(np.mean(vals_1won)) if vals_1won else np.nan
                avg2won = float(np.mean(vals_2won)) if vals_2won else np.nan
                avgbp   = float(np.mean(vals_bp))   if vals_bp   else np.nan
            else:
                wrate = avg1in = avg1won = avg2won = avgbp = np.nan

            cols[f"recent_win_rate_{suffix}"].append(wrate)
            cols[f"serve_1stIn_{suffix}"].append(avg1in)
            cols[f"serve_1stWon_{suffix}"].append(avg1won)
            cols[f"serve_2ndWon_{suffix}"].append(avg2won)
            cols[f"serve_bpSaved_{suffix}"].append(avgbp)

        # Append this match to each player's history
        def _srv_pct(svpt_col, won_col, row):
            svpt = getattr(row, svpt_col, None)
            won  = getattr(row, won_col, None)
            if svpt and won and svpt > 0:
                return 100.0 * won / svpt
            return np.nan

        def _1stIn_pct(row, prefix):
            svpt = getattr(row, f"{prefix}_svpt", None)
            inn  = getattr(row, f"{prefix}_1stIn", None)
            if svpt and inn and svpt > 0:
                return 100.0 * inn / svpt
            return np.nan

        w_rec = {
            "won":     True,
            "1stIn":   _1stIn_pct(row, "w"),
            "1stWon":  _srv_pct("w_1stIn", "w_1stWon", row),
            "2ndWon":  _srv_pct("w_svpt",  "w_2ndWon",  row),
            "bpSaved": _safe_pct(getattr(row, "w_bpSaved", None),
                                  getattr(row, "w_bpFaced", None)),
        }
        l_rec = {
            "won":     False,
            "1stIn":   _1stIn_pct(row, "l"),
            "1stWon":  _srv_pct("l_1stIn", "l_1stWon", row),
            "2ndWon":  _srv_pct("l_svpt",  "l_2ndWon",  row),
            "bpSaved": _safe_pct(getattr(row, "l_bpSaved", None),
                                  getattr(row, "l_bpFaced", None)),
        }

        player_recent[w].append(w_rec)
        player_recent[l].append(l_rec)
        # Keep only the most recent window to bound memory
        if len(player_recent[w]) > ROLLING_WINDOW * 2:
            player_recent[w] = player_recent[w][-ROLLING_WINDOW:]
        if len(player_recent[l]) > ROLLING_WINDOW * 2:
            player_recent[l] = player_recent[l][-ROLLING_WINDOW:]

    df = df.copy()
    for col, vals in cols.items():
        df[col] = vals
    print("Rolling serve stats computed.")
    return df


# ── 4. Surface win rate ────────────────────────────────────────────────────────

def _surface_win_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Win rate on the specific surface of this match, last SURFACE_WINDOW appearances."""
    surf_history: dict[tuple[str, str], list[bool]] = defaultdict(list)

    swr_w, swr_l = [], []
    for row in df.itertuples(index=False):
        w, l = row.winner_id, row.loser_id
        surf = str(row.surface).strip().capitalize()

        wh = surf_history[(w, surf)][-SURFACE_WINDOW:]
        lh = surf_history[(l, surf)][-SURFACE_WINDOW:]

        swr_w.append(sum(wh) / len(wh) if wh else np.nan)
        swr_l.append(sum(lh) / len(lh) if lh else np.nan)

        surf_history[(w, surf)].append(True)
        surf_history[(l, surf)].append(False)

    df = df.copy()
    df["surface_win_rate_w"] = swr_w
    df["surface_win_rate_l"] = swr_l
    print("Surface win rates computed.")
    return df


# ── 5. H2H ────────────────────────────────────────────────────────────────────

def _h2h(df: pd.DataFrame) -> pd.DataFrame:
    """Head-to-head wins for both players in each match, up to but not including the match."""
    h2h_counts: dict[tuple, int] = defaultdict(int)

    h2h_w, h2h_l = [], []
    for row in df.itertuples(index=False):
        w, l    = str(row.winner_id), str(row.loser_id)
        lo, hi  = (w, l) if w < l else (l, w)  # lexicographic canonical order
        key_wl  = (lo, hi, w)   # canonical pair + winner
        key_lw  = (lo, hi, l)

        h2h_w.append(h2h_counts[key_wl])
        h2h_l.append(h2h_counts[key_lw])

        h2h_counts[key_wl] += 1

    df = df.copy()
    df["h2h_wins_w"] = h2h_w
    df["h2h_wins_l"] = h2h_l
    print("H2H counts computed.")
    return df


# ── 6. Market implied probability ─────────────────────────────────────────────

def _market_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive market implied probabilities from AvgW / AvgL (Oddsportal average).
    Uses Pinnacle (PSW/PSL) as fallback, then B365.
    Result: mkt_prob_w in [0,1], representing the market's win probability for
    the winner.
    """
    def _implied_prob(row, w_col, l_col):
        w_odds = getattr(row, w_col, None)
        l_odds = getattr(row, l_col, None)
        try:
            w_odds = float(w_odds)
            l_odds = float(l_odds)
            if w_odds > 1 and l_odds > 1:
                raw_w = 1.0 / w_odds
                raw_l = 1.0 / l_odds
                total = raw_w + raw_l  # normalise to remove overround
                return raw_w / total
        except (TypeError, ValueError):
            pass
        return np.nan

    probs = []
    for row in df.itertuples(index=False):
        p = _implied_prob(row, "AvgW", "AvgL")
        if np.isnan(p):
            p = _implied_prob(row, "PSW", "PSL")
        if np.isnan(p):
            p = _implied_prob(row, "B365W", "B365L")
        probs.append(p)

    df = df.copy()
    df["mkt_prob_w"] = probs
    filled = sum(1 for p in probs if not np.isnan(p))
    print(f"Market prob: {filled:,}/{len(df):,} rows filled ({filled/len(df)*100:.1f}%)")
    return df


# ── 7. Rank features ──────────────────────────────────────────────────────────

def _rank_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    wr = pd.to_numeric(df["winner_rank"], errors="coerce")
    lr = pd.to_numeric(df["loser_rank"],  errors="coerce")
    df["rank_diff"]  = lr - wr       # positive = winner ranked higher (better)
    df["rank_ratio"] = wr / lr.replace(0, np.nan)
    return df


# ── 8. Encode categoricals ────────────────────────────────────────────────────

LEVEL_MAP = {"G": 4, "M": 3, "A": 2, "B": 1, "C": 0, "D": 0, "F": 2, "O": 1}
ROUND_MAP = {
    "F": 7, "SF": 6, "QF": 5, "R16": 4, "R32": 3,
    "R64": 2, "R128": 1, "RR": 3, "BR": 4,
}
SURFACE_MAP = {"Hard": 0, "Clay": 1, "Grass": 2, "Carpet": 3}

def _encode(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["tourney_level_enc"] = df["tourney_level"].map(LEVEL_MAP).fillna(1).astype(int)
    df["surface_enc"]       = df["surface"].str.strip().str.capitalize().map(SURFACE_MAP).fillna(0).astype(int)
    df["round_enc"]         = df["round"].map(ROUND_MAP).fillna(2).astype(int)
    df["best_of"]           = pd.to_numeric(df.get("best_of", 3), errors="coerce").fillna(3).astype(int)
    return df


# ── 9. Diff features (FEATURE_COLS) ───────────────────────────────────────────

def _diff_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the 8 signed diff features that make up FEATURE_COLS.
    rank_diff is already computed by _rank_features() with the correct sign;
    this function adds the remaining 7 composite diffs.
    """
    df = df.copy()
    df["elo_diff"]         = df["elo_pre_w"]      - df["elo_pre_l"]
    df["surface_elo_diff"] = df["elo_surf_pre_w"] - df["elo_surf_pre_l"]
    df["total_elo_diff"]   = (
        (df["elo_pre_w"] + df["elo_surf_pre_w"])
        - (df["elo_pre_l"] + df["elo_surf_pre_l"])
    )
    df["h2h_diff"]  = df["h2h_wins_w"] - df["h2h_wins_l"]
    df["form_diff"] = df["recent_win_rate_w"] - df["recent_win_rate_l"]
    df["height_diff"] = (
        pd.to_numeric(df.get("winner_ht", np.nan), errors="coerce")
        - pd.to_numeric(df.get("loser_ht",  np.nan), errors="coerce")
    )
    df["age_diff"] = (
        pd.to_numeric(df.get("winner_age", np.nan), errors="coerce")
        - pd.to_numeric(df.get("loser_age",  np.nan), errors="coerce")
    )
    return df


# ── 10. Fatigue index ─────────────────────────────────────────────────────────

def _fatigue(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute fatigue features for winner and loser *before* each match:
      - days_since_last_w / _l : days elapsed since that player's previous match
      - matches_14d_w / _l     : number of matches played in the preceding 14 days
    """
    last_date: dict[str, pd.Timestamp] = {}
    match_dates: dict[str, list[pd.Timestamp]] = defaultdict(list)

    days_since_w, days_since_l = [], []
    m14d_w, m14d_l = [], []

    for row in df.itertuples(index=False):
        w, l   = row.winner_id, row.loser_id
        d      = row.tourney_date
        cutoff = d - pd.Timedelta(days=14)

        # Record BEFORE updating state
        dsw = int((d - last_date[w]).days) if w in last_date else np.nan
        dsl = int((d - last_date[l]).days) if l in last_date else np.nan
        mw  = sum(1 for x in match_dates[w] if x >= cutoff)
        ml  = sum(1 for x in match_dates[l] if x >= cutoff)

        days_since_w.append(dsw)
        days_since_l.append(dsl)
        m14d_w.append(mw)
        m14d_l.append(ml)

        last_date[w] = d
        last_date[l] = d
        match_dates[w].append(d)
        match_dates[l].append(d)

    df = df.copy()
    df["days_since_last_w"] = days_since_w
    df["days_since_last_l"] = days_since_l
    df["matches_14d_w"]     = m14d_w
    df["matches_14d_l"]     = m14d_l
    print("Fatigue features computed.")
    return df


# ── 11. Momentum (streak) ─────────────────────────────────────────────────────

def _momentum(df: pd.DataFrame) -> pd.DataFrame:
    """
    Current consecutive win/loss streak before each match.
    streak_w > 0 : winner is on a winning run of that length.
    streak_l < 0 : loser is on a losing run of that magnitude.
    """
    streak: dict[str, int] = defaultdict(int)

    streak_w, streak_l = [], []
    for row in df.itertuples(index=False):
        w, l = row.winner_id, row.loser_id
        streak_w.append(streak[w])
        streak_l.append(streak[l])
        # After recording, update: winner extends positive run, loser extends negative
        streak[w] = max(streak[w], 0) + 1
        streak[l] = min(streak[l], 0) - 1

    df = df.copy()
    df["streak_w"] = streak_w
    df["streak_l"] = streak_l
    print("Momentum (streak) computed.")
    return df


# ── 12. Assemble output columns ───────────────────────────────────────────────

OUTPUT_COLS = [
    # Identity
    "tourney_id", "tourney_name", "tourney_date", "surface", "tourney_level",
    "round", "best_of", "match_num",
    # Encoded
    "tourney_level_enc", "surface_enc", "round_enc",
    # Players
    "winner_id", "winner_name", "winner_rank", "winner_age", "winner_ht", "winner_hand", "winner_ioc",
    "loser_id",  "loser_name",  "loser_rank",  "loser_age",  "loser_ht",  "loser_hand",  "loser_ioc",
    # ELO (raw pre-match)
    "elo_pre_w", "elo_pre_l", "elo_surf_pre_w", "elo_surf_pre_l",
    # FEATURE_COLS — signed diffs for model training
    "elo_diff", "surface_elo_diff", "total_elo_diff",
    "h2h_diff", "form_diff", "rank_diff", "height_diff", "age_diff",
    # Rank extras
    "rank_ratio",
    # Recent form (raw)
    "recent_win_rate_w", "recent_win_rate_l",
    "surface_win_rate_w", "surface_win_rate_l",
    # Serve stats
    "serve_1stIn_w",   "serve_1stIn_l",
    "serve_1stWon_w",  "serve_1stWon_l",
    "serve_2ndWon_w",  "serve_2ndWon_l",
    "serve_bpSaved_w", "serve_bpSaved_l",
    # H2H
    "h2h_wins_w", "h2h_wins_l",
    # Fatigue (extension)
    "days_since_last_w", "days_since_last_l",
    "matches_14d_w",     "matches_14d_l",
    # Momentum / streak (extension)
    "streak_w", "streak_l",
    # Market
    "mkt_prob_w",
    # Raw odds (kept for reference / model calibration)
    "B365W", "B365L", "PSW", "PSL", "AvgW", "AvgL",
    # Result (label for training — winner is always player_w by convention)
    "score", "winner_name",
]


# ── Main ──────────────────────────────────────────────────────────────────────

def build_features(write_csv: bool = False) -> pd.DataFrame:
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Load all TML data (for ELO seeding from 1968)
    raw = _load_tml_all()
    raw = _clean(raw)

    # ── Load odds for 2020–2025 and merge in
    odds_file = os.path.join(TML_DIR, "atp_2020_2025_with_odds.csv")
    if os.path.exists(odds_file):
        odds = pd.read_csv(odds_file, low_memory=False)
        odds_cols = ["B365W", "B365L", "PSW", "PSL", "AvgW", "AvgL",
                     "MaxW", "MaxL", "BFEW", "BFEL"]
        odds_cols = [c for c in odds_cols if c in odds.columns]
        odds["tourney_date_dt"] = pd.to_datetime(
            odds["tourney_date"].astype(str), format="%Y%m%d", errors="coerce"
        )
        # Use tourney_id + match_num as join key if available, else date+names
        merge_keys = ["tourney_id", "match_num"]
        odds_sub = odds[merge_keys + odds_cols].drop_duplicates(subset=merge_keys)
        raw = raw.merge(odds_sub, on=merge_keys, how="left")
        print(f"Odds merged: {raw['B365W'].notna().sum():,}/{len(raw):,} rows with Bet365 odds")
    else:
        print(f"[warn] {odds_file} not found — odds columns will be absent")
        for c in ["B365W", "B365L", "PSW", "PSL", "AvgW", "AvgL"]:
            raw[c] = np.nan

    # ── Compute all features on the full chronological sequence
    df = _compute_elo(raw)
    df = _rolling_serve(df)
    df = _surface_win_rate(df)
    df = _h2h(df)
    df = _rank_features(df)
    df = _market_features(df)
    df = _encode(df)
    df = _diff_features(df)
    df = _fatigue(df)
    df = _momentum(df)

    # ── Filter to 2020+
    df = df[df["tourney_date"] >= TRAIN_START].reset_index(drop=True)
    print(f"After filtering to 2020+: {len(df):,} rows")

    # ── Select output columns (only those present)
    out_cols = [c for c in OUTPUT_COLS if c in df.columns]
    # Deduplicate in case winner_name appears twice
    seen, deduped = set(), []
    for c in out_cols:
        if c not in seen:
            deduped.append(c)
            seen.add(c)
    out = df[deduped]

    # ── Write
    out.to_parquet(OUT_PARQUET, index=False)
    print(f"Written: {OUT_PARQUET}  ({len(out):,} rows × {len(out.columns)} cols)")

    if write_csv:
        out.to_csv(OUT_CSV, index=False)
        print(f"Written: {OUT_CSV}")

    return out


def prepare_training_data(
    feature_df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    test_year: int = 2025,
) -> tuple:
    """
    Split into train / test and produce a balanced label set.

    Because every row has the winner as 'player W', we randomly flip 50% of
    rows (negate all diffs) so the model cannot learn a positional bias.

    Parameters
    ----------
    feature_df  : output of build_features()
    feature_cols: columns to include in X (defaults to FEATURE_COLS)
    test_year   : first year held out as test set (default 2025)

    Returns
    -------
    X_train, y_train, X_test, y_test, train_df, test_df
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    # Drop rows missing any feature column
    valid = feature_df.dropna(subset=feature_cols)
    train_df = valid[valid["tourney_date"].dt.year <  test_year].copy()
    test_df  = valid[valid["tourney_date"].dt.year >= test_year].copy()

    def create_balanced(df: pd.DataFrame):
        rng   = np.random.default_rng(42)
        flips = rng.random(len(df)) < 0.5       # True  → keep winner-first  (y=1)
        rows  = df[feature_cols].to_numpy(dtype=float)
        X     = np.where(flips[:, None], rows, -rows)
        y     = flips.astype(int)
        return X, y

    X_train, y_train = create_balanced(train_df)
    X_test,  y_test  = create_balanced(test_df)

    print(
        f"Training set : {X_train.shape[0]:,} rows (before {test_year})\n"
        f"Test set     : {X_test.shape[0]:,} rows ({test_year} onward)\n"
        f"Features     : {feature_cols}"
    )
    return X_train, y_train, X_test, y_test, train_df, test_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build tennis feature matrix from TML data.")
    parser.add_argument("--csv", action="store_true", help="Also write CSV output")
    args = parser.parse_args()
    feat = build_features(write_csv=args.csv)
    print("\nSample (last 5 rows):")
    sample_cols = ["tourney_date", "tourney_name", "surface",
                   "winner_name", "loser_name",
                   "elo_pre_w", "elo_pre_l", "rank_diff", "mkt_prob_w"]
    print(feat[[c for c in sample_cols if c in feat.columns]].tail().to_string())
