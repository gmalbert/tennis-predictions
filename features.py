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
    df["rank_diff"]  = wr - lr       # negative = winner ranked higher (better)
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


# ── 9. Assemble output columns ────────────────────────────────────────────────

OUTPUT_COLS = [
    # Identity
    "tourney_id", "tourney_name", "tourney_date", "surface", "tourney_level",
    "round", "best_of", "match_num",
    # Encoded
    "tourney_level_enc", "surface_enc", "round_enc",
    # Players
    "winner_id", "winner_name", "winner_rank", "winner_age", "winner_hand", "winner_ioc",
    "loser_id",  "loser_name",  "loser_rank",  "loser_age",  "loser_hand",  "loser_ioc",
    # ELO
    "elo_pre_w", "elo_pre_l", "elo_surf_pre_w", "elo_surf_pre_l",
    # Rank
    "rank_diff", "rank_ratio",
    # Recent form
    "recent_win_rate_w", "recent_win_rate_l",
    "surface_win_rate_w", "surface_win_rate_l",
    # Serve stats
    "serve_1stIn_w",   "serve_1stIn_l",
    "serve_1stWon_w",  "serve_1stWon_l",
    "serve_2ndWon_w",  "serve_2ndWon_l",
    "serve_bpSaved_w", "serve_bpSaved_l",
    # H2H
    "h2h_wins_w", "h2h_wins_l",
    # Market
    "mkt_prob_w",
    # Raw odds (kept for reference / model calibration)
    "B365W", "B365L", "PSW", "PSL", "AvgW", "AvgL",
    # Result (label for training)
    "score", "winner_name",   # winner is always player_w by convention
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
