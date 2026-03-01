"""
compute_backlog.py
------------------
Pre-compute model probabilities for all historical matches and save to
data_files/prediction_backlog.parquet.

Run by the daily GitHub Action after features.py and train.py so the
Prediction Backlog tab loads instantly without running inference on-demand.

Reads:   data_files/features_2020_present.parquet
         data_files/tennis_predictor.pkl
Writes:  data_files/prediction_backlog.parquet

Usage:
    python compute_backlog.py               # score last 90 days (default)
    python compute_backlog.py --days 180    # last 180 days
    python compute_backlog.py --all         # score every match (slow)
"""

import argparse
import os

import numpy as np
import pandas as pd

DATA_DIR      = "data_files"
FEATURES_FILE = os.path.join(DATA_DIR, "features_2020_present.parquet")
MODEL_FILE    = os.path.join(DATA_DIR, "tennis_predictor.pkl")
OUTPUT_FILE   = os.path.join(DATA_DIR, "prediction_backlog.parquet")

# Columns to keep in the backlog (keeps the file lean)
KEEP_COLS = [
    "tourney_date", "tourney_name", "surface", "tourney_level", "round",
    "winner_name", "loser_name",
    "winner_rank", "loser_rank",
    "elo_pre_w", "elo_pre_l",
    "mkt_prob_w",
    "model_prob_w",     # filled by this script
]


def main(days: int | None = 90) -> None:
    print(f"[backlog] Loading features from {FEATURES_FILE}")
    if not os.path.exists(FEATURES_FILE):
        raise FileNotFoundError(f"Features file not found: {FEATURES_FILE}")

    df = pd.read_parquet(FEATURES_FILE)
    df["tourney_date"] = pd.to_datetime(df["tourney_date"])
    print(f"[backlog] Loaded {len(df):,} matches")

    # Slice to the relevant window
    if days is not None:
        cutoff = pd.Timestamp.today() - pd.Timedelta(days=days)
        df = df[df["tourney_date"] >= cutoff].copy()
        print(f"[backlog] Filtered to last {days} days → {len(df):,} matches")
    else:
        df = df.copy()
        print(f"[backlog] Scoring all {len(df):,} matches (full history)")

    if df.empty:
        print("[backlog] No matches in window — writing empty file")
        pd.DataFrame(columns=KEEP_COLS).to_parquet(OUTPUT_FILE, index=False)
        return

    # Load model
    print(f"[backlog] Loading model from {MODEL_FILE}")
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"Model file not found: {MODEL_FILE}")

    from predict import load_model, MatchPredictor
    model_data = load_model(MODEL_FILE)
    full_features = pd.read_parquet(FEATURES_FILE)
    full_features["tourney_date"] = pd.to_datetime(full_features["tourney_date"])
    predictor = MatchPredictor.from_model_dict(model_data, full_features)

    # Score every match
    probs: list[float] = []
    n = len(df)
    for i, (_, row) in enumerate(df.iterrows()):
        if i % 500 == 0:
            print(f"[backlog]   {i:,}/{n:,}…", end="\r", flush=True)
        try:
            _, prob = predictor.predict(row["winner_name"], row["loser_name"], row["surface"])
            probs.append(float(prob))
        except Exception:
            probs.append(np.nan)

    print(f"\n[backlog] Scored {n:,} matches")
    df["model_prob_w"] = probs

    # Keep only wanted columns (fill missing ones with NaN)
    out_cols = [c for c in KEEP_COLS if c in df.columns]
    out = df[out_cols].sort_values("tourney_date", ascending=False).reset_index(drop=True)

    out.to_parquet(OUTPUT_FILE, index=False)
    print(f"[backlog] Saved → {OUTPUT_FILE}  ({len(out):,} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-compute prediction backlog")
    group  = parser.add_mutually_exclusive_group()
    group.add_argument("--days", type=int, default=90,
                       help="Number of days of history to score (default: 90)")
    group.add_argument("--all",  action="store_true",
                       help="Score the entire match history (slow)")
    args = parser.parse_args()
    main(days=None if args.all else args.days)
