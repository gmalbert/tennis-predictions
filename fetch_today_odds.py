"""
fetch_today_odds.py
-------------------
Nightly snapshot of Flashscore pre-match odds for all upcoming ATP/WTA singles.

Appends one row per (event_id × bookmaker) to
    data_files/flashscore_odds_history.csv

Rows are keyed on (event_id, bookmaker_id, date_fetched) so re-running on
the same day is safe — duplicates are silently dropped.

Usage:
    python fetch_today_odds.py              # fetch and append
    python fetch_today_odds.py --dry-run    # print rows, do not write
    python fetch_today_odds.py --limit 30   # cap number of matches queried
"""

import argparse
import os
from datetime import datetime, timezone

import pandas as pd

DATA_DIR    = "data_files"
HISTORY_CSV = os.path.join(DATA_DIR, "flashscore_odds_history.csv")

# Columns written to the CSV (in order)
COLS = [
    "date_fetched",       # YYYY-MM-DD UTC — partition key for deduplication
    "fetched_at",         # full ISO-8601 UTC timestamp
    "event_id",           # Flashscore match AA field
    "ts_start",           # Unix UTC start time of the match
    "tourney",            # full tournament header (includes circuit + surface)
    "surface",            # clay / hard / grass
    "p1",                 # home player name
    "p2",                 # away player name
    "wu",                 # p1 Flashscore URL slug
    "wv",                 # p2 Flashscore URL slug
    "bookmaker_id",
    "bookmaker_name",
    "p1_odds",            # current decimal odds
    "p2_odds",
    "p1_opening",         # opening decimal odds
    "p2_opening",
]


def fetch_and_append(dry_run: bool = False, limit: int | None = None) -> int:
    """
    Fetch today's Flashscore odds and append new rows to HISTORY_CSV.

    Returns the number of new rows written (0 on dry-run).
    """
    from flashscore_odds import fetch_flashscore_odds

    now_utc     = datetime.now(timezone.utc)
    date_str    = now_utc.strftime("%Y-%m-%d")
    fetched_at  = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

    print(f"Fetching Flashscore odds at {fetched_at} ...")
    rows = fetch_flashscore_odds(
        request_delay_s=0.3,
        max_matches=limit,
    )
    print(f"  Received {len(rows)} rows ({len({r['event_id'] for r in rows})} matches)")

    if not rows:
        print("  Nothing to write.")
        return 0

    new_df = pd.DataFrame(rows)
    new_df["date_fetched"] = date_str
    new_df["fetched_at"]   = fetched_at
    new_df = new_df[COLS]

    if dry_run:
        print(new_df.to_string(index=False))
        return 0

    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(HISTORY_CSV):
        existing = pd.read_csv(HISTORY_CSV, dtype={"event_id": str, "bookmaker_id": str})

        # Deduplicate: skip rows already stored for this date's event × bookmaker
        dedup_keys = {"event_id", "bookmaker_id", "date_fetched"}
        existing_keys = set(
            zip(existing["event_id"].astype(str),
                existing["bookmaker_id"].astype(str),
                existing["date_fetched"].astype(str))
        )
        new_df = new_df[
            ~new_df.apply(
                lambda r: (str(r["event_id"]), str(r["bookmaker_id"]), r["date_fetched"])
                          in existing_keys,
                axis=1,
            )
        ]

        if new_df.empty:
            print("  All rows already present in history — nothing to append.")
            return 0

        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined.to_csv(HISTORY_CSV, index=False)
    print(f"  Appended {len(new_df)} rows -> {HISTORY_CSV}  "
          f"(total: {len(combined):,})")
    return len(new_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch today's Flashscore odds and store historically.")
    parser.add_argument("--dry-run", action="store_true", help="Print rows without writing")
    parser.add_argument("--limit",   type=int, default=None, help="Max matches to query")
    args = parser.parse_args()
    fetch_and_append(dry_run=args.dry_run, limit=args.limit)
