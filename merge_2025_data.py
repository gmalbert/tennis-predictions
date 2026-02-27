"""
merge_2025_data.py
------------------
Merge all scraped 2025 match CSV files (from scraper_atp.py,
scraper_tennis_abstract.py, scraper_flashscore.py, scraper_itf.py)
into a single, deduplicated file that follows the Sackmann schema.

Where possible, player metadata (ID, hand, height, IOC code) is
backfilled from the local atp_players.csv so the merged file works
seamlessly with the existing ELO / feature-engineering pipeline.

Output:
  data_files/atp_matches_2025_merged.csv     — all sources combined
  tennis_atp/atp_matches_2025.csv            — copy into the ATP data
                                               directory (requires write
                                               access to tennis_atp/)

Usage:
    python merge_2025_data.py
    python merge_2025_data.py --year 2024          # any year
    python merge_2025_data.py --no-enrich          # skip player backfill
    python merge_2025_data.py --sources atp,fs     # only specific sources
    python merge_2025_data.py --sources all        # default
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data_files"
ATP_DIR    = BASE_DIR / "tennis_atp"

SACKMANN_COLS = [
    "tourney_id", "tourney_name", "surface", "draw_size", "tourney_level",
    "tourney_date", "match_num",
    "winner_id", "winner_seed", "winner_entry",
    "winner_name", "winner_hand", "winner_ht", "winner_ioc", "winner_age",
    "loser_id",  "loser_seed",  "loser_entry",
    "loser_name",  "loser_hand",  "loser_ht",  "loser_ioc",  "loser_age",
    "score", "best_of", "round", "minutes",
    "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon",
    "w_SvGms", "w_bpSaved", "w_bpFaced",
    "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon",
    "l_SvGms", "l_bpSaved", "l_bpFaced",
    "winner_rank", "winner_rank_points",
    "loser_rank",  "loser_rank_points",
]

# Which scraped CSV patterns map to which source tag
SOURCE_FILES = {
    "atp": "atp_results_{year}.csv",
    "ta":  "ta_results_{year}.csv",
    "fs":  "flashscore_results_{year}.csv",
    "itf": "itf_results_{year}_M.csv",  # men's; extend to W/B/G as needed
}


# ---------------------------------------------------------------------------
# Player metadata lookup
# ---------------------------------------------------------------------------

def build_player_lookup(atp_dir: Path) -> dict[str, dict]:
    """
    Build a dict keyed by normalised last-name (lower-case, no spaces) to
    player metadata rows from atp_players.csv.

    Also builds secondary keys: "firstname_lastname" for richer matching.
    """
    p = atp_dir / "atp_players.csv"
    if not p.exists():
        print("  [merge] atp_players.csv not found – skipping player enrichment.")
        return {}

    df = pd.read_csv(p, encoding="utf-8", on_bad_lines="skip")
    required = {"player_id", "name_first", "name_last"}
    if not required.issubset(df.columns):
        return {}

    lookup: dict[str, dict] = {}
    for _, row in df.iterrows():
        first = str(row.get("name_first", "") or "").strip().lower()
        last  = str(row.get("name_last",  "") or "").strip().lower()
        meta  = {
            "player_id": row.get("player_id"),
            "hand":      row.get("hand"),
            "ht":        row.get("height"),
            "ioc":       row.get("ioc"),
        }
        # Primary key: last name only (handles "Djokovic" form)
        lookup.setdefault(last, meta)
        # Secondary key: "first last" normalised
        if first:
            lookup[f"{first} {last}"] = meta
            lookup[f"{first}{last}"]  = meta
    return lookup


def _norm_name(name: str) -> str:
    """Lower-case, strip punctuation, collapse whitespace."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", "", name.lower())).strip()


def enrich_player(
    df: pd.DataFrame, lookup: dict, winner: bool
) -> pd.DataFrame:
    """
    For each row, try to back-fill id / hand / ht / ioc from `lookup`
    using the player name.  Only fills blank / NaN cells.
    """
    prefix = "winner" if winner else "loser"
    name_col = f"{prefix}_name"
    id_col   = f"{prefix}_id"
    hand_col = f"{prefix}_hand"
    ht_col   = f"{prefix}_ht"
    ioc_col  = f"{prefix}_ioc"

    for i, row in df.iterrows():
        raw_name = str(row.get(name_col, "") or "")
        if not raw_name:
            continue

        norm = _norm_name(raw_name)
        # Try full normalised name, then last name only
        last = norm.split()[-1] if norm else ""
        meta = lookup.get(norm) or lookup.get(last) or {}
        if not meta:
            continue

        if not str(row.get(id_col, "") or "").strip():
            pid = meta.get("player_id")
            if pid is not None:
                df.at[i, id_col] = pid

        if not str(row.get(hand_col, "") or "").strip():
            h = meta.get("hand")
            if h:
                df.at[i, hand_col] = h

        if pd.isna(row.get(ht_col)):
            ht = meta.get("ht")
            if ht:
                df.at[i, ht_col] = ht

        if not str(row.get(ioc_col, "") or "").strip():
            ioc = meta.get("ioc")
            if ioc:
                df.at[i, ioc_col] = ioc

    return df


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _dup_key(row: pd.Series) -> str:
    """
    A match is uniquely identified by the tournament date, sorted player
    names, and round.  We sort so A-vs-B == B-vs-A.
    """
    p1 = _norm_name(str(row.get("winner_name", "") or ""))
    p2 = _norm_name(str(row.get("loser_name",  "") or ""))
    ps = f"{min(p1, p2)}|{max(p1, p2)}"
    return f"{row.get('tourney_date', '')}|{ps}|{row.get('round', '')}"


# ---------------------------------------------------------------------------
# Source quality ordering (higher = preferred when deduplicating)
# ---------------------------------------------------------------------------
SOURCE_PRIORITY = {
    "atp": 4,    # most authoritative: official ATP data
    "fs":  3,    # comprehensive but fewer metadata fields
    "ta":  2,    # good per-player coverage
    "itf": 1,    # ITF-level events only
}


# ---------------------------------------------------------------------------
# Main merge logic
# ---------------------------------------------------------------------------

def load_source(year: int, source: str) -> pd.DataFrame:
    pattern = SOURCE_FILES.get(source, "")
    if not pattern:
        return pd.DataFrame()
    path = DATA_DIR / pattern.format(year=year)
    if not path.exists():
        print(f"  [merge] {path.name} not found – skipping.")
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, low_memory=False, on_bad_lines="skip")
        for col in SACKMANN_COLS:
            if col not in df.columns:
                df[col] = None
        df = df[SACKMANN_COLS].copy()
        df["_source"] = source
        df["_priority"] = SOURCE_PRIORITY.get(source, 0)
        print(f"  [merge] Loaded {len(df):>5d} rows from {path.name}")
        return df
    except Exception as e:
        print(f"  [merge] Failed to read {path.name}: {e}")
        return pd.DataFrame()


def merge(
    year: int,
    sources: list[str],
    enrich: bool,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for src in sources:
        df = load_source(year, src)
        if not df.empty:
            frames.append(df)

    if not frames:
        print("[merge] No source data found.")
        return pd.DataFrame(columns=SACKMANN_COLS)

    combined = pd.concat(frames, ignore_index=True)
    print(f"  [merge] Combined: {len(combined)} rows before dedup")

    # Remove rows with no player names
    combined = combined[
        combined["winner_name"].fillna("").str.strip().ne("") &
        combined["loser_name"].fillna("").str.strip().ne("")
    ]

    # Remove walkovers / retirements
    combined = combined[
        ~combined["score"].fillna("").str.upper().str.contains("W/O|RET|DEF|WALKOVER", regex=True, na=False)
    ]

    # Build dedup key
    combined["_dup_key"] = combined.apply(_dup_key, axis=1)

    # For duplicates, keep the row from the highest-priority source
    combined = (
        combined
        .sort_values("_priority", ascending=False)
        .drop_duplicates(subset="_dup_key", keep="first")
        .sort_values(["tourney_date", "tourney_id", "match_num"], na_position="last")
        .reset_index(drop=True)
    )
    combined.drop(columns=["_dup_key", "_source", "_priority"], inplace=True, errors="ignore")

    print(f"  [merge] After dedup:  {len(combined)} rows")

    # Enrich player metadata from atp_players.csv
    if enrich and ATP_DIR.exists():
        print("  [merge] Enriching player metadata…")
        lookup = build_player_lookup(ATP_DIR)
        if lookup:
            combined = enrich_player(combined, lookup, winner=True)
            combined = enrich_player(combined, lookup, winner=False)
            print(f"  [merge] Player lookup has {len(lookup)} entries")

    # Re-number match_num per tournament
    combined["match_num"] = combined.groupby("tourney_id").cumcount() + 1

    return combined[SACKMANN_COLS]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Merge scraped tennis data into Sackmann schema")
    parser.add_argument("--year",    type=int, default=2025)
    parser.add_argument("--sources", type=str, default="all",
                        help="Comma-separated sources: atp,ta,fs,itf  or  'all'")
    parser.add_argument("--no-enrich", action="store_true",
                        help="Skip backfilling player metadata from atp_players.csv")
    parser.add_argument("--no-copy-to-atp", action="store_true",
                        help="Do not copy output to tennis_atp/ directory")
    args = parser.parse_args()

    if args.sources.lower() == "all":
        sources = list(SOURCE_FILES.keys())
    else:
        sources = [s.strip().lower() for s in args.sources.split(",")]

    print(f"[merge] Year={args.year}  Sources={sources}")

    df = merge(year=args.year, sources=sources, enrich=not args.no_enrich)

    if df.empty:
        print("[merge] Nothing to save.")
        return

    # Primary output
    merged_path = DATA_DIR / f"atp_matches_{args.year}_merged.csv"
    df.to_csv(merged_path, index=False)
    print(f"\n[merge] Saved {len(df)} matches → {merged_path}")

    # Also place in tennis_atp/ so existing pipeline picks it up automatically
    if not args.no_copy_to_atp and ATP_DIR.exists():
        atp_path = ATP_DIR / f"atp_matches_{args.year}.csv"
        if atp_path.exists():
            print(f"[merge] {atp_path.name} already exists in tennis_atp/.")
            ans = input("  Overwrite? [y/N] ").strip().lower()
            if ans != "y":
                print("  Skipped.")
                return
        shutil.copy2(merged_path, atp_path)
        print(f"[merge] Copied → {atp_path}")

    # Quick stats printout
    print(f"\n--- Summary ---")
    print(f"Total matches : {len(df)}")
    if "tourney_level" in df.columns:
        print(df["tourney_level"].value_counts().to_string())
    if "surface" in df.columns:
        print(df["surface"].value_counts().to_string())
    # Coverage check: how many rows have serve stats vs just scores
    stat_cols = ["w_ace", "w_df", "w_svpt"]
    stat_coverage = df[stat_cols].notna().all(axis=1).sum()
    print(f"Rows with full serve stats: {stat_coverage}/{len(df)} "
          f"({100*stat_coverage/max(len(df),1):.0f}%)")


if __name__ == "__main__":
    main()
