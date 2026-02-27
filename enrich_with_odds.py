"""
enrich_with_odds.py
--------------------
Joins Bet365 / Pinnacle / Max / Avg / Betfair Exchange odds from
tennis-data.co.uk Excel files onto the TennisMyLife CSVs, which have the
full Sackmann-compatible schema (serve stats, player metadata, etc.).

Name format mismatch:
  TennisMyLife : "Felix Auger-Aliassime"   (First Last)
  tennis-data  : "Auger-Aliassime F."      (Last F.)

Join key used:
  last_word(name).lower() + "_" + first_char(name).lower()
  e.g. "auger-aliassime_f"

  Combined key: winner_key + "|" + loser_key + "|" + YYYYMM
  Month is included so players who meet twice in one season
  (different tournaments) are not cross-contaminated.

Output:
  tml-data/{year}_with_odds.csv   — TML rows + 10 odds columns appended

Usage:
    python enrich_with_odds.py                   # 2025 only
    python enrich_with_odds.py --year 2024
    python enrich_with_odds.py --year-range 2020 2025

Requirements:
    data_files/{year}.xlsx must exist (tennis-data.co.uk raw file).
    tml-data/{year}.csv must exist (downloaded via update_tml_data.py).
    openpyxl installed (pip install openpyxl).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

BASE_DIR  = Path(__file__).parent
DATA_DIR  = BASE_DIR / "data_files"
TML_DIR   = BASE_DIR / "tml-data"

ODDS_COLS = ["B365W", "B365L", "PSW", "PSL", "MaxW", "MaxL", "AvgW", "AvgL", "BFEW", "BFEL"]


# ---------------------------------------------------------------------------
# Name normalisation
# ---------------------------------------------------------------------------

def _letters(s: str) -> str:
    """Lowercase letters only — strip spaces, hyphens, apostrophes, dots."""
    return re.sub(r"[^a-z]", "", s.lower())


def _name_keys(name: str) -> list[str]:
    """
    Return 1-2 candidate match keys for a player name.

    tennis-data.co.uk uses "Surname F." (multi-word surname possible).
    TennisMyLife / Sackmann use "First [Middle] Last".

    Two keys are generated for full names: last-word-only and all-after-first,
    so "Tomas Martin Etcheverry" → ["etcheverry_t", "martinetcheverry_t"] and
    "Carreno Busta P." → ["carrenobusta_p"].  Special chars stripped so
    O'Connell == oconnell, Auger-Aliassime == augeraliassime.
    """
    name = name.strip()
    if not name:
        return []

    # td format: ends with " X." or " Xx." or " Xxx."
    td_match = re.match(r"^(.+?)\s+([A-Z][a-z]{0,2})\.$", name)
    if td_match:
        surname = _letters(td_match.group(1))   # full surname, letters only
        initial = td_match.group(2)[0].lower()  # first char of abbreviation
        return [f"{surname}_{initial}"]

    parts = name.split()
    if len(parts) < 2:
        return [_letters(name)]

    initial   = parts[0][0].lower()
    last_word = _letters(parts[-1])
    all_after = _letters(" ".join(parts[1:]))

    keys = [f"{last_word}_{initial}"]
    if all_after != last_word:
        keys.append(f"{all_after}_{initial}")
    return keys


def _pair_keys(winner_name: str, loser_name: str) -> list[str]:
    """All (winner_key|loser_key) combinations — usually 1, up to 4."""
    return [f"{w}|{l}" for w in _name_keys(winner_name) for l in _name_keys(loser_name)]


def _to_timestamp(date_val) -> pd.Timestamp | None:
    if pd.isna(date_val):
        return None
    if isinstance(date_val, pd.Timestamp):
        return date_val
    try:
        s = re.sub(r"[^\d]", "", str(date_val))
        return pd.Timestamp(s[:4] + "-" + s[4:6] + "-" + s[6:8])
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Core enrichment
# ---------------------------------------------------------------------------

def enrich_year(year: int) -> pd.DataFrame | None:
    tml_path  = TML_DIR  / f"{year}.csv"
    xlsx_path = DATA_DIR / f"{year}.xlsx"

    if not tml_path.exists():
        print(f"  [odds] Missing TML file: {tml_path.name}  →  run update_tml_data.py first")
        return None
    if not xlsx_path.exists():
        print(
            f"  [odds] Missing odds file: {xlsx_path.name}\n"
            f"         Download manually from: https://www.tennis-data.co.uk/{year}/{year}.xlsx\n"
            f"         and save to: {xlsx_path}"
        )
        return None

    print(f"  [odds] Loading TML {year}.csv …")
    tml = pd.read_csv(tml_path, low_memory=False)

    print(f"  [odds] Loading {year}.xlsx …")
    raw = pd.read_excel(xlsx_path, engine="openpyxl")

    # Keep only completed matches in raw (remove retired/WO)
    if "Comment" in raw.columns:
        raw = raw[raw["Comment"].fillna("Completed") == "Completed"].copy()

    # Also load adjacent-year xlsx to handle tournaments spanning year boundaries
    # (e.g. Brisbane International starts Dec 29 — td dates appear in prev year)
    adj_raws = []
    for adj_year in (year - 1, year + 1):
        adj_path = DATA_DIR / f"{adj_year}.xlsx"
        if adj_path.exists():
            try:
                adj = pd.read_excel(adj_path, engine="openpyxl")
                if "Comment" in adj.columns:
                    adj = adj[adj["Comment"].fillna("Completed") == "Completed"]
                adj_raws.append(adj)
                print(f"  [odds] Also loading {adj_year}.xlsx for year-boundary coverage")
            except Exception:
                pass
    if adj_raws:
        raw = pd.concat([raw] + adj_raws, ignore_index=True)

    # Build td lookup: pair_key → list of {date, odds} candidates
    print(f"  [odds] Building odds lookup ({len(raw)} td rows) …")
    odds_lookup: dict[str, list] = {}
    for _, row in raw.iterrows():
        for key in _pair_keys(
            str(row.get("Winner") or ""),
            str(row.get("Loser")  or "")
        ):
            odds_lookup.setdefault(key, []).append({
                "date": row.get("Date"),
                **{c: row.get(c) for c in ODDS_COLS}
            })

    print(f"  [odds] Unique pair-keys in odds lookup: {len(odds_lookup)}")

    # Build TML pair keys + parse tournament-week date for proximity matching
    tml["_pkeys"] = tml.apply(
        lambda r: _pair_keys(
            str(r.get("winner_name") or ""),
            str(r.get("loser_name")  or "")
        ),
        axis=1
    )
    tml["_td"] = tml["tourney_date"].apply(_to_timestamp)

    def _pick_odds(row):
        # Try all candidate pair-key combinations
        candidates = []
        for pk in row["_pkeys"]:
            candidates.extend(odds_lookup.get(pk, []))
        if not candidates:
            return pd.Series({c: None for c in ODDS_COLS})
        if len(candidates) == 1:
            return pd.Series({c: candidates[0][c] for c in ODDS_COLS})
        tml_dt = row["_td"]

        def _dist(c):
            if tml_dt is None:
                return float("inf")
            try:
                d    = pd.Timestamp(c["date"])
                diff = abs((d - tml_dt).days)
                return diff if diff <= 21 else float("inf")
            except Exception:
                return float("inf")

        best_score = min((_dist(c) for c in candidates), default=float("inf"))
        if best_score == float("inf"):
            return pd.Series({c: None for c in ODDS_COLS})
        best = min(candidates, key=_dist)
        return pd.Series({c: best[c] for c in ODDS_COLS})

    odds_df = tml.apply(_pick_odds, axis=1)
    for col in ODDS_COLS:
        tml[col] = odds_df[col]

    matched   = tml[ODDS_COLS[0]].notna().sum()
    total     = len(tml)
    match_pct = matched / total * 100 if total else 0
    print(f"  [odds] Matched {matched}/{total} rows ({match_pct:.1f}%)")

    # Spot-check failures for diagnostics
    unmatched_rows = tml[tml[ODDS_COLS[0]].isna()]
    if not unmatched_rows.empty:
        samples = (unmatched_rows["winner_name"] + " vs " + unmatched_rows["loser_name"]).head(5).tolist()
        print(f"  [odds] Sample unmatched: {samples}")

    tml.drop(columns=["_pkeys", "_td"], inplace=True)

    out = TML_DIR / f"{year}_with_odds.csv"
    tml.to_csv(out, index=False)
    print(f"  [odds] Saved → {out.name}")
    return tml


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enrich TennisMyLife CSVs with tennis-data.co.uk betting odds"
    )
    parser.add_argument("--year",       type=int, default=None)
    parser.add_argument("--year-range", type=int, nargs=2, metavar=("START", "END"))
    args = parser.parse_args()

    if args.year_range:
        years = list(range(args.year_range[0], args.year_range[1] + 1))
    else:
        years = [args.year or 2025]

    for y in years:
        print(f"\n[odds] ── {y} ──────────────────────────")
        enrich_year(y)

    print("\n[odds] Done.")


if __name__ == "__main__":
    main()
