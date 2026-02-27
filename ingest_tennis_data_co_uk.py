"""
ingest_tennis_data_co_uk.py
----------------------------
Downloads and converts free Excel match files from tennis-data.co.uk
into the Sackmann CSV schema used by the rest of this project.

tennis-data.co.uk provides free annual Excel files going back to 2000 that
include:
  • All main-tour ATP matches (ATP250 / ATP500 / Masters / Slams)
  • Pre-match betting odds from Bet365, Pinnacle, and market average
  • Set-by-set scores, round, surface, rankings at match time

URL pattern:
  ATP men:  https://www.tennis-data.co.uk/{year}/{year}.xlsx
  ATP WTA:  https://www.tennis-data.co.uk/{year}w/{year}.xlsx   (optional)

If the file already exists locally (as data_files/2025.xlsx) the download
step is skipped and the local file is used directly.

After conversion, player metadata (hand, height, IOC code, player_id) is
backfilled from the local atp_players.csv by fuzzy name matching.

Output:
  data_files/td_atp_{year}.csv       — converted Sackmann-schema CSV
  tennis_atp/atp_matches_{year}.csv  — optional copy into ATP data dir

Usage:
    python ingest_tennis_data_co_uk.py                     # 2025 only
    python ingest_tennis_data_co_uk.py --year 2024
    python ingest_tennis_data_co_uk.py --year-range 2020 2025
    python ingest_tennis_data_co_uk.py --year 2025 --copy-to-atp
"""

from __future__ import annotations

import argparse
import re
import shutil
import ssl
import warnings
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class _LaxSSLAdapter(HTTPAdapter):
    """Allow legacy servers that only speak older TLS/cipher sets."""
    def init_poolmanager(self, *args, **kwargs):
        ctx = ssl.create_default_context()
        ctx.set_ciphers("DEFAULT:@SECLEVEL=1")
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        ctx.options |= ssl.OP_LEGACY_SERVER_CONNECT
        kwargs["ssl_context"] = ctx
        super().init_poolmanager(*args, **kwargs)


def _get_session() -> requests.Session:
    s = requests.Session()
    adapter = _LaxSSLAdapter()
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update(HEADERS)
    return s

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data_files"
ATP_DIR  = BASE_DIR / "tennis_atp"
DATA_DIR.mkdir(exist_ok=True)

TD_ATP_URL = "https://www.tennis-data.co.uk/{year}/{year}.xlsx"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.tennis-data.co.uk/",
}

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

# tennis-data.co.uk "Series" → Sackmann tourney_level code
LEVEL_MAP = {
    "Grand Slam":    "G",
    "Masters Cup":   "F",
    "Masters 1000":  "M",
    "International Gold": "A",  # older naming convention
    "International":      "A",
    "ATP500":        "A",
    "ATP250":        "A",
}

SURFACE_NORM = {
    "Hard":   "Hard",
    "Clay":   "Clay",
    "Grass":  "Grass",
    "Carpet": "Carpet",
}


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _local_xlsx(year: int) -> Path | None:
    """
    Return a local .xlsx path if one already exists for `year`.
    Checks both data_files/<year>.xlsx and data_files/td_atp_<year>.xlsx.
    """
    for name in (f"{year}.xlsx", f"td_atp_{year}.xlsx"):
        p = DATA_DIR / name
        if p.exists():
            return p
    return None


def download_xlsx(year: int) -> Path | None:
    """
    Download the tennis-data.co.uk Excel file for `year`.
    Returns the local path, or None if download fails.
    """
    existing = _local_xlsx(year)
    if existing:
        print(f"  [TD] Using existing file: {existing.name}")
        return existing

    url  = TD_ATP_URL.format(year=year)
    dest = DATA_DIR / f"{year}.xlsx"
    print(f"  [TD] Downloading {url} …", end="", flush=True)
    try:
        sess = _get_session()
        resp = sess.get(url, timeout=30, verify=False)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        print(f" ✓ ({len(resp.content)//1024} KB)")
        return dest
    except requests.RequestException as e:
        print(f" ✗ ({e})")
        print(
            f"\n  [TD] MANUAL DOWNLOAD FALLBACK\n"
            f"       1. Open this URL in your browser: {url}\n"
            f"       2. Save the file as: {dest}\n"
            f"       3. Re-run this script – it will use the local file automatically.\n"
        )
        return None


# ---------------------------------------------------------------------------
# Column mapping: tennis-data.co.uk → Sackmann
# ---------------------------------------------------------------------------

def _level(series_str: str | None) -> str:
    if not series_str:
        return "A"
    s = str(series_str).strip()
    return LEVEL_MAP.get(s, "A")


def _surface(s: str | None) -> str:
    if not s:
        return "Hard"
    return SURFACE_NORM.get(str(s).strip().title(), str(s).strip().title())


def _norm_date(val) -> str:
    """Convert any date-like value to YYYYMMDD string."""
    if pd.isna(val):
        return ""
    if hasattr(val, "strftime"):
        return val.strftime("%Y%m%d")
    s = str(val).strip()
    clean = re.sub(r"[^\d]", "", s)
    return clean[:8].ljust(8, "0") if clean else ""


def _reconstruct_score(row: pd.Series) -> str:
    """
    Reconstruct a score string from W1/L1 … W5/L5 set columns.
    E.g. "6-4 7-5" or "6-3 4-6 7-6(4)"
    """
    sets = []
    for i in range(1, 6):
        wk, lk = f"W{i}", f"L{i}"
        w = row.get(wk)
        l = row.get(lk)
        if pd.isna(w) or pd.isna(l):
            break
        w_int = int(w)
        l_int = int(l)
        # A set score of 7-6 almost certainly had a tiebreak
        if w_int == 7 and l_int == 6:
            sets.append("7-6")
        elif l_int == 7 and w_int == 6:
            sets.append("6-7")
        else:
            sets.append(f"{w_int}-{l_int}")
    return " ".join(sets)


def _tourney_id(tourney_name: str, date_str: str, year: int) -> str:
    """Produce a synthetic Sackmann-style tourney_id."""
    slug = re.sub(r"[^\w]", "", str(tourney_name or "UNKNOWN"))[:10].upper()
    return f"{year}-{slug}"


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def convert_xlsx(xlsx_path: Path, year: int) -> pd.DataFrame:
    """
    Read a tennis-data.co.uk Excel file and return a Sackmann-schema DataFrame.
    Removes retirements/walkovers unless --keep-incomplete is set.
    """
    print(f"  [TD] Reading {xlsx_path.name} …")
    try:
        raw = pd.read_excel(xlsx_path, engine="openpyxl")
    except Exception as e:
        print(f"  [TD] Failed to read Excel file: {e}")
        return pd.DataFrame(columns=SACKMANN_COLS)

    print(f"  [TD] {len(raw)} rows, {len(raw.columns)} columns")

    # Filter to completed matches only (skip Retired, Walkover)
    if "Comment" in raw.columns:
        before = len(raw)
        raw = raw[raw["Comment"].fillna("Completed") == "Completed"]
        print(f"  [TD] After filtering retirements/walkovers: {len(raw)} rows "
              f"(removed {before - len(raw)})")

    rows = []
    for match_num, (_, row) in enumerate(raw.iterrows(), 1):
        date_norm    = _norm_date(row.get("Date"))
        tourney_name = str(row.get("Tournament") or "Unknown")
        surface      = _surface(row.get("Surface"))
        level        = _level(row.get("Series"))
        score        = _reconstruct_score(row)

        rows.append({
            "tourney_id":          _tourney_id(tourney_name, date_norm, year),
            "tourney_name":        tourney_name,
            "surface":             surface,
            "draw_size":           None,
            "tourney_level":       level,
            "tourney_date":        date_norm,
            "match_num":           match_num,
            "winner_id":           "",
            "winner_seed":         "",
            "winner_entry":        "",
            "winner_name":         str(row.get("Winner") or ""),
            "winner_hand":         "",
            "winner_ht":           None,
            "winner_ioc":          "",
            "winner_age":          None,
            "loser_id":            "",
            "loser_seed":          "",
            "loser_entry":         "",
            "loser_name":          str(row.get("Loser") or ""),
            "loser_hand":          "",
            "loser_ht":            None,
            "loser_ioc":           "",
            "loser_age":           None,
            "score":               score,
            "best_of":             row.get("Best of") or 3,
            "round":               str(row.get("Round") or ""),
            "minutes":             None,
            # No serve stats in tennis-data.co.uk files
            "w_ace": None, "w_df": None, "w_svpt": None,
            "w_1stIn": None, "w_1stWon": None, "w_2ndWon": None,
            "w_SvGms": None, "w_bpSaved": None, "w_bpFaced": None,
            "l_ace": None, "l_df": None, "l_svpt": None,
            "l_1stIn": None, "l_1stWon": None, "l_2ndWon": None,
            "l_SvGms": None, "l_bpSaved": None, "l_bpFaced": None,
            "winner_rank":         row.get("WRank"),
            "winner_rank_points":  row.get("WPts"),
            "loser_rank":          row.get("LRank"),
            "loser_rank_points":   row.get("LPts"),
        })

    df = pd.DataFrame(rows, columns=SACKMANN_COLS)

    # Re-number match_num per tournament
    df["match_num"] = df.groupby("tourney_id").cumcount() + 1

    return df


# ---------------------------------------------------------------------------
# Player metadata backfill (same logic as merge_2025_data.py)
# ---------------------------------------------------------------------------

def _norm_name(name: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", "", name.lower())).strip()


def _build_lookup(atp_dir: Path) -> dict:
    p = atp_dir / "atp_players.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p, encoding="utf-8", on_bad_lines="skip", low_memory=False)
    if not {"player_id", "name_first", "name_last"}.issubset(df.columns):
        return {}
    lookup: dict = {}
    for _, row in df.iterrows():
        first = str(row.get("name_first", "") or "").strip().lower()
        last  = str(row.get("name_last",  "") or "").strip().lower()
        meta  = {
            "player_id": row.get("player_id"),
            "hand":      row.get("hand"),
            "ht":        row.get("height"),
            "ioc":       row.get("ioc"),
        }
        lookup[last]                  = meta
        lookup[f"{first} {last}"]     = meta
        lookup[f"{first}{last}"]      = meta
    return lookup


def _expand_abbreviated_name(name: str) -> str:
    """
    tennis-data.co.uk uses "Djokovic N." format.
    Return a normalised lowercase "first last" form for matching.
    """
    name = name.strip()
    # Already "First Last" form
    if not re.search(r"\s[A-Z]\.$", name):
        return _norm_name(name)
    # "Last I." → just "last" for lookup
    parts = name.rsplit(" ", 1)
    return _norm_name(parts[0])


def enrich_df(df: pd.DataFrame, lookup: dict) -> pd.DataFrame:
    """Backfill player metadata on winner/loser columns."""
    for prefix in ("winner", "loser"):
        id_col   = f"{prefix}_id"
        hand_col = f"{prefix}_hand"
        ht_col   = f"{prefix}_ht"
        ioc_col  = f"{prefix}_ioc"
        name_col = f"{prefix}_name"

        for i, row in df.iterrows():
            raw_name = str(row.get(name_col) or "")
            if not raw_name:
                continue
            key  = _expand_abbreviated_name(raw_name)
            meta = lookup.get(key) or {}
            if not meta:
                # try just last name
                key = key.split()[-1] if key else ""
                meta = lookup.get(key) or {}
            if not meta:
                continue

            if not str(row.get(id_col) or "").strip():
                pid = meta.get("player_id")
                if pid is not None:
                    df.at[i, id_col] = pid
            if not str(row.get(hand_col) or "").strip():
                h = meta.get("hand")
                if h:
                    df.at[i, hand_col] = h
            if pd.isna(row.get(ht_col)):
                ht = meta.get("ht")
                if ht:
                    df.at[i, ht_col] = ht
            if not str(row.get(ioc_col) or "").strip():
                ioc = meta.get("ioc")
                if ioc:
                    df.at[i, ioc_col] = ioc
    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def process_year(year: int, copy_to_atp: bool, enrich: bool) -> pd.DataFrame | None:
    print(f"\n[TD] ── Year {year} ──────────────────────────")
    xlsx = download_xlsx(year)
    if xlsx is None:
        print(f"[TD] Skipping {year} – no file available.")
        return None

    df = convert_xlsx(xlsx, year)
    if df.empty:
        print(f"[TD] No usable data for {year}.")
        return None

    if enrich and ATP_DIR.exists():
        print(f"  [TD] Enriching player metadata…")
        lookup = _build_lookup(ATP_DIR)
        if lookup:
            df = enrich_df(df, lookup)

    out_csv = DATA_DIR / f"td_atp_{year}.csv"
    df.to_csv(out_csv, index=False)
    print(f"  [TD] Saved {len(df)} matches → {out_csv.name}")

    if copy_to_atp and ATP_DIR.exists():
        dest = ATP_DIR / f"atp_matches_{year}.csv"
        shutil.copy2(out_csv, dest)
        print(f"  [TD] Copied → tennis_atp/atp_matches_{year}.csv")

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download & convert tennis-data.co.uk Excel files to Sackmann CSV schema"
    )
    parser.add_argument("--year",       type=int, default=None,
                        help="Single year to process (default: 2025)")
    parser.add_argument("--year-range", type=int, nargs=2, metavar=("START", "END"),
                        help="Inclusive year range, e.g. --year-range 2020 2025")
    parser.add_argument("--copy-to-atp", action="store_true",
                        help="Also copy output to tennis_atp/atp_matches_YYYY.csv")
    parser.add_argument("--no-enrich",   action="store_true",
                        help="Skip player metadata backfill")
    args = parser.parse_args()

    if args.year_range:
        years = list(range(args.year_range[0], args.year_range[1] + 1))
    else:
        years = [args.year or 2025]

    all_dfs = []
    for y in years:
        df = process_year(y, copy_to_atp=args.copy_to_atp, enrich=not args.no_enrich)
        if df is not None:
            all_dfs.append(df)

    if all_dfs:
        total = sum(len(d) for d in all_dfs)
        print(f"\n[TD] Done. {len(all_dfs)} year(s) processed, {total} total matches.")


if __name__ == "__main__":
    main()
