"""
update_tml_data.py
------------------
Keeps the TennisMyLife (TML) dataset current by re-downloading only the
files that change frequently: the current year, current year challengers,
and the live ongoing-tourneys file.

Run this daily (or before each model refresh) to stay up to date.

Initial full download (one-time):
    PowerShell:
        New-Item -ItemType Directory -Force -Path .\tml-data | Out-Null
        Invoke-RestMethod 'https://stats.tennismylife.org/api/data-files' |
          Select-Object -ExpandProperty files |
          ForEach-Object { Invoke-WebRequest -Uri $_.url -OutFile (Join-Path '.\tml-data' $_.name) }

    Or download the ZIP directly:
        https://stats.tennismylife.org/api/download-all

Daily update:
    python update_tml_data.py
    python update_tml_data.py --year 2026          # explicit year
    python update_tml_data.py --full               # re-download ALL files
"""

import argparse
import datetime
import glob
import re
from pathlib import Path

import pandas as pd
import requests

TML_BASE = "https://stats.tennismylife.org/data"
TML_DATA = Path(__file__).parent / "tml-data"
TML_DATA.mkdir(exist_ok=True)

HEADERS = {"User-Agent": "tennis-predictions-updater/1.0"}


def _download(filename: str) -> bool:
    url  = f"{TML_BASE}/{filename}"
    dest = TML_DATA / filename
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        dest.write_bytes(r.content)
        print(f"  OK {filename} ({len(r.content)//1024} KB)")
        return True
    except requests.RequestException as e:
        print(f"  FAIL {filename}: {e}")
        return False


def update(year: int | None = None, full: bool = False) -> None:
    if year is None:
        year = datetime.date.today().year

    if full:
        manifest = requests.get(
            "https://stats.tennismylife.org/api/data-files",
            headers=HEADERS, timeout=15
        ).json()["files"]
        files = [f["name"] for f in manifest]
        print(f"[TML] Full re-download: {len(files)} files")
    else:
        files = [
            f"{year}.csv",
            f"{year}_challenger.csv",
            "ongoing_tourneys.csv",
        ]
        print(f"[TML] Incremental update for {year}")

    ok = sum(_download(f) for f in files)
    print(f"[TML] Done — {ok}/{len(files)} files updated.")


def supplement_from_tennis_data_co_uk(year: int) -> None:
    """
    Download the tennis-data.co.uk XLSX for `year`, convert it to TML schema,
    and append any rows newer than TML's latest date for that year.

    This fills the typical 2–3 week lag in TML source data so that
    features.py (and therefore ELO ratings) stay current.

    Player IDs are resolved by matching the abbreviated TD name (e.g. "Djokovic N.")
    to TML-format IDs via a last-name lookup built from all existing TML CSVs.
    """
    try:
        import importlib
        td = importlib.import_module("ingest_tennis_data_co_uk")
    except ImportError as exc:
        print(f"[supplement] Could not import ingest_tennis_data_co_uk: {exc}")
        return

    tml_path = TML_DATA / f"{year}.csv"
    if not tml_path.exists():
        print(f"[supplement] {tml_path.name} not found — skipping")
        return

    # ── 1. Build last-name → TML player-ID lookup from all TML annual files ───
    # TML IDs are proprietary (e.g. "B0BI"); they are NOT the same as Sackmann
    # numeric IDs.  We derive them from TML's own name columns.
    name_to_id: dict[str, str] = {}
    lastname_to_id: dict[str, str] = {}   # last word of name → id (for TD abbreviated names)
    for f in sorted(glob.glob(str(TML_DATA / "[0-9][0-9][0-9][0-9].csv"))):
        try:
            chunk = pd.read_csv(
                f,
                usecols=["winner_name", "winner_id", "loser_name", "loser_id"],
                low_memory=False,
            )
            for nc, ic in [("winner_name", "winner_id"), ("loser_name", "loser_id")]:
                for name, pid in zip(chunk[nc], chunk[ic]):
                    name = str(name or "").strip()
                    pid  = str(pid  or "").strip()
                    if not name or not pid or pid in ("nan", "None", ""):
                        continue
                    name_lower = name.lower()
                    name_to_id[name_lower] = pid
                    # Index by last token (surname) for abbreviated-name matching
                    last = name_lower.rsplit(" ", 1)[-1]
                    lastname_to_id.setdefault(last, pid)   # first occurrence wins
        except Exception:
            pass
    print(f"[supplement] Name->ID lookup: {len(name_to_id):,} full names, "
          f"{len(lastname_to_id):,} last-name entries")

    def _resolve_id(raw_name: str) -> str:
        """
        Return a TML player ID for a tennis-data.co.uk player name.
        TD uses "Lastname F." format; TML uses "Firstname Lastname".
        """
        name = str(raw_name or "").strip()
        # Direct full-name hit (handles unusual rows that already have full names)
        hit = name_to_id.get(name.lower())
        if hit:
            return hit
        # Extract surname from "Lastname F." → "Lastname"
        m_abbr = re.match(r"^(.+?)\s+[A-Z]\.$", name)
        if m_abbr:
            surname = m_abbr.group(1).lower().strip()
        else:
            # Fallback: last space-separated token
            surname = name.lower().rsplit(" ", 1)[-1]
        # Scan full-name dict for any entry whose last token matches
        for full, pid in name_to_id.items():
            if full.rsplit(" ", 1)[-1] == surname:
                return pid
        # Last resort: direct last-name index
        return lastname_to_id.get(surname, "")

    # ── 2. Load TML max date ───────────────────────────────────────────────────
    tml_df = pd.read_csv(tml_path, low_memory=False)
    tml_df["tourney_date"] = pd.to_numeric(tml_df["tourney_date"], errors="coerce")
    tml_max = int(tml_df["tourney_date"].max())
    print(f"[supplement] TML max date for {year}: {tml_max}")

    # ── 3. Download & convert tennis-data.co.uk ───────────────────────────────
    xlsx_path = td.download_xlsx(year)
    if xlsx_path is None:
        print(f"[supplement] tennis-data.co.uk {year}.xlsx unavailable — skipping")
        return

    td_df = td.convert_xlsx(xlsx_path, year)
    if td_df.empty:
        print("[supplement] Converted 0 rows — skipping")
        return

    td_df["tourney_date"] = pd.to_numeric(td_df["tourney_date"], errors="coerce")
    new_rows = td_df[td_df["tourney_date"] > tml_max].copy()
    if new_rows.empty:
        td_max = int(td_df["tourney_date"].max())
        print(f"[supplement] No new rows (TD max {td_max} ≤ TML max {tml_max})")
        return

    # ── 4. Resolve player IDs ─────────────────────────────────────────────────
    new_rows["winner_id"] = new_rows["winner_name"].apply(_resolve_id)
    new_rows["loser_id"]  = new_rows["loser_name"].apply(_resolve_id)

    resolved = new_rows[
        (new_rows["winner_id"].str.strip() != "") &
        (new_rows["loser_id"].str.strip()  != "")
    ].copy()
    n_dropped = len(new_rows) - len(resolved)
    if n_dropped:
        print(f"[supplement] Could not resolve IDs for {n_dropped} rows (unknown players)")
    if resolved.empty:
        print("[supplement] No rows with resolved IDs — skipping")
        return

    d_min = int(resolved["tourney_date"].min())
    d_max = int(resolved["tourney_date"].max())
    print(f"[supplement] Appending {len(resolved)} rows "
          f"(dates {d_min} – {d_max}) to {tml_path.name}")

    # ── 5. Merge, sort, save ──────────────────────────────────────────────────
    combined = pd.concat([tml_df, resolved], ignore_index=True, sort=False)
    combined["tourney_date"] = pd.to_numeric(combined["tourney_date"], errors="coerce")
    combined = combined.sort_values(
        ["tourney_date", "tourney_id", "match_num"], na_position="last"
    ).reset_index(drop=True)
    combined.to_csv(tml_path, index=False)
    print(f"[supplement] {tml_path.name} saved — {len(combined):,} rows total")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=None,
                        help="Year to refresh (default: current year)")
    parser.add_argument("--full", action="store_true",
                        help="Re-download all 111 files")
    parser.add_argument("--supplement", action="store_true",
                        help="After TML update, fill the TML lag gap with tennis-data.co.uk data")
    args = parser.parse_args()
    update(year=args.year, full=args.full)
    if args.supplement:
        supplement_from_tennis_data_co_uk(year=args.year or datetime.date.today().year)
