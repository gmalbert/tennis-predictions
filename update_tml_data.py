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
from pathlib import Path

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
        print(f"  ✓ {filename} ({len(r.content)//1024} KB)")
        return True
    except requests.RequestException as e:
        print(f"  ✗ {filename}: {e}")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=None,
                        help="Year to refresh (default: current year)")
    parser.add_argument("--full", action="store_true",
                        help="Re-download all 111 files")
    args = parser.parse_args()
    update(year=args.year, full=args.full)
