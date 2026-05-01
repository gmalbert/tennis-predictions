"""
scripts/export_best_bets.py — Tennis (tennis-predictions)
Reads data_files/prediction_backlog.parquet and writes
data_files/best_bets_today.json in the unified Sports Picks Grid schema.
"""
import json
from datetime import date, datetime, timezone
from pathlib import Path

SPORT = "Tennis"
MODEL_VERSION = "1.0.0"
SEASON = str(date.today().year)
OUT_PATH = Path("data_files/best_bets_today.json")
BACKLOG_PATH = Path("data_files/prediction_backlog.parquet")
ODDS_PATH    = Path("data_files/flashscore_odds_history.csv")
CONF_MIN = 0.65  # Only HIGH and MEDIUM confidence picks


def _write(bets: list, notes: str = "") -> None:
    payload: dict = {
        "meta": {
            "sport": SPORT,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model_version": MODEL_VERSION,
            "season": SEASON,
        },
        "bets": bets,
    }
    if notes:
        payload["meta"]["notes"] = notes
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"[{SPORT}] Wrote {len(bets)} bets -> {OUT_PATH}")


def _tier(confidence: float, edge: float) -> str:
    if confidence >= 0.75 and edge >= 0.06:
        return "Elite"
    elif confidence >= 0.70 and edge >= 0.03:
        return "Strong"
    elif confidence >= CONF_MIN:
        return "Good"
    return "Standard"


def _safe_float(val) -> float | None:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def main() -> None:
    today = date.today()

    if not BACKLOG_PATH.exists():
        _write([], "prediction_backlog.parquet not found — run compute_backlog.py first")
        return

    try:
        import pandas as pd
        df = pd.read_parquet(BACKLOG_PATH)
    except Exception as e:
        _write([], f"Failed to read backlog: {e}")
        return

    if df.empty:
        _write([], "Prediction backlog is empty")
        return

    # Date filter
    date_col = next((c for c in ["match_date", "date", "tourney_date"] if c in df.columns), None)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.date
        df = df[df[date_col] == today]

    if df.empty:
        _write([], f"No Tennis predictions for {today}")
        return

    # Load odds if available
    odds_lookup: dict = {}
    if ODDS_PATH.exists():
        try:
            odds_df = pd.read_csv(ODDS_PATH)
            for _, orow in odds_df.iterrows():
                key = (str(orow.get("player1", "")), str(orow.get("player2", "")),
                       str(orow.get("match_date", orow.get("date", ""))))
                odds_lookup[key] = orow
        except Exception:
            pass

    bets = []
    for _, row in df.iterrows():
        p1 = str(row.get("player1_name", row.get("player1", row.get("winner_name", ""))))
        p2 = str(row.get("player2_name", row.get("player2", row.get("loser_name", ""))))
        p1_win_prob = _safe_float(row.get("p1_win_prob", row.get("win_probability")))
        if p1_win_prob is None:
            continue

        # Determine winner pick
        if p1_win_prob >= 0.5:
            pick, conf = p1, p1_win_prob
        else:
            pick, conf = p2, 1.0 - p1_win_prob

        if conf < CONF_MIN:
            continue

        # Edge from odds
        game_date_str = str(today)
        odds_row = odds_lookup.get((p1, p2, game_date_str))
        if odds_row is not None:
            implied = _safe_float(odds_row.get("implied_prob"))
            edge = (conf - implied) if implied else 0.0
        else:
            edge = conf - 0.50  # Crude fallback

        if edge < 0.01:
            continue

        surface = str(row.get("surface", ""))
        notes = f"Surface: {surface}" if surface and surface != "nan" else None

        bet: dict = {
            "game_date": game_date_str,
            "game_time": str(row.get("match_time", "")) or None,
            "game": f"{p1} vs {p2}",
            "home_team": p1,
            "away_team": p2,
            "bet_type": "Match Winner",
            "pick": pick,
            "confidence": round(conf, 4),
            "edge": round(edge, 4),
            "tier": _tier(conf, edge),
            "odds": None,
            "line": None,
            "notes": notes,
        }
        bets.append(bet)

    _write(bets, "" if bets else f"No qualifying Tennis picks for {today}")


if __name__ == "__main__":
    main()
