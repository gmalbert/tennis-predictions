"""
scripts/export_best_bets.py — Tennis (tennis-predictions)

Primary path: reads data_files/odds_api_today.json (upcoming matches with
market odds written by fetch_odds_api.py), applies the trained ELO-based
model via predict.py to compute win probabilities, and exports picks where
the model shows meaningful edge over the market.

Fallback: if odds_api_today.json is missing or stale, tries the historical
prediction_backlog.parquet for recent completed-match predictions.

Writes data_files/best_bets_today.json in the unified Sports Picks Grid schema.
"""
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

# Ensure repo root is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SPORT = "Tennis"
MODEL_VERSION = "1.0.0"
SEASON = str(date.today().year)
OUT_PATH      = ROOT / "data_files" / "best_bets_today.json"
ODDS_API_PATH = ROOT / "data_files" / "odds_api_today.json"
BACKLOG_PATH  = ROOT / "data_files" / "prediction_backlog.parquet"
CONF_MIN = 0.55  # minimum model confidence to consider
MIN_EDGE = 0.03  # minimum edge over market implied probability

# Surface by sport key (used when predicting)
SURFACE_MAP = {
    "tennis_atp_french_open":      "Clay",
    "tennis_wta_french_open":      "Clay",
    "tennis_atp_roland_garros":    "Clay",
    "tennis_wta_roland_garros":    "Clay",
    "tennis_atp_wimbledon":        "Grass",
    "tennis_wta_wimbledon":        "Grass",
    "tennis_atp_aus_open_singles": "Hard",
    "tennis_wta_aus_open_singles": "Hard",
    "tennis_atp_us_open":          "Hard",
    "tennis_wta_us_open":          "Hard",
    "tennis_atp_monte_carlo_masters": "Clay",
    "tennis_atp_madrid_open":      "Clay",
    "tennis_atp_italian_open":     "Clay",
    "tennis_wta_italian_open":     "Clay",
    "tennis_wta_madrid_open":      "Clay",
    "tennis_atp_indian_wells":     "Hard",
    "tennis_atp_miami_open":       "Hard",
    "tennis_atp_canadian_open":    "Hard",
    "tennis_atp_cincinnati_open":  "Hard",
    "tennis_atp_shanghai_masters": "Hard",
    "tennis_atp_paris_masters":    "Hard (Indoor)",
    "tennis_atp_dubai":            "Hard",
    "tennis_atp_qatar_open":       "Hard",
    "tennis_atp_china_open":       "Hard",
    "tennis_wta_canadian_open":    "Hard",
    "tennis_wta_cincinnati_open":  "Hard",
    "tennis_wta_miami_open":       "Hard",
    "tennis_wta_indian_wells_masters": "Hard",
}



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


def _american_from_decimal(dec: float) -> int | None:
    """Convert decimal odds to American format (rough)."""
    try:
        if dec >= 2.0:
            return round((dec - 1) * 100)
        else:
            return round(-100 / (dec - 1))
    except Exception:
        return None


def _bets_from_odds_api() -> list[dict]:
    """
    Primary path: use odds_api_today.json (upcoming matches with market odds)
    as the fixture list, apply the trained ELO model, and return qualifying bets.
    """
    if not ODDS_API_PATH.exists():
        print("[Tennis] odds_api_today.json not found — skipping primary path")
        return []

    try:
        with open(ODDS_API_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as e:
        print(f"[Tennis] Failed to read odds_api_today.json: {e}")
        return []

    today_str = date.today().strftime("%Y-%m-%d")
    if data.get("date_fetched") != today_str:
        print(f"[Tennis] odds_api_today.json is from {data.get('date_fetched')} (not today) — skipping")
        return []

    matches = data.get("matches", [])
    if not matches:
        print("[Tennis] odds_api_today.json has 0 matches")
        return []

    # Load trained model and features for ELO-based predictions
    try:
        from predict import load_model, MatchPredictor
        import pandas as pd
        features_df = pd.read_parquet(ROOT / "data_files" / "features_2020_present.parquet")
        model_data = load_model(str(ROOT / "data_files" / "tennis_predictor.pkl"))
        predictor = MatchPredictor.from_model_dict(model_data, features_df)
        print(f"[Tennis] Model loaded; applying to {len(matches)} upcoming matches")
    except Exception as e:
        print(f"[Tennis] Failed to load model: {e}")
        return []

    bets: list[dict] = []
    today = date.today()

    for m in matches:
        p1 = m.get("player1", "")
        p2 = m.get("player2", "")
        if not p1 or not p2:
            continue

        sport_key = m.get("sport_key", "")
        surface = SURFACE_MAP.get(sport_key, "Hard")

        # Apply model
        try:
            _, p1_model_prob = predictor.predict_by_name(p1, p2, surface)
        except Exception:
            # Name lookup fails for players not in training data — skip
            continue

        p2_model_prob = 1.0 - p1_model_prob

        # Market devigged probabilities from the odds file
        p1_mkt = m.get("p1_prob", 0.5)
        p2_mkt = m.get("p2_prob", 0.5)

        # Pick the player with the larger model edge
        edge1 = p1_model_prob - p1_mkt
        edge2 = p2_model_prob - p2_mkt

        if edge1 >= edge2:
            pick, conf, edge, mkt_prob, odds_decimal = p1, p1_model_prob, edge1, p1_mkt, m.get("player1_odds")
        else:
            pick, conf, edge, mkt_prob, odds_decimal = p2, p2_model_prob, edge2, p2_mkt, m.get("player2_odds")

        if conf < CONF_MIN or edge < MIN_EDGE:
            continue

        # Parse game date from commence_time (ISO 8601)
        ct = m.get("commence_time", "")
        try:
            game_date = datetime.fromisoformat(ct.replace("Z", "+00:00")).date()
        except Exception:
            game_date = today

        odds_american = _american_from_decimal(odds_decimal) if odds_decimal else None

        bets.append({
            "game_date":  str(game_date),
            "game_time":  ct,
            "game":       f"{p1} vs {p2}",
            "home_team":  p1,
            "away_team":  p2,
            "bet_type":   "Match Winner",
            "pick":       pick,
            "confidence": round(conf, 4),
            "edge":       round(edge, 4),
            "tier":       _tier(conf, edge),
            "odds":       odds_american,
            "line":       None,
            "notes":      f"Surface: {surface} | Mkt: {mkt_prob:.1%}",
        })

    return bets


def main() -> None:
    today = date.today()

    # ── Primary: odds_api_today.json (upcoming matches) ──────────────────────
    bets = _bets_from_odds_api()
    if bets:
        _write(bets)
        return

    # ── Fallback: prediction backlog for very recent matches (last 2 days) ───
    # This covers the case where odds_api_today.json is stale but the workflow
    # has just run and completed recent matches might still be relevant.
    if BACKLOG_PATH.exists():
        try:
            import pandas as pd
            df = pd.read_parquet(BACKLOG_PATH)
            if not df.empty and "tourney_date" in df.columns:
                df["tourney_date"] = pd.to_datetime(df["tourney_date"], errors="coerce").dt.date
                from datetime import timedelta
                df = df[df["tourney_date"] >= today - timedelta(days=1)]
                if not df.empty:
                    recent: list[dict] = []
                    for _, row in df.iterrows():
                        conf = _safe_float(row.get("model_prob_w"))
                        mkt  = _safe_float(row.get("mkt_prob_w"))
                        if conf is None:
                            continue
                        edge = (conf - mkt) if mkt is not None else (conf - 0.5)
                        if conf >= CONF_MIN and edge >= MIN_EDGE:
                            winner = str(row.get("winner_name", ""))
                            loser  = str(row.get("loser_name", ""))
                            recent.append({
                                "game_date":  str(row.get("tourney_date", today)),
                                "game_time":  None,
                                "game":       f"{winner} vs {loser}",
                                "home_team":  winner,
                                "away_team":  loser,
                                "bet_type":   "Match Winner",
                                "pick":       winner,
                                "confidence": round(conf, 4),
                                "edge":       round(edge, 4),
                                "tier":       _tier(conf, edge),
                                "odds":       None,
                                "line":       None,
                                "notes":      f"Surface: {row.get('surface','')}",
                            })
                    if recent:
                        _write(recent, "Fallback: recent completed matches")
                        return
        except Exception as e:
            print(f"[Tennis] Backlog fallback failed: {e}")

    _write([], f"No qualifying Tennis picks for {today}")


if __name__ == "__main__":
    main()
