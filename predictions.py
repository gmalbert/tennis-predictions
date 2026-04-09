import json
import os
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from footer import add_betting_oracle_footer
import textwrap

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Tennis Predictions",
    page_icon="🎾",
    layout="wide",
)

DATA_DIR      = "data_files"
FEATURES_FILE = os.path.join(DATA_DIR, "features_2020_present.parquet")
MODEL_FILE    = os.path.join(DATA_DIR, "tennis_predictor.pkl")

# ── Logo ──────────────────────────────────────────────────────────────────────
logo = os.path.join(DATA_DIR, "logo.png")
if os.path.exists(logo):
    st.image(logo, width=180)

st.title("Tennis Predictions")
st.caption(f"Data current to: {date.today().isoformat()}")

def get_dataframe_height(df, row_height=35, header_height=38, padding=2, max_height=600):
    """
    Calculate the optimal height for a Streamlit dataframe based on number of rows.
    
    Args:
        df (pd.DataFrame): The dataframe to display
        row_height (int): Height per row in pixels. Default: 35
        header_height (int): Height of header row in pixels. Default: 38
        padding (int): Extra padding in pixels. Default: 2
        max_height (int): Maximum height cap in pixels. Default: 600 (None for no limit)
    
    Returns:
        int: Calculated height in pixels
    
    Example:
        height = get_dataframe_height(my_df)
        st.dataframe(my_df, height=height)
    """
    num_rows = len(df)
    calculated_height = (num_rows * row_height) + header_height + padding
    
    if max_height is not None:
        return min(calculated_height, max_height)
    return calculated_height

# ── Load feature matrix ───────────────────────────────────────────────────────
# COPILOT: when displaying dataframes use width='stretch' instead of
# use_container_width=True; Streamlit deprecated the latter and we want
# consistent wide tables.  Always search for "use_container_width" when
# editing.
@st.cache_data(ttl=3600)
def load_features() -> pd.DataFrame:
    if not os.path.exists(FEATURES_FILE):
        return pd.DataFrame()
    df = pd.read_parquet(FEATURES_FILE)
    df["tourney_date"] = pd.to_datetime(df["tourney_date"])
    return df

# ── Load trained prediction model ─────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_model_data() -> dict | None:
    """Return the raw pkl dict (metrics, charts data, metadata). None if not trained."""
    if not os.path.exists(MODEL_FILE):
        return None
    try:
        from predict import load_model
        return load_model(MODEL_FILE)
    except Exception:
        return None


@st.cache_resource          # model object is not JSON-serialisable, use cache_resource
def load_predictor():
    """Load the trained MatchPredictor; returns None if model not yet trained."""
    if not os.path.exists(MODEL_FILE):
        return None
    try:
        from predict import load_model, MatchPredictor
        data = load_model(MODEL_FILE)
        df   = load_features()
        return MatchPredictor.from_model_dict(data, df)
    except Exception as e:
        return None

# ── Load today's matches (Matchstat primary, Bzzoiro supplements) ─────────────
def _norm(name: str) -> str:
    """Normalise a player name for cross-source matching."""
    return name.lower().strip()


@st.cache_data(ttl=120)
def load_today_matches(circuit: str = "ATP") -> list[dict]:
    """
    ATP:  Matchstat is primary (full odds, 10+ matches).
          Bzzoiro supplements each match with bzz_* prediction fields,
          joined by normalised player name pair.
          Falls back to Bzzoiro-only if Matchstat is unavailable.
    WTA:  Bzzoiro only (Matchstat has no WTA data).
    """

    # ── WTA: Bzzoiro only ──────────────────────────────────────────────────────
    if circuit != "ATP":
        try:
            from bzzoiro_api import get_today_matches_with_predictions
            return get_today_matches_with_predictions(circuit=circuit)
        except Exception as e:
            st.warning(f"Could not load WTA schedule: {e}")
            return []

    # ── ATP step 1: Matchstat primary ──────────────────────────────────────────
    matchstat_matches: list[dict] = []
    try:
        from matchstat_api import get_today_odds, has_upcoming_matches
        if has_upcoming_matches():
            matchstat_matches = get_today_odds()
    except Exception as e:
        msg = str(e)
        if "temporarily unavailable" in msg or "authentication failed" in msg or "budget" in msg.lower():
            st.warning(f"Matchstat unavailable: {msg}")
        else:
            st.warning(f"Could not load live odds: {msg}")

    # ── ATP step 2: Bzzoiro predictions supplement ─────────────────────────────
    bzz_by_names: dict[tuple, dict] = {}
    try:
        from bzzoiro_api import get_today_matches_with_predictions
        for bm in get_today_matches_with_predictions(circuit="ATP"):
            key = (_norm(bm.get("player1_name", "")), _norm(bm.get("player2_name", "")))
            bzz_by_names[key] = bm
    except Exception:
        pass  # predictions are optional — Matchstat data still shown without them

    if matchstat_matches:
        for m in matchstat_matches:
            p1  = _norm(m.get("player1_name", ""))
            p2  = _norm(m.get("player2_name", ""))
            bm  = bzz_by_names.get((p1, p2))
            rev = bzz_by_names.get((p2, p1))    # Bzzoiro may list players reversed
            if bm or rev:
                src  = bm or rev
                flip = rev is not None and bm is None
                m["bzz_prob_p1"]          = src.get("bzz_prob_p2" if flip else "bzz_prob_p1")
                m["bzz_prob_p2"]          = src.get("bzz_prob_p1" if flip else "bzz_prob_p2")
                m["bzz_confidence"]       = src.get("bzz_confidence")
                m["bzz_predicted_winner"] = src.get("bzz_predicted_winner")
                m["bzz_expected_sets"]    = src.get("bzz_expected_sets")
                m["bzz_expected_games"]   = src.get("bzz_expected_games")
                m["bzz_prob_over_22_5"]   = src.get("bzz_prob_over_22_5")
        return matchstat_matches

    # ── Fallback: Bzzoiro only if Matchstat returned nothing ───────────────────
    return list(bzz_by_names.values())


@st.cache_data(ttl=30)   # live scores — match server-side 30 s TTL
def load_live_scores(circuit: str = "ATP") -> list[dict]:
    try:
        from bzzoiro_api import get_live_matches
        return get_live_matches(circuit=circuit)
    except Exception:
        return []


@st.cache_data(ttl=300)
def load_atp_rankings_live(top: int = 200) -> pd.DataFrame:
    """Live official ATP rankings from Bzzoiro. Falls back to empty DataFrame."""
    try:
        from bzzoiro_api import get_rankings
        rows = get_rankings(circuit="ATP", top=top)
        return pd.DataFrame([
            {
                "Rank":    r["ranking"],
                "Player":  r["player"]["name"],
                "Country": r["player"].get("country", ""),
                "Points":  r["points"],
            }
            for r in rows
        ])
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=120)
def load_flashscore_odds() -> dict[tuple, dict]:
    """
    Fetches live Flashscore odds for today's matches.
    Returns a dict keyed by (_norm(p1), _norm(p2)) → best-odds row dict,
    selecting the bookmaker with the tightest overround per match.
    Falls back to an empty dict on any error.
    """
    try:
        from flashscore_odds import fetch_flashscore_odds
        rows = fetch_flashscore_odds(request_delay_s=0.3)
    except Exception:
        return {}
    by_match: dict[tuple, dict] = {}
    for row in rows:
        o1, o2 = row.get("p1_odds"), row.get("p2_odds")
        if not (o1 and o2 and o1 > 1 and o2 > 1):
            continue
        key = (_norm(row["p1"]), _norm(row["p2"]))
        if key in by_match:
            ex = by_match[key]
            ex_o1, ex_o2 = ex.get("p1_odds", 1), ex.get("p2_odds", 1)
            if (1 / ex_o1 + 1 / ex_o2) <= (1 / o1 + 1 / o2):
                continue  # existing row has equal or tighter margin — keep it
        by_match[key] = row
    return by_match


@st.cache_data(ttl=3600)
def load_odds_api_today() -> dict[tuple, dict]:
    """
    Load today's cached Odds API tennis odds from data_files/odds_api_today.json.
    Returns a dict keyed by (_norm(p1), _norm(p2)) → match dict.
    Falls back to empty dict if file is missing, stale, or unreadable.
    The file is written once per day by fetch_odds_api.py (nightly workflow).
    """
    path = os.path.join(DATA_DIR, "odds_api_today.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as fh:
            payload = json.load(fh)
        if payload.get("date_fetched") != str(date.today()):
            return {}  # stale — nightly job hasn't refreshed yet
        result: dict[tuple, dict] = {}
        for m in payload.get("matches", []):
            p1 = m.get("player1", "")
            p2 = m.get("player2", "")
            result[(_norm(p1), _norm(p2))] = m
        return result
    except Exception:
        return {}


features            = load_features()
today_matches       = load_today_matches("ATP")
wta_matches         = load_today_matches("WTA")
live_scores_atp     = load_live_scores("ATP")
predictor           = load_predictor()
model_data          = load_model_data()
flashscore_odds_map = load_flashscore_odds()
odds_api_map        = load_odds_api_today()

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.match-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 0.75rem 1.25rem;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 0.5rem;
}
.match-card:hover { box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
.winner-highlight {
    background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
}
.prediction-high   { color: #059669; font-weight: 600; }
.prediction-medium { color: #d97706; font-weight: 600; }
.prediction-low    { color: #9ca3af; }
.status-dot {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    margin-right: 6px;
    vertical-align: middle;
}
.status-live     { background: #ef4444; animation: pulse-red 1s infinite; }
.status-starting { background: #f59e0b; }
.status-upcoming { background: #22c55e; }
@keyframes pulse-red {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.4; }
}
.tournament-group-header {
    font-size: 1.1rem;
    font-weight: 700;
    color: #1e40af;
    margin: 1.25rem 0 0.6rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 2px solid #dbeafe;
}
.slip-card {
    background: #1a1a2e;
    border-radius: 16px;
    padding: 1.5rem 2rem;
    color: #e0e0e0;
    font-family: 'Courier New', monospace;
    max-width: 520px;
    margin: 1rem auto;
}
.slip-header h2 { color: #38ef7d; letter-spacing: 2px; margin: 0 0 1rem 0; }
.slip-winner    { color: #38ef7d; font-weight: 700; font-size: 1rem; }
.slip-confidence{ color: #f59e0b; }
.confidence-meter {
    height: 28px;
    border-radius: 14px;
    background: #e0e0e0;
    overflow: hidden;
    margin: 0.75rem 0;
}
.confidence-fill { height: 100%; border-radius: 14px; transition: width 0.5s ease; }
</style>
""", unsafe_allow_html=True)


# ── Shared helper functions ────────────────────────────────────────────────────
def latest_elo(name: str) -> float | None:
    if features.empty:
        return None
    mask = (features["winner_name"] == name) | (features["loser_name"] == name)
    sub = features[mask]
    if sub.empty:
        return None
    last = sub.iloc[-1]
    if last["winner_name"] == name:
        return round(last["elo_pre_w"], 0)
    return round(last["elo_pre_l"], 0)


def norm_surface(s: str) -> str:
    if not s or s == "" or s == "None":
        return "—"
    s2 = s.lower().replace("i.", "").strip()
    if "hard" in s2:
        return "Hard"
    if "clay" in s2:
        return "Clay"
    if "grass" in s2:
        return "Grass"
    return s.strip().title()


def has_match_data(name: str) -> bool:
    """True if the player appears at least once in the features parquet."""
    if features.empty:
        return False
    return (
        features["winner_name"].eq(name).any()
        or features["loser_name"].eq(name).any()
    )


def render_confidence_meter(confidence: float) -> None:
    """Reusable confidence meter widget — renders a colored bar + HIGH/MEDIUM/LOW label."""
    if confidence >= 0.75:
        color, label = "#11998e", "HIGH"
    elif confidence >= 0.65:
        color, label = "#f7971e", "MEDIUM"
    else:
        color, label = "#eb3349", "LOW"
    st.markdown(
        f'<div class="confidence-meter">'
        f'<div class="confidence-fill" style="width:{confidence * 100:.1f}%;background:{color}"></div>'
        f'</div>'
        f'<p style="text-align:center;margin:0">'
        f'<strong>Confidence: {confidence:.1%}</strong> ({label})</p>',
        unsafe_allow_html=True,
    )

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_today, tab_wta, tab_backlog, tab_predict, tab_player, tab_elo, tab_data, tab_model = st.tabs([
    "📡 Today's ATP", "🎾 Today's WTA", "📋 Prediction Backlog", "🔮 Match Prediction",
    "👤 Player Analysis", "📊 ELO Rankings", "📂 Match Explorer", "📈 Model Stats",
])


# ══════════════════════════════════════════════════════════════════════════════
# Tab 1 — Today's Matches
# ══════════════════════════════════════════════════════════════════════════════
with tab_today:
    st.subheader(f"ATP · {date.today().strftime('%A, %B %d %Y')}")

    # ── Live scores ticker ────────────────────────────────────────────────────
    if live_scores_atp:
        st.markdown("### 🔴 Live Now")
        for _lm in live_scores_atp:
            _lp1    = (_lm.get("player1_obj") or {}).get("name", _lm.get("player1", "P1"))
            _lp2    = (_lm.get("player2_obj") or {}).get("name", _lm.get("player2", "P2"))
            _ls1    = _lm.get("player1_sets", 0)
            _ls2    = _lm.get("player2_sets", 0)
            _is_p1_srv = _lm.get("is_serving_p1")
            _ll_p1  = f"🎾 {_lp1}" if _is_p1_srv is True else _lp1
            _ll_p2  = f"🎾 {_lp2}" if _is_p1_srv is False else _lp2
            _gp1  = _lm.get("current_game_p1", "")
            _gp2  = _lm.get("current_game_p2", "")
            _lgame = f"{_gp1}-{_gp2}" if (_gp1 != "" and _gp2 != "") else ""
            st.markdown(
                f'<div class="match-card">'
                f'<span><strong>{_ll_p1}</strong>&nbsp;{_ls1} – {_ls2}&nbsp;<strong>{_ll_p2}</strong></span>'
                f'<span style="color:#ef4444;font-weight:700">LIVE&nbsp;&nbsp;</span>'
                f'<span style="color:#6b7280">{_lgame}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.markdown("---")

    if not today_matches:
        st.info("No upcoming ATP singles matches found for today, or API unavailable.")
    else:
        # Enrich each match with status and model probabilities
        enriched = []
        for m in today_matches:
            p1, p2 = m["player1_name"], m["player2_name"]
            o1, o2 = m["odds_p1"], m["odds_p2"]
            surf   = norm_surface(m.get("surface"))

            # ── Flashscore odds supplement (fallback when Matchstat has none) ──
            _fs_key = (_norm(p1), _norm(p2))
            _fs_rev = (_norm(p2), _norm(p1))
            _fs = flashscore_odds_map.get(_fs_key) or flashscore_odds_map.get(_fs_rev)
            _fs_flipped = _fs is not None and _fs_key not in flashscore_odds_map
            if _fs:
                fs_o1 = _fs.get("p2_odds" if _fs_flipped else "p1_odds")
                fs_o2 = _fs.get("p1_odds" if _fs_flipped else "p2_odds")
                fs_bk = _fs.get("bookmaker_name")
            else:
                fs_o1 = fs_o2 = fs_bk = None
            # ── Odds API supplement ───────────────────────────────────────────
            _oa = odds_api_map.get(_fs_key) or odds_api_map.get(_fs_rev)
            _oa_flipped = _oa is not None and _fs_key not in odds_api_map
            if _oa:
                oa_o1 = _oa.get("player2_odds" if _oa_flipped else "player1_odds")
                oa_o2 = _oa.get("player1_odds" if _oa_flipped else "player2_odds")
            else:
                oa_o1 = oa_o2 = None
            # Effective odds: Matchstat → Odds API → Flashscore
            eff_o1 = o1 if (o1 and o1 > 1) else oa_o1 if (oa_o1 and oa_o1 > 1) else fs_o1
            eff_o2 = o2 if (o2 and o2 > 1) else oa_o2 if (oa_o2 and oa_o2 > 1) else fs_o2

            # Market implied probability (devigged)
            if eff_o1 and eff_o2 and eff_o1 > 1 and eff_o2 > 1:
                raw1, raw2 = 1 / eff_o1, 1 / eff_o2
                tot  = raw1 + raw2
                mkt1 = raw1 / tot
            else:
                mkt1 = None

            # Model probability
            model_reliable = (
                predictor is not None
                and surf != "—"
                and (has_match_data(p1) or has_match_data(p2))
            )
            if model_reliable:
                try:
                    _w, p1_prob = predictor.predict(p1, p2, surf, market_prob_p1=mkt1)
                except Exception:
                    p1_prob = mkt1
            else:
                p1_prob = mkt1

            # Status (API returns only upcoming fixtures; default accordingly)
            raw_status = str(m.get("status", "upcoming")).lower()
            if any(k in raw_status for k in ("live", "progress", "playing")):
                status = "live"
            elif any(k in raw_status for k in ("soon", "starting")):
                status = "soon"
            else:
                status = "upcoming"

            enriched.append({
                **m,
                "status":      status,
                "p1_prob":     p1_prob,
                "elo1":        latest_elo(p1),
                "elo2":        latest_elo(p2),
                "fs_p1_odds":  fs_o1,
                "fs_p2_odds":  fs_o2,
                "fs_bookmaker": fs_bk,
            })

        # ── Summary metrics ────────────────────────────────────────────────────
        n_total    = len(enriched)
        n_live     = sum(1 for m in enriched if m["status"] == "live")
        n_soon     = sum(1 for m in enriched if m["status"] == "soon")
        n_upcoming = sum(1 for m in enriched if m["status"] == "upcoming")

        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.metric("Total Matches", n_total)
        with mc2:
            st.metric("🔴 Live", n_live)
        with mc3:
            st.metric("🟠 Starting Soon", n_soon)
        with mc4:
            st.metric("🟢 Upcoming", n_upcoming)

        # ── Status filter pills ────────────────────────────────────────────────
        status_filter = st.pills(
            "Filter by status",
            ["All", "🔴 Live", "🟠 Soon", "🟢 Upcoming"],
            default="All",
            selection_mode="single",
        )
        if status_filter and status_filter != "All":
            _key = status_filter.split()[-1].lower()
            display_matches = [m for m in enriched if m["status"] == _key]
        else:
            display_matches = enriched

        # ── Bookie slip ────────────────────────────────────────────────────────
        high_conf_picks = [
            m for m in enriched
            if m.get("p1_prob") is not None
            and (m["p1_prob"] >= 0.75 or (1 - m["p1_prob"]) >= 0.75)
        ]
        if high_conf_picks:
            if "show_slip" not in st.session_state:
                st.session_state.show_slip = False
            slip_label = "✕ Close Slip" if st.session_state.show_slip else "🎫 Generate Bookie Slip"
            if st.button(slip_label, help="High-confidence picks (model ≥75%)"):
                st.session_state.show_slip = not st.session_state.show_slip
                st.rerun()
            if st.session_state.show_slip:
                slip_html = (
                    '<div class="slip-card">'
                    '<div class="slip-header"><h2>🎾 HIGH CONFIDENCE PICKS</h2></div>'
                )
                for m in high_conf_picks:
                    prob = m["p1_prob"]
                    if prob is not None and (1 - prob) > prob:
                        pick, conf = m["player2_name"], 1 - prob
                    else:
                        pick, conf = m["player1_name"], prob or 0.5
                    slip_html += (
                        f'<p><span class="slip-winner">✅ {pick}</span> '
                        f'<span class="slip-confidence">({conf:.0%})</span></p>'
                    )
                slip_html += "</div>"
                st.markdown(slip_html, unsafe_allow_html=True)

        # ── Match cards grouped by tournament ──────────────────────────────────
        by_tournament: dict[str, list] = {}
        for m in display_matches:
            t = m.get("tournament") or "Unknown Tournament"
            by_tournament.setdefault(t, []).append(m)

        for tournament, matches in by_tournament.items():
            surf_label = norm_surface(matches[0].get("surface"))
            st.markdown(
                f'<div class="tournament-group-header">'
                f'🏆 {tournament} · {surf_label}</div>',
                unsafe_allow_html=True,
            )
            for m in matches:
                p1, p2      = m["player1_name"], m["player2_name"]
                o1, o2      = m["odds_p1"],      m["odds_p2"]
                fs_o1, fs_o2 = m.get("fs_p1_odds"), m.get("fs_p2_odds")
                fs_bk        = m.get("fs_bookmaker")
                # Use Flashscore odds in display when Matchstat odds are absent
                disp_o1 = o1 if (o1 and o1 > 1) else fs_o1
                disp_o2 = o2 if (o2 and o2 > 1) else fs_o2
                using_fs_odds = not (o1 and o1 > 1) and bool(disp_o1)
                prob        = m.get("p1_prob")
                e1, e2      = m.get("elo1"),      m.get("elo2")
                status      = m["status"]
                rnd         = m.get("round") or ""

                # Status dot
                if status == "live":
                    dot, status_label = '<span class="status-dot status-live"></span>', "LIVE"
                elif status == "soon":
                    dot, status_label = '<span class="status-dot status-starting"></span>', "SOON"
                else:
                    dot, status_label = '<span class="status-dot status-upcoming"></span>', rnd or "Upcoming"

                # Prediction label with confidence tiers
                if prob is not None:
                    p2_prob = 1 - prob
                    if prob >= 0.75:
                        pred_html = f'<span class="prediction-high">{p1} {prob:.0%} ✅</span>'
                    elif prob >= 0.65:
                        pred_html = f'<span class="prediction-medium">{p1} {prob:.0%}</span>'
                    elif p2_prob >= 0.75:
                        pred_html = f'<span class="prediction-high">{p2} {p2_prob:.0%} ✅</span>'
                    elif p2_prob >= 0.65:
                        pred_html = f'<span class="prediction-medium">{p2} {p2_prob:.0%}</span>'
                    else:
                        pred_html = '<span class="prediction-low">skip</span>'
                else:
                    pred_html = '<span class="prediction-low">no prediction</span>'

                elo_str  = f" · ELO {e1:.0f} vs {e2:.0f}" if (e1 and e2) else ""
                odds_str = f" · {disp_o1:.2f} / {disp_o2:.2f}" if (disp_o1 and disp_o2) else ""

                st.markdown(
                    f'<div class="match-card">'
                    f'<span>{dot} <strong>{p1}</strong> vs <strong>{p2}</strong>'
                    f'<small style="color:#888">{elo_str}{odds_str}</small></span>'
                    f'<span>{pred_html}&nbsp;<small style="color:#aaa">{status_label}</small></span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                # Bzzoiro second-opinion overlay
                # prob values are 0-1 fractions (same scale as model prob)
                _bzz_p1  = m.get("bzz_prob_p1")
                _bzz_conf = m.get("bzz_confidence")
                if _bzz_p1 is not None and _bzz_conf is not None:
                    _bzz_sets  = m.get("bzz_expected_sets")
                    _bzz_games = m.get("bzz_expected_games")
                    _detail = ""
                    if _bzz_sets:
                        _detail += f" · {float(_bzz_sets):.1f} sets"
                    if _bzz_games:
                        _detail += f" / {float(_bzz_games):.1f} games"
                    st.caption(
                        f"🤖 Bzzoiro: {p1} {float(_bzz_p1) * 100:.0f}% · "
                        f"confidence {float(_bzz_conf) * 100:.0f}%{_detail}"
                    )
                # Flashscore odds caption (shown when used as fallback)
                if using_fs_odds and fs_o1 and fs_o2:
                    _bk_label = f" ({fs_bk})" if fs_bk else ""
                    st.caption(f"📊 Flashscore{_bk_label}: {disp_o1:.2f} / {disp_o2:.2f}")

        # ── Legend ─────────────────────────────────────────────────────────────
        st.markdown(
            '<div style="margin-top:1rem;font-size:12px;color:#888">'
            '<span class="status-dot status-live"></span> Live &nbsp;'
            '<span class="status-dot status-starting"></span> Starting soon &nbsp;'
            '<span class="status-dot status-upcoming"></span> Upcoming &nbsp;|&nbsp;'
            ' ✅ High confidence ≥75% · 🟠 Medium 65–75% · <i>skip</i> &lt;65%'
            '</div>',
            unsafe_allow_html=True,
        )

        try:
            from matchstat_api import calls_used_this_month, calls_remaining
            st.caption(
                f"API calls this month: {calls_used_this_month()} / 500 "
                f"({calls_remaining()} remaining)"
            )
        except Exception:
            pass

    add_betting_oracle_footer()


# ══════════════════════════════════════════════════════════════════════════════
# Tab 2 — Today's WTA
# ══════════════════════════════════════════════════════════════════════════════
with tab_wta:
    st.subheader(f"WTA · {date.today().strftime('%A, %B %d %Y')}")

    # ── WTA live scores ───────────────────────────────────────────────────────
    wta_live = load_live_scores("WTA")
    if wta_live:
        st.markdown("### 🔴 Live Now")
        for _lm in wta_live:
            _lp1   = (_lm.get("player1_obj") or {}).get("name", _lm.get("player1", "P1"))
            _lp2   = (_lm.get("player2_obj") or {}).get("name", _lm.get("player2", "P2"))
            _ls1   = _lm.get("player1_sets", 0)
            _ls2   = _lm.get("player2_sets", 0)
            _is_p1_srv = _lm.get("is_serving_p1")
            _ll_p1 = f"🎾 {_lp1}" if _is_p1_srv is True else _lp1
            _ll_p2 = f"🎾 {_lp2}" if _is_p1_srv is False else _lp2
            _gp1 = _lm.get("current_game_p1", "")
            _gp2 = _lm.get("current_game_p2", "")
            _lgame = f"{_gp1}-{_gp2}" if (_gp1 != "" and _gp2 != "") else ""
            st.markdown(
                f'<div class="match-card">'
                f'<span><strong>{_ll_p1}</strong>&nbsp;{_ls1} – {_ls2}&nbsp;<strong>{_ll_p2}</strong></span>'
                f'<span style="color:#ef4444;font-weight:700">LIVE&nbsp;&nbsp;</span>'
                f'<span style="color:#6b7280">{_lgame}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.markdown("---")

    if not wta_matches:
        st.info("No upcoming WTA singles matches found for today, or API unavailable.")
    else:
        by_t_wta: dict[str, list] = {}
        for _m in wta_matches:
            _t = _m.get("tournament") or "Unknown Tournament"
            by_t_wta.setdefault(_t, []).append(_m)

        wta_total = len(wta_matches)
        st.metric("WTA Matches Today", wta_total)

        for _t_name, _t_matches in by_t_wta.items():
            _surf_label = norm_surface(_t_matches[0].get("surface"))
            st.markdown(
                f'<div class="tournament-group-header">'
                f'🏆 {_t_name} · {_surf_label}</div>',
                unsafe_allow_html=True,
            )
            for _m in _t_matches:
                _p1, _p2 = _m["player1_name"], _m["player2_name"]
                _o1, _o2 = _m.get("odds_p1"), _m.get("odds_p2")
                _rnd     = _m.get("round") or ""
                _bzz_p1  = _m.get("bzz_prob_p1")
                _bzz_conf = _m.get("bzz_confidence")

                # Flashscore odds supplement for WTA
                _fs_key  = (_norm(_p1), _norm(_p2))
                _fs_rev_wta = (_norm(_p2), _norm(_p1))
                _fs = flashscore_odds_map.get(_fs_key) or flashscore_odds_map.get(_fs_rev_wta)
                _fs_flipped = _fs is not None and _fs_key not in flashscore_odds_map
                if _fs:
                    _fs_o1 = _fs.get("p2_odds" if _fs_flipped else "p1_odds")
                    _fs_o2 = _fs.get("p1_odds" if _fs_flipped else "p2_odds")
                    _fs_bk = _fs.get("bookmaker_name")
                else:
                    _fs_o1 = _fs_o2 = _fs_bk = None
                # Odds API supplement for WTA
                _oa_wta = odds_api_map.get(_fs_key) or odds_api_map.get(_fs_rev_wta)
                _oa_wta_flipped = _oa_wta is not None and _fs_key not in odds_api_map
                if _oa_wta:
                    _oa_o1 = _oa_wta.get("player2_odds" if _oa_wta_flipped else "player1_odds")
                    _oa_o2 = _oa_wta.get("player1_odds" if _oa_wta_flipped else "player2_odds")
                else:
                    _oa_o1 = _oa_o2 = None
                _disp_o1 = _o1 if (_o1 and _o1 > 1) else _oa_o1 if (_oa_o1 and _oa_o1 > 1) else _fs_o1
                _disp_o2 = _o2 if (_o2 and _o2 > 1) else _oa_o2 if (_oa_o2 and _oa_o2 > 1) else _fs_o2
                _using_fs = not (_o1 and _o1 > 1) and bool(_disp_o1)

                # Probability — Bzzoiro predictions are already 0-1 fractions
                if _bzz_p1 is not None:
                    _prob = float(_bzz_p1)
                elif _disp_o1 and _disp_o2 and _disp_o1 > 1 and _disp_o2 > 1:
                    _raw1, _raw2 = 1 / _disp_o1, 1 / _disp_o2
                    _prob = _raw1 / (_raw1 + _raw2)
                else:
                    _prob = None

                _odds_str = f" · {_disp_o1:.2f} / {_disp_o2:.2f}" if (_disp_o1 and _disp_o2) else ""

                if _prob is not None:
                    _p2_prob = 1 - _prob
                    if _prob >= 0.75:
                        _pred_html = f'<span class="prediction-high">{_p1} {_prob:.0%} ✅</span>'
                    elif _prob >= 0.65:
                        _pred_html = f'<span class="prediction-medium">{_p1} {_prob:.0%}</span>'
                    elif _p2_prob >= 0.75:
                        _pred_html = f'<span class="prediction-high">{_p2} {_p2_prob:.0%} ✅</span>'
                    elif _p2_prob >= 0.65:
                        _pred_html = f'<span class="prediction-medium">{_p2} {_p2_prob:.0%}</span>'
                    else:
                        _pred_html = '<span class="prediction-low">skip</span>'
                else:
                    _pred_html = '<span class="prediction-low">no prediction</span>'

                st.markdown(
                    f'<div class="match-card">'
                    f'<span><span class="status-dot status-upcoming"></span>'
                    f'<strong>{_p1}</strong> vs <strong>{_p2}</strong>'
                    f'<small style="color:#888">{_odds_str}</small></span>'
                    f'<span>{_pred_html}&nbsp;<small style="color:#aaa">{_rnd}</small></span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                if _bzz_p1 is not None and _bzz_conf is not None:
                    _bzz_sets  = _m.get("bzz_expected_sets")
                    _bzz_games = _m.get("bzz_expected_games")
                    _detail = ""
                    if _bzz_sets:
                        _detail += f" · {float(_bzz_sets):.1f} sets"
                    if _bzz_games:
                        _detail += f" / {float(_bzz_games):.1f} games"
                    st.caption(
                        f"🤖 Bzzoiro: {_p1} {float(_bzz_p1) * 100:.0f}% · "
                        f"confidence {float(_bzz_conf) * 100:.0f}%{_detail}"
                    )
                # Flashscore odds caption (shown when used as fallback for WTA)
                if _using_fs and _fs_o1 and _fs_o2:
                    _bk_lbl = f" ({_fs_bk})" if _fs_bk else ""
                    st.caption(f"📊 Flashscore{_bk_lbl}: {_disp_o1:.2f} / {_disp_o2:.2f}")

    add_betting_oracle_footer()


# ══════════════════════════════════════════════════════════════════════════════
# Tab 3 — Prediction Backlog
# ══════════════════════════════════════════════════════════════════════════════
with tab_backlog:
    st.subheader("📋 Prediction Backlog")
    st.caption("Recent model predictions vs actual match outcomes")

    BACKLOG_FILE = os.path.join(DATA_DIR, "prediction_backlog.parquet")

    if features.empty:
        st.info("No match data available. Run `python features.py` to build the feature matrix.")
    else:
        # ── Load pre-computed backlog if available (built by GH Action) ─────
        if os.path.exists(BACKLOG_FILE):
            recent = pd.read_parquet(BACKLOG_FILE)
            recent["tourney_date"] = pd.to_datetime(recent["tourney_date"])
        else:
            # Fallback: compute on-the-fly (slower, runs only if artifact missing)
            cutoff = pd.Timestamp.today() - pd.Timedelta(days=14)
            recent = features[features["tourney_date"] >= cutoff].copy()
            if len(recent) < 10:
                recent = features.tail(200).copy()

            if predictor is not None and (
                "model_prob_w" not in recent.columns or recent["model_prob_w"].isna().all()
            ):
                def _p(row):
                    try:
                        _, prob = predictor.predict(row["winner_name"], row["loser_name"], row["surface"])
                        return prob
                    except Exception:
                        return np.nan

                with st.spinner("Computing model predictions (one-time, slow)…"):
                    recent["model_prob_w"] = recent.apply(_p, axis=1)

        if "model_prob_w" not in recent.columns:
            recent["model_prob_w"] = np.nan

        valid = recent.dropna(subset=["model_prob_w"]).copy()
        valid["model_correct"] = valid["model_prob_w"] >= 0.5
        valid["tier"] = pd.cut(
            valid["model_prob_w"],
            bins=[0, 0.65, 0.75, 1.0],
            labels=["Low (<65%)", "Medium (65–75%)", "High (≥75%)"],
        )

        # Summary metrics
        sc1, sc2, sc3, sc4 = st.columns(4)
        high_v = valid[valid["model_prob_w"] >= 0.75]
        with sc1:
            st.metric("Matches", len(recent))
        with sc2:
            st.metric("Predicted", len(valid))
        with sc3:
            acc = valid["model_correct"].mean() if len(valid) else 0
            st.metric("Overall Accuracy", f"{acc:.1%}" if len(valid) else "—")
        with sc4:
            hacc = high_v["model_correct"].mean() if len(high_v) else 0
            st.metric("High-Conf Accuracy", f"{hacc:.1%}" if len(high_v) else "—")

        # Charts
        if len(valid) > 0:
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                st.markdown("#### Accuracy by Confidence Tier")
                acc_by_tier = valid.groupby("tier", observed=True)["model_correct"].mean()
                fig_tier = go.Figure(go.Bar(
                    x=acc_by_tier.index.astype(str),
                    y=acc_by_tier.values,
                    marker_color=["#9ca3af", "#d97706", "#059669"],
                    text=[f"{v:.1%}" for v in acc_by_tier.values],
                    textposition="outside",
                ))
                fig_tier.update_layout(
                    yaxis=dict(range=[0, 1.05], tickformat=".0%"),
                    margin=dict(t=10, b=30), height=280,
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_tier, width="stretch", key="backlog_tier")

            with col_c2:
                st.markdown("#### Match Breakdown by Tier")
                tier_counts = valid["tier"].value_counts().sort_index()
                fig_bk = go.Figure(go.Bar(
                    x=tier_counts.index.astype(str),
                    y=tier_counts.values,
                    marker_color=["#9ca3af", "#d97706", "#059669"],
                    text=tier_counts.values,
                    textposition="outside",
                ))
                fig_bk.update_layout(
                    margin=dict(t=10, b=30), height=280,
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_bk, width="stretch", key="backlog_breakdown")

        # Filter pills
        filter_opt = st.pills(
            "Filter results",
            ["All", "✓ Correct", "✗ Wrong", "High Conf Only"],
            default="All",
            selection_mode="single",
        )
        filt = valid.copy()
        if filter_opt == "✓ Correct":
            filt = filt[filt["model_correct"] == True]
        elif filter_opt == "✗ Wrong":
            filt = filt[filt["model_correct"] == False]
        elif filter_opt == "High Conf Only":
            filt = filt[filt["model_prob_w"] >= 0.75]

        for _, row in filt.sort_values("tourney_date", ascending=False).head(50).iterrows():
            correct       = bool(row["model_correct"])
            result_icon   = "✓" if correct else "✗"
            result_color  = "#059669" if correct else "#dc2626"
            prob          = row["model_prob_w"]
            tier_cls      = (
                "prediction-high"   if prob >= 0.75 else
                "prediction-medium" if prob >= 0.65 else
                "prediction-low"
            )
            st.markdown(
                f'<div class="match-card">'
                f'<span><strong>{row["winner_name"]}</strong> def. {row["loser_name"]}'
                f'&nbsp;<small style="color:#888">{row.get("tourney_name","")} '
                f'· {row["surface"]}</small></span>'
                f'<span><span class="{tier_cls}">{prob:.0%}</span>'
                f'&nbsp;<strong style="color:{result_color}">{result_icon}</strong></span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    add_betting_oracle_footer()


# ══════════════════════════════════════════════════════════════════════════════
# Tab 3 — Match Prediction
# ══════════════════════════════════════════════════════════════════════════════
with tab_predict:
    st.subheader("🔮 Match Prediction")

    if predictor is None:
        st.warning("No trained model found. Run `python train.py` to train the model.")
    elif features.empty:
        st.warning("Feature matrix not found. Run `python features.py` first.")
    else:
        all_players = sorted(
            set(features["winner_name"].tolist() + features["loser_name"].tolist())
        )

        col_p1, col_surf, col_p2 = st.columns([5, 2, 5])
        with col_p1:
            p1_sel = st.selectbox("Player 1", all_players, key="pred_p1")
        with col_surf:
            surf_sel = st.selectbox("Surface", ["Hard", "Clay", "Grass"], key="pred_surf")
        with col_p2:
            p2_sel = st.selectbox(
                "Player 2", all_players,
                index=min(1, len(all_players) - 1),
                key="pred_p2",
            )

        if p1_sel and p2_sel and p1_sel != p2_sel:
            try:
                _w, p1_prob = predictor.predict(p1_sel, p2_sel, surf_sel)
                p2_prob = 1 - p1_prob
                confidence = max(p1_prob, p2_prob)
                winner     = p1_sel if p1_prob >= p2_prob else p2_sel

                conf_color = "#11998e" if confidence >= 0.75 else "#f7971e" if confidence >= 0.65 else "#eb3349"

                # Confidence meter widget
                render_confidence_meter(confidence)

                # Winner card
                st.markdown(
                    f'<div class="winner-highlight" style="margin:1rem 0">'
                    f'<h3 style="margin:0">🏆 Predicted Winner: {winner}</h3>'
                    f'<p style="margin:0.5rem 0 0 0;opacity:0.9">{confidence:.1%} probability</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # Player comparison (ELO + win probability side by side)
                e1 = latest_elo(p1_sel)
                e2 = latest_elo(p2_sel)
                cmp_l, cmp_r = st.columns(2)
                with cmp_l:
                    st.markdown(f"#### {p1_sel}")
                    if e1:
                        st.metric("ELO", f"{e1:.0f}")
                    st.metric("Win Probability", f"{p1_prob:.1%}")
                with cmp_r:
                    st.markdown(f"#### {p2_sel}")
                    if e2:
                        st.metric("ELO", f"{e2:.0f}")
                    st.metric("Win Probability", f"{p2_prob:.1%}")

                # Horizontal probability bar chart
                fig_prob = go.Figure()
                fig_prob.add_trace(go.Bar(
                    y=["Win Probability"],
                    x=[p1_prob],
                    name=p1_sel,
                    orientation="h",
                    marker_color=conf_color if p1_prob >= p2_prob else "#9ca3af",
                    text=[f"{p1_prob:.1%}"],
                    textposition="inside",
                    textfont=dict(color="white"),
                ))
                fig_prob.add_trace(go.Bar(
                    y=["Win Probability"],
                    x=[p2_prob],
                    name=p2_sel,
                    orientation="h",
                    marker_color=conf_color if p2_prob > p1_prob else "#9ca3af",
                    text=[f"{p2_prob:.1%}"],
                    textposition="inside",
                    textfont=dict(color="white"),
                ))
                fig_prob.update_layout(
                    barmode="stack",
                    xaxis=dict(range=[0, 1], tickformat=".0%", visible=False),
                    height=80,
                    margin=dict(l=0, r=0, t=20, b=0),
                    legend=dict(orientation="h", y=-1.0, x=0),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_prob, width="stretch", key="predict_prob")

            except Exception as e:
                st.error(f"Prediction failed: {e}")

        elif p1_sel == p2_sel:
            st.warning("Select two different players.")

    add_betting_oracle_footer()


# ══════════════════════════════════════════════════════════════════════════════
# Tab 4 — Match Explorer
# ══════════════════════════════════════════════════════════════════════════════
with tab_data:
    st.subheader("Historical Match Data (2020 – present)")

    if features.empty:
        st.warning(
            f"`{FEATURES_FILE}` not found. Run `python features.py` to build it."
        )
    else:
        st.markdown(
            f"**{len(features):,} matches** · "
            f"{features['tourney_date'].min().date()} to {features['tourney_date'].max().date()} · "
            f"**{features['mkt_prob_w'].notna().sum():,}** rows with market odds"
        )

        # ── Filters ───────────────────────────────────────────────────────────
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            years = sorted(features["tourney_date"].dt.year.unique(), reverse=True)
            sel_years = st.multiselect("Year", years, default=years[:2])
        with col2:
            surfaces = sorted(features["surface"].dropna().unique())
            sel_surf = st.multiselect("Surface", surfaces, default=surfaces)
        with col3:
            levels_raw = features["tourney_level"].dropna().unique()
            level_labels = {"G": "Grand Slam (G)", "M": "Masters (M)",
                            "A": "ATP 500 (A)", "B": "ATP 250 (B)",
                            "C": "Challenger (C)", "F": "Finals (F)"}
            sel_level = st.multiselect(
                "Level",
                sorted(levels_raw),
                default=sorted(levels_raw),
                format_func=lambda x: level_labels.get(x, x),
            )
        with col4:
            player_filter = st.text_input("Player name contains", "")

        # ── Apply filters ─────────────────────────────────────────────────────
        filtered = features.copy()
        if sel_years:
            filtered = filtered[filtered["tourney_date"].dt.year.isin(sel_years)]
        if sel_surf:
            filtered = filtered[filtered["surface"].isin(sel_surf)]
        if sel_level:
            filtered = filtered[filtered["tourney_level"].isin(sel_level)]
        if player_filter:
            mask = (
                filtered["winner_name"].str.contains(player_filter, case=False, na=False) |
                filtered["loser_name"].str.contains(player_filter, case=False, na=False)
            )
            filtered = filtered[mask]

        st.caption(f"{len(filtered):,} matches match filters")

        # ── optionally compute model probabilities & edge for filtered history
        compute_model = False
        if predictor is not None and "mkt_prob_w" in filtered.columns:
            compute_model = st.checkbox(
                "Compute model probabilities and edge for these matches",
                value=False,
                help="This may be slow for large result sets; filter down to reduce time."
            )
        if compute_model:
            # to avoid long loops on massive frames, warn if >5000 rows
            if len(filtered) > 5000:
                st.warning("Applying the model to more than 5,000 rows; this may take a minute.")
            def _row_prob(r):
                try:
                    return predictor.predict_prob(r["winner_name"], r["loser_name"], r["surface"])
                except Exception:
                    return np.nan
            filtered = filtered.copy()
            filtered["model_prob_w"] = filtered.apply(_row_prob, axis=1)
            filtered["edge_w"] = filtered["model_prob_w"] - filtered["mkt_prob_w"]

            # time-series chart by month
            if filtered["edge_w"].notna().any():
                ts = (
                    filtered.set_index("tourney_date")["edge_w"]
                    .resample("M").mean()
                )
                st.markdown("#### Average model–market edge (winner) over time")
                st.line_chart(ts)


        # ── Display ───────────────────────────────────────────────────────────
        display_cols = [
            "tourney_date", "tourney_name", "surface", "tourney_level", "round",
            "winner_name", "loser_name",
            "elo_pre_w", "elo_pre_l",
            "winner_rank", "loser_rank", "rank_diff",
            "recent_win_rate_w", "recent_win_rate_l",
            "mkt_prob_w", "model_prob_w", "edge_w",
        ]
        display_cols = [c for c in display_cols if c in filtered.columns]
        styled_hist = filtered[display_cols].sort_values("tourney_date", ascending=False).head(500)
        # make headers human-readable
        styled_hist = styled_hist.rename(columns=lambda s: s.replace('_', ' ').title())
        st.dataframe(
            styled_hist,
            width="stretch",
            hide_index=True,
            height=get_dataframe_height(filtered),
            column_config={
                "Tourney Date":       st.column_config.DateColumn("Date"),
                "Elo Pre W":          st.column_config.NumberColumn("Elo Winner", format="%.0f"),
                "Elo Pre L":          st.column_config.NumberColumn("Elo Loser",  format="%.0f"),
                "Rank Diff":          st.column_config.NumberColumn("Rank diff",  format="%d"),
                "Recent Win Rate W":  st.column_config.NumberColumn("Form (W)",   format="%.2f"),
                "recent_win_rate_l":  st.column_config.NumberColumn("Form (L)",   format="%.2f"),
                "mkt_prob_w":         st.column_config.NumberColumn("Mkt prob",   format="%.3f"),
                "model_prob_w":       st.column_config.NumberColumn("Model prob", format="%.3f", help="Model probability that winner wins"),
                "edge_w":             st.column_config.NumberColumn("Edge",       format="%.3f", help="Model prob − market prob"),
            },
        )

        # footer for Match Explorer tab
        add_betting_oracle_footer()


# ══════════════════════════════════════════════════════════════════════════════
# Tab 5 — ELO Rankings
# ══════════════════════════════════════════════════════════════════════════════
with tab_elo:
    st.subheader("📊 ELO Rankings")

    # ── Live official ATP rankings (Bzzoiro) ──────────────────────────────────
    with st.expander("🌐 Live Official ATP Rankings", expanded=False):
        _live_top = st.slider(
            "Show top N", min_value=10, max_value=200, value=100, step=10,
            key="live_rank_top",
        )
        _live_df = load_atp_rankings_live(top=_live_top)
        if _live_df.empty:
            st.info("Live rankings unavailable — check BZZOIRO_KEY or API connectivity.")
        else:
            st.caption(
                f"Official ATP rankings · {len(_live_df):,} players · refreshes every 5 minutes"
            )
            st.dataframe(
                _live_df,
                width="stretch",
                hide_index=True,
                height=get_dataframe_height(_live_df),
                column_config={
                    "Rank":   st.column_config.NumberColumn("Rank",   format="%d"),
                    "Points": st.column_config.NumberColumn("Points", format="%d"),
                },
            )

    with st.expander("What is ELO and how should I interpret it?"):
        st.markdown(textwrap.dedent(
            """
            **ELO** is a rating system originally developed for chess and widely
            adopted across sports to quantify skill.  It is named after its
            creator, Arpad Elo, a Hungarian-American physics professor.  Every
            player starts at 1500 and gains or loses points after each match
            based on the expected outcome.  Beating a stronger opponent yields a
            larger increase, while losing to a weaker player causes a bigger
            drop.

            - A higher ELO indicates a better performer; differences of ~200
              points correspond to roughly a 75% win probability.
            - Surface-specific ELOs (hard, clay, grass) are computed separately
              and reflect a player's strength on that court type.
            - Use ELO in conjunction with market odds or other features when
              evaluating matches; it represents the model's baseline prediction
              before accounting for betting lines or recent form.

            This app displays the *pre-match* ELO, i.e. the rating **before** the
            tie was played, so it is directly comparable to the odds you're seeing.
            """
        ))

    if features.empty:
        st.warning("Run `python features.py` to generate the feature matrix.")
    else:
        # Build current ELO: last known ELO for every player
        has_ioc = "winner_ioc" in features.columns
        w_cols = ["winner_name", "elo_pre_w", "tourney_date"] + (["winner_ioc"] if has_ioc else [])
        l_cols = ["loser_name",  "elo_pre_l", "tourney_date"] + (["loser_ioc"]  if has_ioc else [])
        rename_w = {"winner_name": "player", "elo_pre_w": "elo", "tourney_date": "date"}
        rename_l = {"loser_name":  "player", "elo_pre_l": "elo", "tourney_date": "date"}
        if has_ioc:
            rename_w["winner_ioc"] = "country"
            rename_l["loser_ioc"]  = "country"
        all_w = features[w_cols].rename(columns=rename_w)
        all_l = features[l_cols].rename(columns=rename_l)
        all_players_elo = pd.concat([all_w, all_l], ignore_index=True)
        latest = (
            all_players_elo.sort_values("date")
            .groupby("player", as_index=False)
            .last()
            .sort_values("elo", ascending=False)
            .reset_index(drop=True)
        )
        latest.index += 1
        latest["elo"] = latest["elo"].round(0).astype(int)

        # Top-N slider
        top_n = st.slider("Show top N players", min_value=10, max_value=100, value=30, step=5)

        overall_tab, surf_tab, surf_cmp_tab = st.tabs(
            ["Overall ELO", "Surface ELO", "Surface Comparison"]
        )

        with overall_tab:
            col_a, col_b = st.columns([2, 1])
            with col_a:
                show_cols = ["player"] + (["country"] if has_ioc else []) + ["elo", "date"]
                st.dataframe(
                    latest[show_cols].head(top_n),
                    width="stretch",
                    column_config={
                        "elo":  st.column_config.NumberColumn("ELO",        format="%d"),
                        "date": st.column_config.DateColumn("Last match"),
                    },
                )
            with col_b:
                top_slice = latest.head(min(top_n, 20))
                # Color-code bars by ELO quartile
                elo_vals  = top_slice["elo"].tolist()
                q75, q50  = (
                    pd.Series(elo_vals).quantile(0.75),
                    pd.Series(elo_vals).quantile(0.50),
                )
                colors = [
                    "#059669" if v >= q75 else
                    "#1f77b4" if v >= q50 else
                    "#9ca3af"
                    for v in elo_vals
                ]
                fig_bar = go.Figure(go.Bar(
                    x=elo_vals,
                    y=top_slice["player"].tolist(),
                    orientation="h",
                    marker_color=colors,
                    text=[str(v) for v in elo_vals],
                    textposition="outside",
                ))
                fig_bar.update_layout(
                    yaxis=dict(autorange="reversed"),
                    xaxis=dict(range=[min(elo_vals) - 50, max(elo_vals) + 80]),
                    margin=dict(l=0, r=50, t=10, b=10),
                    height=max(300, min(top_n, 20) * 28),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_bar, width="stretch", key="elo_overall_bar")

        with surf_tab:
            surf_sel_elo = st.selectbox(
                "Surface", ["Hard", "Clay", "Grass", "Carpet"], key="elo_surf_sel"
            )
            surf_col_w = "elo_surf_pre_w"
            surf_col_l = "elo_surf_pre_l"
            if surf_col_w in features.columns:
                sw = features[features["surface"].str.capitalize() == surf_sel_elo][
                    ["winner_name", surf_col_w, "tourney_date"]
                ].rename(columns={"winner_name": "player", surf_col_w: "elo"})
                sl = features[features["surface"].str.capitalize() == surf_sel_elo][
                    ["loser_name", surf_col_l, "tourney_date"]
                ].rename(columns={"loser_name": "player", surf_col_l: "elo"})
                surf_latest = (
                    pd.concat([sw, sl], ignore_index=True)
                    .sort_values("tourney_date")
                    .groupby("player", as_index=False).last()
                    .sort_values("elo", ascending=False)
                    .reset_index(drop=True)
                )
                surf_latest.index += 1
                surf_latest["elo"] = surf_latest["elo"].round(0).astype(int)
                st.dataframe(
                    surf_latest[["player", "elo"]].head(top_n),
                    width="stretch",
                    height=get_dataframe_height(surf_latest.head(top_n)),
                    column_config={"elo": st.column_config.NumberColumn("ELO", format="%d")},
                )
            else:
                st.info("Surface ELO columns not present in feature data.")

        with surf_cmp_tab:
            st.markdown("#### Surface ELO Comparison — Top 10")
            surf_col_w = "elo_surf_pre_w"
            surf_col_l = "elo_surf_pre_l"
            if surf_col_w in features.columns:
                top10_players = latest["player"].head(10).tolist()
                surf_rows = []
                for surf_name in ["Hard", "Clay", "Grass"]:
                    sw2 = features[features["surface"].str.capitalize() == surf_name][
                        ["winner_name", surf_col_w, "tourney_date"]
                    ].rename(columns={"winner_name": "player", surf_col_w: "elo"})
                    sl2 = features[features["surface"].str.capitalize() == surf_name][
                        ["loser_name", surf_col_l, "tourney_date"]
                    ].rename(columns={"loser_name": "player", surf_col_l: "elo"})
                    sdf = (
                        pd.concat([sw2, sl2], ignore_index=True)
                        .sort_values("tourney_date")
                        .groupby("player", as_index=False).last()
                    )
                    sdf["surface"] = surf_name
                    surf_rows.append(sdf)

                if surf_rows:
                    surf_cmp = pd.concat(surf_rows, ignore_index=True)
                    surf_cmp = surf_cmp[surf_cmp["player"].isin(top10_players)]
                    surf_cmp["elo"] = surf_cmp["elo"].round(0)

                    colors_map = {"Hard": "#1f77b4", "Clay": "#e07a3a", "Grass": "#2ca02c"}
                    fig_cmp = go.Figure()
                    for surf_name, grp in surf_cmp.groupby("surface"):
                        grp_sorted = grp.set_index("player").reindex(top10_players).dropna()
                        fig_cmp.add_trace(go.Bar(
                            name=surf_name,
                            x=grp_sorted.index.tolist(),
                            y=grp_sorted["elo"].tolist(),
                            marker_color=colors_map.get(surf_name, "#aaa"),
                        ))
                    fig_cmp.update_layout(
                        barmode="group",
                        xaxis_title=None,
                        yaxis_title="ELO",
                        legend=dict(orientation="h", y=1.05),
                        margin=dict(l=0, r=0, t=30, b=10),
                        height=380,
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig_cmp, width="stretch", key="elo_surf_cmp")
            else:
                st.info("Surface ELO columns not present in feature data.")

    # footer for ELO Rankings tab
    add_betting_oracle_footer()


# ══════════════════════════════════════════════════════════════════════════════
# Tab 6 — Model Stats
# ══════════════════════════════════════════════════════════════════════════════
with tab_model:
    st.subheader("Prediction Model Statistics")

    if model_data is None:
        st.warning(
            "No trained model found. Run `python train.py` to train the model first."
        )
    else:
        md = model_data   # shorthand

        # ── Metric cards ──────────────────────────────────────────────────────
        elo_base = md.get("elo_baseline", {})
        st.markdown(f"**Best model: {md['model_name']}** — test set: {md.get('test_year', '2025')}+  "
                    f"({md.get('test_rows', '?'):,} matches)")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            delta_acc = md["accuracy"] - elo_base.get("accuracy", 0)
            st.metric("Accuracy", f"{md['accuracy']:.1%}",
                      delta=f"{delta_acc:+.1%} vs ELO baseline",
                      help="% of match winners correctly predicted (test set)")
        with c2:
            st.metric("AUC-ROC", f"{md['auc']:.3f}",
                      delta=f"{md['auc'] - elo_base.get('auc', 0):+.3f} vs ELO baseline",
                      help="Area Under ROC Curve — 0.5 = random, 1.0 = perfect")
        with c3:
            delta_brier = md["brier"] - elo_base.get("brier", 0)
            st.metric("Brier Score", f"{md['brier']:.4f}",
                      delta=f"{delta_brier:+.4f} vs ELO baseline",
                      delta_color="inverse",   # lower is better
                      help="Probability calibration: lower = better (0.25 = random)")
        with c4:
            delta_ll = md["log_loss"] - elo_base.get("log_loss", 0)
            st.metric("Log Loss", f"{md['log_loss']:.4f}",
                      delta=f"{delta_ll:+.4f} vs ELO baseline",
                      delta_color="inverse",
                      help="Log loss — lower = better; penalises confident wrong predictions")

        st.markdown("---")

        # ── Feature importance + Calibration (side by side) ───────────────────
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("#### Feature Importances")
            fi = md.get("feature_importances")
            if fi:
                fi_df = pd.DataFrame(fi).sort_values("importance")
                LABEL_MAP = {
                    "elo_diff":         "Overall ELO diff",
                    "surface_elo_diff": "Surface ELO diff",
                    "total_elo_diff":   "Total ELO diff",
                    "h2h_diff":         "Head-to-head diff",
                    "form_diff":        "Recent form diff",
                    "rank_diff":        "ATP rank diff",
                    "height_diff":      "Height diff (cm)",
                    "age_diff":         "Age diff (yrs)",
                }
                fi_df["label"] = fi_df["feature"].map(lambda x: LABEL_MAP.get(x, x))
                fig_fi = go.Figure(go.Bar(
                    x=fi_df["importance"],
                    y=fi_df["label"],
                    orientation="h",
                    marker_color="#1f77b4",
                    text=[f"{v:.3f}" for v in fi_df["importance"]],
                    textposition="outside",
                ))
                fig_fi.update_layout(
                    xaxis_title="Importance",
                    yaxis_title=None,
                    margin=dict(l=0, r=40, t=10, b=30),
                    height=340,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_fi, width="stretch", key="model_feature_imp")
                with st.expander("What is feature importance?"):
                    st.markdown(
                        """
                        Feature importance for Random Forest measures how much each
                        feature reduces impurity (Gini) across all decision trees.
                        Higher values mean the feature is more influential in
                        predicting match outcomes.  ELO-based features dominate
                        because they encode cumulative historical performance; rank
                        adds complementary short-term signal.
                        """
                    )
            else:
                st.info("Feature importances not available for this model type.")

        with col_right:
            st.markdown("#### Calibration Curve")
            cal = md.get("calibration_data")
            if cal:
                fig_cal = go.Figure()
                # Perfect calibration line
                fig_cal.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode="lines",
                    line=dict(color="grey", dash="dash", width=1),
                    name="Perfect calibration",
                    showlegend=True,
                ))
                # Model calibration
                fig_cal.add_trace(go.Scatter(
                    x=cal["mean_predicted_value"],
                    y=cal["fraction_of_positives"],
                    mode="lines+markers",
                    name=md["model_name"],
                    line=dict(color="#1f77b4", width=2),
                    marker=dict(size=7),
                ))
                fig_cal.update_layout(
                    xaxis_title="Mean predicted probability",
                    yaxis_title="Fraction of positives",
                    xaxis=dict(range=[0, 1]),
                    yaxis=dict(range=[0, 1]),
                    legend=dict(x=0.02, y=0.98),
                    margin=dict(l=0, r=10, t=10, b=40),
                    height=340,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_cal, width="stretch", key="model_calibration")
                with st.expander("How to read the calibration curve"):
                    st.markdown(
                        """
                        A perfectly calibrated model follows the dashed diagonal:
                        when it says "70% chance of winning", the player wins
                        exactly 70% of the time. Points **above** the diagonal mean
                        the model is **under-confident** (actual win rate is higher
                        than predicted); points **below** mean over-confident.
                        Good calibration matters most when using Model% to size
                        bets or compare against market odds.
                        """
                    )
            else:
                st.info("Calibration data not stored — re-run `python train.py`.")

        st.markdown("---")

        # ── ROC Curve ────────────────────────────────────────────────────────
        st.markdown("#### ROC Curve")
        roc = md.get("roc_data")
        if roc:
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode="lines",
                line=dict(color="grey", dash="dash", width=1),
                name="Random (AUC=0.50)",
                showlegend=True,
            ))
            fig_roc.add_trace(go.Scatter(
                x=roc["fpr"], y=roc["tpr"],
                mode="lines",
                fill="tozeroy",
                fillcolor="rgba(31,119,180,0.15)",
                line=dict(color="#1f77b4", width=2),
                name=f"{md['model_name']} (AUC={roc['auc']:.3f})",
            ))
            fig_roc.update_layout(
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                xaxis=dict(range=[0, 1]),
                yaxis=dict(range=[0, 1.02]),
                legend=dict(x=0.55, y=0.1),
                margin=dict(l=0, r=10, t=10, b=40),
                height=380,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_roc, width="stretch", key="model_roc")

        st.markdown("---")

        # ── All-model comparison table ────────────────────────────────────────
        st.markdown("#### All Models vs ELO Baseline")
        all_m = md.get("all_metrics", {})
        if all_m:
            # Prepend ELO baseline row
            if elo_base:
                all_m = {"ELO Baseline": elo_base, **all_m}
            cmp_df = pd.DataFrame(
                [
                    {
                        "Model":    name,
                        "Accuracy": f"{v['accuracy']:.2%}",
                        "AUC-ROC":  f"{v['auc']:.3f}",
                        "Brier ↓":  f"{v['brier']:.4f}",
                        "Log Loss ↓": f"{v['log_loss']:.4f}",
                    }
                    for name, v in all_m.items()
                ]
            )
            best_model_name = md["model_name"]
            st.dataframe(
                cmp_df,
                width="stretch",
                hide_index=True,
                height=get_dataframe_height(cmp_df),
            )

        # ── Training metadata ─────────────────────────────────────────────────
        st.markdown("---")
        meta_c1, meta_c2, meta_c3, meta_c4 = st.columns(4)
        with meta_c1:
            st.caption(f"**Trained:** {md.get('train_date', 'unknown')[:10]}")
        with meta_c2:
            st.caption(f"**Train rows:** {md.get('train_rows', '?'):,}")
        with meta_c3:
            st.caption(f"**Test rows:** {md.get('test_rows', '?'):,}")
        with meta_c4:
            st.caption(f"**Test year:** {md.get('test_year', '?')}+")

    # footer for Model Stats tab
    add_betting_oracle_footer()


# ══════════════════════════════════════════════════════════════════════════════
# Tab 7 — Player Analysis
# ══════════════════════════════════════════════════════════════════════════════
with tab_player:
    st.subheader("👤 Player Analysis")

    if features.empty:
        st.warning("Feature matrix not found. Run `python features.py` first.")
    else:
        all_players_pa = sorted(
            set(features["winner_name"].tolist() + features["loser_name"].tolist())
        )

        selected_player = st.selectbox(
            "Select a player", all_players_pa, key="pa_player"
        )

        if selected_player:
            mask_w = features["winner_name"] == selected_player
            mask_l = features["loser_name"]  == selected_player
            player_matches = features[mask_w | mask_l].sort_values("tourney_date")

            if player_matches.empty:
                st.info(f"No match history found for {selected_player}.")
            else:
                # ── Overall ELO & Recent form ──────────────────────────
                last_row = player_matches.iloc[-1]
                if last_row["winner_name"] == selected_player:
                    overall_elo = last_row["elo_pre_w"]
                else:
                    overall_elo = last_row["elo_pre_l"]

                # Recent form: last 15 matches
                FORM_N = 15
                recent_pm = player_matches.tail(FORM_N)
                wins = (recent_pm["winner_name"] == selected_player).sum()
                form_pct = wins / len(recent_pm)

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Overall ELO", f"{overall_elo:.0f}")
                with c2:
                    st.metric(f"Recent Form (last {FORM_N})", f"{wins}/{len(recent_pm)}",
                              delta=f"{form_pct:.0%}")
                with c3:
                    st.metric("Total Matches (since 2020)", f"{len(player_matches):,}")

                # ── Surface ELO bar chart ──────────────────────────────
                has_surf_elo = (
                    "elo_surf_pre_w" in features.columns
                    and "elo_surf_pre_l" in features.columns
                )
                if has_surf_elo:
                    surf_elos: dict[str, float] = {}
                    for surf in ["Hard", "Clay", "Grass"]:
                        sub = player_matches[
                            player_matches["surface"].str.capitalize() == surf
                        ]
                        if sub.empty:
                            continue
                        last_s = sub.iloc[-1]
                        if last_s["winner_name"] == selected_player:
                            surf_elos[surf] = float(last_s["elo_surf_pre_w"])
                        else:
                            surf_elos[surf] = float(last_s["elo_surf_pre_l"])

                    if surf_elos:
                        st.markdown("#### ELO by Surface")
                        surf_colors = {"Hard": "#1f77b4", "Clay": "#e07a3a", "Grass": "#2ca02c"}
                        fig_surf = go.Figure(go.Bar(
                            x=list(surf_elos.keys()),
                            y=list(surf_elos.values()),
                            marker_color=[surf_colors.get(s, "#aaa") for s in surf_elos],
                            text=[f"{v:.0f}" for v in surf_elos.values()],
                            textposition="outside",
                            width=0.4,
                        ))
                        fig_surf.update_layout(
                            yaxis=dict(
                                range=[
                                    min(surf_elos.values()) - 80,
                                    max(surf_elos.values()) + 60,
                                ]
                            ),
                            margin=dict(l=0, r=0, t=10, b=10),
                            height=260,
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                        )
                        st.plotly_chart(fig_surf, width="stretch", key="pa_surf_elo")

                st.markdown("---")

                # ── H2H prediction tool ────────────────────────────────
                st.markdown("#### Head-to-Head Prediction")
                h2h_col1, h2h_col2 = st.columns([4, 2])
                with h2h_col1:
                    opponent_options = [
                        p for p in all_players_pa if p != selected_player
                    ]
                    opponent = st.selectbox(
                        "Opponent", opponent_options, key="pa_opponent"
                    )
                with h2h_col2:
                    h2h_surf = st.selectbox(
                        "Surface", ["Hard", "Clay", "Grass"], key="pa_surf"
                    )

                if predictor is not None and opponent:
                    try:
                        _w, sel_prob = predictor.predict(
                            selected_player, opponent, h2h_surf
                        )
                        opp_prob   = 1 - sel_prob
                        confidence = max(sel_prob, opp_prob)
                        winner_h2h = (
                            selected_player if sel_prob >= opp_prob else opponent
                        )

                        # Confidence meter widget
                        render_confidence_meter(confidence)

                        # Winner card
                        st.markdown(
                            f'<div class="winner-highlight" style="margin:1rem 0">'
                            f'<h3 style="margin:0">🏆 Predicted Winner: {winner_h2h}</h3>'
                            f'<p style="margin:0.5rem 0 0 0;opacity:0.9">{confidence:.1%} probability · {h2h_surf}</p>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                        # Probability bar chart
                        conf_color_h2h = (
                            "#11998e" if confidence >= 0.75 else
                            "#f7971e" if confidence >= 0.65 else
                            "#eb3349"
                        )
                        fig_h2h = go.Figure()
                        fig_h2h.add_trace(go.Bar(
                            y=["Win Probability"],
                            x=[sel_prob],
                            name=selected_player,
                            orientation="h",
                            marker_color=conf_color_h2h if sel_prob >= opp_prob else "#9ca3af",
                            text=[f"{sel_prob:.1%}"],
                            textposition="inside",
                            textfont=dict(color="white"),
                        ))
                        fig_h2h.add_trace(go.Bar(
                            y=["Win Probability"],
                            x=[opp_prob],
                            name=opponent,
                            orientation="h",
                            marker_color=conf_color_h2h if opp_prob > sel_prob else "#9ca3af",
                            text=[f"{opp_prob:.1%}"],
                            textposition="inside",
                            textfont=dict(color="white"),
                        ))
                        fig_h2h.update_layout(
                            barmode="stack",
                            xaxis=dict(range=[0, 1], tickformat=".0%", visible=False),
                            height=80,
                            margin=dict(l=0, r=0, t=20, b=0),
                            legend=dict(orientation="h", y=-1.0, x=0),
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                        )
                        st.plotly_chart(fig_h2h, width="stretch", key="pa_h2h_prob")

                    except Exception as e:
                        st.error(f"H2H prediction failed: {e}")
                elif predictor is None:
                    st.info("Train a model first (`python train.py`) to see H2H predictions.")

    add_betting_oracle_footer()
