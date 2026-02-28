import os
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
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

# ── Load today's matches from Matchstat API ───────────────────────────────────
@st.cache_data(ttl=1800)   # re-check every 30 min but don't burn extra API calls
def load_today_matches() -> list[dict]:
    try:
        from matchstat_api import get_today_odds, has_upcoming_matches, calls_remaining
        if not has_upcoming_matches():
            return []
        return get_today_odds()
    except Exception as e:
        st.warning(f"Could not load live odds: {e}")
        return []

features      = load_features()
today_matches = load_today_matches()
predictor     = load_predictor()
model_data    = load_model_data()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_today, tab_data, tab_elo, tab_model = st.tabs(
    ["Today's Matches", "Match Explorer", "ELO Rankings", "Model Stats"]
)


# ══════════════════════════════════════════════════════════════════════════════
# Tab 1 — Today's Matches
# ══════════════════════════════════════════════════════════════════════════════
with tab_today:
    st.subheader(f"Upcoming matches — {date.today().strftime('%A, %B %d %Y')}")

    if not today_matches:
        st.info("No upcoming singles matches found for today, or API unavailable.")
    else:
        # Pull current ELO for each player from the feature matrix
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
            # common canonical forms
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

        rows = []
        for m in today_matches:
            p1, p2 = m["player1_name"], m["player2_name"]
            o1, o2 = m["odds_p1"], m["odds_p2"]
            elo1 = latest_elo(p1)
            elo2 = latest_elo(p2)

            # Market implied probability (devigged) — stored as 0-100 float
            if o1 and o2 and o1 > 1 and o2 > 1:
                raw1, raw2 = 1 / o1, 1 / o2
                total = raw1 + raw2
                mkt1_val = raw1 / total * 100
                mkt2_val = raw2 / total * 100
            else:
                mkt1_val = mkt2_val = None

            # Model win probability — only if at least one player has ATP tour history.
            # Without real ELO both players default to 1500 → model output is noise.
            surf = norm_surface(m.get("surface"))
            p1_known = has_match_data(p1)
            p2_known = has_match_data(p2)
            model_reliable = predictor is not None and surf != "—" and (p1_known or p2_known)

            if model_reliable:
                try:
                    _w, p1_prob = predictor.predict(p1, p2, surf)
                    mod1_val = p1_prob * 100
                    mod2_val = (1 - p1_prob) * 100
                except Exception:
                    mod1_val = mod2_val = None
            else:
                mod1_val = mod2_val = None

            # Betting edge = model% − market% (positive → model likes this player more than market)
            # Only shown when model output is considered reliable (≥1 known player)
            edge1 = (mod1_val - mkt1_val) if (mod1_val is not None and mkt1_val is not None) else None
            edge2 = (mod2_val - mkt2_val) if (mod2_val is not None and mkt2_val is not None) else None

            rows.append({
                "Tournament":       m["tournament"],
                "Round":            m.get("round") or "—",
                "Surface":          surf,
                "Player 1":         p1,
                "ELO (P1)":         elo1,
                "Odds (P1)":        o1,
                "Mkt% (P1)":        mkt1_val,
                "Model% (P1)":      mod1_val,
                "Edge (P1)":        edge1,
                "Player 2":         p2,
                "ELO (P2)":         elo2,
                "Odds (P2)":        o2,
                "Mkt% (P2)":        mkt2_val,
                "Model% (P2)":      mod2_val,
                "Edge (P2)":        edge2,
                "Total games line": m.get("total_games"),
            })

        display_df = pd.DataFrame(rows)

        # Formatting helpers for Styler (cannot mix Styler + column_config)
        def _fmt_pct(v):
            return f"{v:.1f}%" if pd.notna(v) else "—"

        def _fmt_edge(v):
            return f"{v:+.1f}pp" if pd.notna(v) else "—"

        def _fmt_odds(v):
            return f"{v:.3f}" if pd.notna(v) else "—"

        def _fmt_elo(v):
            return f"{v:.0f}" if pd.notna(v) else "—"

        def _highlight_edge(col):
            """Green cell for positive edge (model favours this player more than market)."""
            return [
                "background-color: #16a34a; color: #f0fdf4; font-weight: 600"
                if pd.notna(v) and float(v) > 0 else ""
                for v in col
            ]

        styled = (
            display_df.style
            .apply(_highlight_edge, subset=["Edge (P1)", "Edge (P2)"])
            .format({
                "Odds (P1)":   _fmt_odds,
                "Odds (P2)":   _fmt_odds,
                "ELO (P1)":    _fmt_elo,
                "ELO (P2)":    _fmt_elo,
                "Mkt% (P1)":   _fmt_pct,
                "Mkt% (P2)":   _fmt_pct,
                "Model% (P1)": _fmt_pct,
                "Model% (P2)": _fmt_pct,
                "Edge (P1)":   _fmt_edge,
                "Edge (P2)":   _fmt_edge,
            }, na_rep="—")
        )

        st.dataframe(
            styled,
            width="stretch",
            hide_index=True,
            height=get_dataframe_height(display_df),
        )
        st.caption(
            "Model% and Edge are blank when neither player has ATP main-tour match history "
            "in the training data (e.g. ITF / futures events). Edge = Model% − Market%: "
            "green cells indicate the model assigns higher probability than the market."
        )

        # explanation expander for odd columns: same text as bottom-of-page
        with st.expander("What do the 'Odds' columns mean?"):
            st.markdown(textwrap.dedent(
                """
                The **Odds (P1)** and **Odds (P2)** columns display decimal
                money-line prices from the bookmaker for each player.  A price of
                1.83 means a successful $1 stake returns $1.83.  Lower odds indicate
                the market favourite.  These figures are pulled from the Matchstat
                API and reflect the line *before* the match has started.  They are
                not the implied winning probability (see the "Mkt%" columns for
                that).  Use the odds together with ELO or other features when
                evaluating value.
                """
            ))

        try:
            from matchstat_api import calls_used_this_month, calls_remaining
            st.caption(
                f"API calls this month: {calls_used_this_month()} / 500 "
                f"({calls_remaining()} remaining)"
            )
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# Tab 2 — Match Explorer
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


# ══════════════════════════════════════════════════════════════════════════════
# Tab 3 — ELO Rankings
# ══════════════════════════════════════════════════════════════════════════════
with tab_elo:
    st.subheader("Current ELO Rankings (from feature data)")
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
        # winner_name / elo_pre_w is ELO *entering* the match, so take the post-match
        # update by finding each player's most recent appearance
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
        all_players = pd.concat([all_w, all_l], ignore_index=True)
        latest = (
            all_players.sort_values("date")
            .groupby("player", as_index=False)
            .last()
            .sort_values("elo", ascending=False)
            .reset_index(drop=True)
        )
        latest.index += 1
        latest["elo"] = latest["elo"].round(0).astype(int)

        overall_tab, surf_tab = st.tabs(["Overall ELO", "Surface ELO"])

        with overall_tab:
            col_a, col_b = st.columns([2, 1])
            with col_a:
                show_cols = ["player"] + (["country"] if has_ioc else []) + ["elo", "date"]
                st.dataframe(
                    latest[show_cols].head(100),
                    width="stretch",
                    column_config={
                        "elo":  st.column_config.NumberColumn("ELO", format="%d"),
                        "date": st.column_config.DateColumn("Last match"),
                    },
                )
            with col_b:
                top20 = latest.head(20).set_index("player")["elo"]
                st.bar_chart(top20, horizontal=True, height=500)

        with surf_tab:
            surf_sel = st.selectbox("Surface", ["Hard", "Clay", "Grass", "Carpet"])
            surf_col_w = "elo_surf_pre_w"
            surf_col_l = "elo_surf_pre_l"
            if surf_col_w in features.columns:
                sw = features[features["surface"].str.capitalize() == surf_sel][
                    ["winner_name", surf_col_w, "tourney_date"]
                ].rename(columns={"winner_name": "player", surf_col_w: "elo"})
                sl = features[features["surface"].str.capitalize() == surf_sel][
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
                    surf_latest[["player", "elo"]].head(50),
                    width="stretch",
                    height=get_dataframe_height(surf_latest),
                    column_config={"elo": st.column_config.NumberColumn("ELO", format="%d")},
                )
            else:
                st.info("Surface ELO columns not present in feature data.")

    # bottom-of-page explanation for odds columns
    with st.expander("What do the 'Odds' columns mean?"):
        st.markdown(textwrap.dedent(
            """
            The **Odds (P1)** and **Odds (P2)** columns display decimal
            money-line prices from the bookmaker for each player.  A price of
            1.83 means a successful $1 stake returns $1.83.  Lower odds indicate
            the market favourite.  These figures are pulled from the Matchstat
            API and reflect the line *before* the match has started.  They are
            not the implied winning probability (see the "Mkt%" columns for
            that).  Use the odds together with ELO or other features when
            evaluating value.
            """
        ))


# ══════════════════════════════════════════════════════════════════════════════
# Tab 4 — Model Stats
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
                st.plotly_chart(fig_fi, width="stretch")
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
                st.plotly_chart(fig_cal, width="stretch")
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
            st.plotly_chart(fig_roc, width="stretch")

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

