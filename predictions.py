import os
from datetime import date

import pandas as pd
import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Tennis Predictions",
    page_icon="🎾",
    layout="wide",
)

DATA_DIR     = "data_files"
FEATURES_FILE = os.path.join(DATA_DIR, "features_2020_present.parquet")

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

features = load_features()
today_matches = load_today_matches()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_today, tab_data, tab_elo = st.tabs(["Today's Matches", "Match Explorer", "ELO Rankings"])


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

        rows = []
        for m in today_matches:
            p1, p2 = m["player1_name"], m["player2_name"]
            o1, o2 = m["odds_p1"], m["odds_p2"]
            elo1 = latest_elo(p1)
            elo2 = latest_elo(p2)

            # Market implied probability (devigged)
            if o1 and o2 and o1 > 1 and o2 > 1:
                raw1, raw2 = 1 / o1, 1 / o2
                total = raw1 + raw2
                mkt1 = f"{raw1 / total * 100:.1f}%"
                mkt2 = f"{raw2 / total * 100:.1f}%"
            else:
                mkt1 = mkt2 = "—"

            rows.append({
                "Tournament":       m["tournament"],
                "Surface":          m["surface"] or "—",
                "Round":            m["round"] or "—",
                "Player 1":         p1,
                "ELO (P1)":         elo1,
                "Odds (P1)":        o1,
                "Mkt% (P1)":        mkt1,
                "Player 2":         p2,
                "ELO (P2)":         elo2,
                "Odds (P2)":        o2,
                "Mkt% (P2)":        mkt2,
                "Total games line": m.get("total_games"),
            })

        display_df = pd.DataFrame(rows)
        st.dataframe(
            display_df,
            width="stretch",
            hide_index=True,
            height = get_dataframe_height(display_df),
            column_config={
                "Odds (P1)":  st.column_config.NumberColumn(format="%.3f"),
                "Odds (P2)":  st.column_config.NumberColumn(format="%.3f"),
                "ELO (P1)":   st.column_config.NumberColumn(format="%.0f"),
                "ELO (P2)":   st.column_config.NumberColumn(format="%.0f"),
            },
        )

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

        # ── Display ───────────────────────────────────────────────────────────
        display_cols = [
            "tourney_date", "tourney_name", "surface", "tourney_level", "round",
            "winner_name", "loser_name",
            "elo_pre_w", "elo_pre_l",
            "winner_rank", "loser_rank", "rank_diff",
            "recent_win_rate_w", "recent_win_rate_l",
            "mkt_prob_w",
        ]
        display_cols = [c for c in display_cols if c in filtered.columns]
        st.dataframe(
            filtered[display_cols].sort_values("tourney_date", ascending=False).head(500),
            width="stretch",
            hide_index=True,
            height=get_dataframe_height(filtered),
            column_config={
                "tourney_date":       st.column_config.DateColumn("Date"),
                "elo_pre_w":          st.column_config.NumberColumn("ELO winner", format="%.0f"),
                "elo_pre_l":          st.column_config.NumberColumn("ELO loser",  format="%.0f"),
                "rank_diff":          st.column_config.NumberColumn("Rank diff",  format="%d"),
                "recent_win_rate_w":  st.column_config.NumberColumn("Form (W)",   format="%.2f"),
                "recent_win_rate_l":  st.column_config.NumberColumn("Form (L)",   format="%.2f"),
                "mkt_prob_w":         st.column_config.NumberColumn("Mkt prob",   format="%.3f"),
            },
        )


# ══════════════════════════════════════════════════════════════════════════════
# Tab 3 — ELO Rankings
# ══════════════════════════════════════════════════════════════════════════════
with tab_elo:
    st.subheader("Current ELO Rankings (from feature data)")

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

