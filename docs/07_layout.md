# Streamlit Layout & UI

## Overview

The app uses a **sidebar navigation** pattern with 7 pages. The layout is `"wide"` mode with custom CSS for match cards, confidence meters, and a bookie-slip widget.

---

## 1. Page Config & Navigation

```python
import streamlit as st

st.set_page_config(
    page_title="Tennis Predictions",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar
st.sidebar.image("data_files/logo.png", width=150)
st.sidebar.title("Navigation")

page = st.sidebar.radio("Go to", [
    "📡 Today's Matches",
    "📋 Prediction Backlog",
    "🔮 Match Prediction",
    "📊 ELO Rankings",
    "🏆 Tournament Simulator",
    "👤 Player Analysis",
    "ℹ️ About",
])
```

---

## 2. Page Descriptions

### 📡 Today's Matches

**Purpose:** Display all professional tennis matches happening today with live predictions.

| Component | Description |
|---|---|
| Summary metrics | Total matches, live, starting soon, upcoming |
| Status filter pills | Filter by live / soon / upcoming / scheduled |
| Bookie slip button | Auto-generate high-confidence picks card |
| Match cards | Grouped by tournament, show players, time, prediction, live score |
| Legend | Status dot meanings and confidence thresholds |

**Key features:**
- Matches grouped by tournament with styled headers
- Color-coded status dots (red=live, orange=soon, green=upcoming)
- Predictions inline on each match card
- High-confidence picks (≥75%) get green checkmark
- Medium confidence (65-75%) get orange text
- Low confidence (<65%) shows "skip"

### 📋 Prediction Backlog

**Purpose:** Track accuracy of today's completed match predictions vs actual results.

| Component | Description |
|---|---|
| Summary metrics | Completed, predicted, overall accuracy, high-conf accuracy |
| Accuracy-by-tier chart | Bar chart of accuracy at each confidence level |
| Match breakdown chart | Count of high/medium/low/no-prediction matches |
| Filter pills | All / Correct / Wrong / High Conf Only |
| Result cards | Each completed match with prediction result (✓ or ✗) |

### 🔮 Match Prediction

**Purpose:** Head-to-head predictor for any two players on any surface.

| Component | Description |
|---|---|
| Player selectors | Two searchable dropdowns |
| Surface picker | Hard / Clay / Grass |
| Confidence meter | Colored bar with percentage |
| Winner card | Highlighted prediction box |
| Player comparison | Side-by-side ELO ratings |
| Probability chart | Horizontal bar chart |

### 📊 ELO Rankings

**Purpose:** Browsable leaderboard of ELO ratings.

| Component | Description |
|---|---|
| Top-N slider | 10–100 players |
| Surface filter | Overall / Hard / Clay / Grass |
| Data table | Rank, player, rating, surface breakdown |
| Top-20 bar chart | Color-coded by rating |
| Surface comparison | Grouped bar chart for top 10 |

### 🏆 Tournament Simulator

**Purpose:** Simulate a single-elimination bracket.

| Component | Description |
|---|---|
| Tournament name | Text input |
| Surface & draw size | Dropdowns (8/16/32) |
| Player selectors | One per seed, pre-filled with top ELO players |
| Round-by-round results | Styled cards for each match |
| Champion announcement | Highlighted winner box |

### 👤 Player Analysis

**Purpose:** Deep dive on a single player's ratings and matchups.

| Component | Description |
|---|---|
| Player selector | Searchable dropdown |
| Rating cards | Overall ELO, recent form |
| Surface bar chart | ELO by surface |
| H2H tool | Pick opponent + surface, see win probability |
| Probability chart | Horizontal bar for selected matchup |

### ℹ️ About

**Purpose:** Explain methodology, confidence levels, data sources, disclaimer.

---

## 3. Custom CSS

The app uses extensive custom CSS injected via `st.markdown(unsafe_allow_html=True)`. Key classes:

```css
/* Main page header */
.main-header {
    font-size: 3rem;
    font-weight: bold;
    color: #1e88e5;
    text-align: center;
    margin-bottom: 2rem;
}

/* Match card (used on Today's Matches and Backlog) */
.match-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 0.5rem;
}
.match-card:hover {
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}

/* Prediction winner highlight */
.winner-highlight {
    background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
}

/* Confidence levels in match cards */
.prediction-high   { color: #059669; }   /* green  */
.prediction-medium { color: #d97706; }   /* amber  */
.prediction-low    { color: #9ca3af; }   /* gray   */

/* Live status dots */
.status-dot { display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; }
.status-live     { background: #ef4444; animation: pulse-red 1s infinite; }
.status-starting { background: #f59e0b; animation: pulse-orange 1.5s infinite; }
.status-upcoming { background: #22c55e; }

/* Tournament group header */
.tournament-group-header {
    font-size: 1.2rem;
    font-weight: 700;
    color: #1e40af;
    margin: 1.5rem 0 0.75rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #dbeafe;
}

/* Bookie slip */
.slip-card {
    background: #1a1a2e;
    border-radius: 16px;
    padding: 2rem;
    color: #e0e0e0;
    font-family: 'Courier New', monospace;
    max-width: 520px;
    margin: 1rem auto;
}
.slip-header h2 { color: #38ef7d; letter-spacing: 2px; }
.slip-winner    { color: #38ef7d; font-weight: 700; }
.slip-confidence{ color: #f59e0b; }

/* Confidence meter bar */
.confidence-meter {
    height: 30px;
    border-radius: 15px;
    background: #e0e0e0;
    overflow: hidden;
    margin: 1rem 0;
}
.confidence-fill {
    height: 100%;
    border-radius: 15px;
    transition: width 0.5s ease;
}
```

---

## 4. Confidence Meter Widget

```python
def render_confidence_meter(confidence: float):
    if confidence >= 0.75:
        color = "#11998e"
        label = "HIGH"
    elif confidence >= 0.65:
        color = "#f7971e"
        label = "MEDIUM"
    else:
        color = "#eb3349"
        label = "LOW"

    st.markdown(f"""
    <div class="confidence-meter">
        <div class="confidence-fill"
             style="width: {confidence*100}%; background: {color}"></div>
    </div>
    <p style="text-align: center; margin: 0">
        <strong>Confidence: {confidence:.1%}</strong> ({label})
    </p>
    """, unsafe_allow_html=True)
```

---

## 5. Caching Strategy

| Decorator | Use Case | TTL |
|---|---|---|
| `@st.cache_resource` | Model + ELO system loading | Forever (app lifetime) |
| `@st.cache_data(ttl=120)` | Today's matches, completed matches | 2 min |
| Internal dict cache | Odds API responses | 5 min |

```python
@st.cache_resource
def load_model_and_data():
    """Load once per app lifetime."""
    ...

@st.cache_data(ttl=120)
def cached_todays_matches():
    return get_todays_matches()
```

---

## 6. Streamlit Cloud Config

### `.streamlit/config.toml`

```toml
[theme]
primaryColor = "#1e88e5"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"
```

### Entry Point

Streamlit Cloud runs the file specified in the deployment settings. Set this to `predictions.py` — your main page.

---

## 7. Wireframe

```
┌─────────────────────────────────────────────────────┐
│  [Logo]  Navigation (sidebar radio)                 │
│  📡 Today's Matches                                 │
│  📋 Prediction Backlog                              │
│  🔮 Match Prediction                                │
│  📊 ELO Rankings                                    │
│  🏆 Tournament Simulator                            │
│  👤 Player Analysis                                 │
│  ℹ️ About                                           │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌─ Summary Metrics ──────────────────────────────┐ │
│  │  Total: 47  │  Live: 3  │  Soon: 5  │  Up: 39 │ │
│  └────────────────────────────────────────────────┘ │
│                                                     │
│  [🔴 Live] [🟠 Soon] [🟢 Upcoming]  [🎫 Slip]     │
│                                                     │
│  🏆 Buenos Aires (ATP - SINGLES)                    │
│  ┌──────────────────────────────────────────┐       │
│  │ 🔴 Sinner J. vs Alcaraz C.  [6-4 3-2]  │       │
│  │                    Sinner 72% ✅          │       │
│  └──────────────────────────────────────────┘       │
│  ┌──────────────────────────────────────────┐       │
│  │ 🟢 Fritz T. vs Zverev A.    14:30       │       │
│  │                    Fritz 68%             │       │
│  └──────────────────────────────────────────┘       │
│                                                     │
└─────────────────────────────────────────────────────┘
```
