# ATTR Analysis — What to Borrow, What to Skip

> Source: [alltimetennis.com](https://www.alltimetennis.com) / [Reddit post](https://www.reddit.com/r/sportsanalytics/comments/1tiyrmp/i_built_a_mens_tennis_stats_website_focused_on_a/)
> Reviewed: 2026-05-20

---

## What ATTR Does

ATTR (All-Time Tennis Rankings) assigns every ATP result since 1990 a single
career score point using three factors:

```
Career Score = Prestige Points × SPS × Set Multiplier
```

| Factor | Description |
|---|---|
| **Prestige Points** | Fixed weights by tier/round. GS title = 2,000. M1000 title = 1,000. Never repriced. |
| **SPS** | Seed Presence Score — logarithmic draw-quality score based on actual opponent ranks. Pre-2000 defaults to 0.80. |
| **Set Multiplier** | 1.00 for a 3-0 win, 0.94 for a 3-2 win (winner); inverse logic for loser score. |

Its **primary goal** is cross-era career comparison (Djokovic vs. Sampras).
That is fundamentally different from this project's goal of predicting the
outcome of tomorrow's match.

---

## Overall Verdict

**Do not incorporate the full ATTR system.** The design objectives conflict:

- ATTR accumulates career-long scores — useful for historical ranking, not
  forward-looking prediction.
- ATTR intentionally ignores week-to-week form, current fitness, and
  draw-path luck — exactly the signals this project needs to exploit.
- The 1990 start date and pre-2000 SPS default would introduce artificial
  gaps; this project already seeds ELO from 1968.

---

## What Is Worth Borrowing

### 1. Margin-of-Victory Feature (Set Multiplier concept)

**Current state:** `form_diff` is a binary win/loss ratio over the last 20
matches. A 7–6, 7–6 squeaker counts identically to a 6–0, 6–0 demolition.

**Idea:** Encode a rolling "dominance score" using the same set-score logic
ATTR uses for its Set Multiplier. For each match in the rolling window, assign
a dominance weight and average it.

ATTR's winner weights:

| Score | Weight |
|---|---|
| 3-0 / 2-0 (straight sets) | 1.00 |
| 3-1 / 2-1 | 0.97 |
| 3-2 | 0.94 |

This maps cleanly to the Sackmann data `score` column (e.g. `"6-3 7-5"`).
Sets won/lost can be parsed from the score string to compute these weights.

**New features this would add:**

| Feature | Description |
|---|---|
| `winner_dominance` | Mean set-score weight over last N wins for the winner |
| `loser_dominance` | Same for the loser |
| `dominance_diff` | `winner_dominance − loser_dominance` (model input) |

#### Prototype implementation

```python
import re
from collections import deque

SET_MULTIPLIER_W = {(2, 0): 1.00, (3, 0): 1.00,
                    (2, 1): 0.97, (3, 1): 0.97,
                    (3, 2): 0.94}

def _sets_from_score(score: str) -> tuple[int, int]:
    """
    Parse a score string like '6-3 7-5' or '6-3 4-6 7-5' and return
    (sets_won_by_winner, sets_won_by_loser).
    Returns (0, 0) if unparseable.
    """
    sets = re.findall(r"(\d+)-(\d+)", str(score))
    if not sets:
        return 0, 0
    w_sets = sum(1 for a, b in sets if int(a) > int(b))
    l_sets = sum(1 for a, b in sets if int(b) > int(a))
    return w_sets, l_sets


def _set_multiplier(w_sets: int, l_sets: int) -> float:
    """Return the ATTR-style dominance weight for the winner."""
    return SET_MULTIPLIER_W.get((w_sets, l_sets), 0.94)


def _compute_dominance(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Add rolling dominance columns to a chronologically-sorted match DataFrame.
    Columns added: dominance_pre_w, dominance_pre_l
    """
    # Deque of (weight,) per player over last `window` matches
    history: dict[str, deque] = {}

    dom_pre_w, dom_pre_l = [], []

    for row in df.itertuples(index=False):
        w, l = row.winner_id, row.loser_id
        w_sets, l_sets = _sets_from_score(row.score)
        weight = _set_multiplier(w_sets, l_sets)

        # Pre-match averages
        dq_w = history.setdefault(w, deque(maxlen=window))
        dq_l = history.setdefault(l, deque(maxlen=window))
        dom_pre_w.append(sum(dq_w) / len(dq_w) if dq_w else 0.97)
        dom_pre_l.append(sum(dq_l) / len(dq_l) if dq_l else 0.97)

        # Update with outcome
        dq_w.append(weight)        # winner earns the dominance weight
        dq_l.append(1.00 - weight) # loser earns the complement (resilience)

    df = df.copy()
    df["dominance_pre_w"] = dom_pre_w
    df["dominance_pre_l"] = dom_pre_l
    return df
```

To wire into `features.py`:
1. Call `_compute_dominance(df)` after `_compute_elo(df)`.
2. Add `"dominance_diff"` (`dominance_pre_w − dominance_pre_l`) to
   `FEATURE_COLS`.
3. Track the same deque in `FeatureEngineer.update()` and expose it in
   `get_prediction_features()`.

> **Effort:** ~1 hour. **Expected lift:** marginal but directionally correct —
> dominance encodes recovery/fatigue signals not captured by binary win/loss.

---

### 2. Opponent-Quality-Adjusted Recent Form (SPS concept)

**Current state:** `form_diff` treats all wins equally regardless of opponent
strength. ELO implicitly adjusts for opponent quality in the rating update, but
it's an accumulating stock, not a recent-window flow feature.

**Idea:** A rolling "quality of wins" metric: for each win in the last N
matches, weight it by the opponent's rank at match time. This is the core SPS
intuition — beating a top-10 player should count more than beating a qualifier.

ATTR's logarithmic formula (simplified for a rolling window):

```
opponent_value(rank) = max(0, 1 - log(rank) / log(max_rank))
```

**New features this would add:**

| Feature | Description |
|---|---|
| `win_quality_w` | Mean opponent-rank-weighted win score over last N matches for winner |
| `win_quality_l` | Same for loser |
| `win_quality_diff` | `win_quality_w − win_quality_l` (model input) |

#### Prototype implementation

```python
import math

MAX_RANK = 500  # ranks beyond this treated as floor
RANK_FLOOR = 0.0

def _opponent_value(rank: float | None) -> float:
    """
    ATTR-style logarithmic opponent quality score.
    rank=1 → 1.0, rank=500 → 0.0, None → 0.5 (unknown)
    """
    if rank is None or rank <= 0:
        return 0.5
    rank = min(float(rank), MAX_RANK)
    return max(RANK_FLOOR, 1.0 - math.log(rank) / math.log(MAX_RANK))


def _compute_win_quality(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Add rolling win-quality columns.  Requires loser_rank / winner_rank cols.
    Columns added: win_quality_pre_w, win_quality_pre_l
    """
    history: dict[str, deque] = {}

    wq_pre_w, wq_pre_l = [], []

    for row in df.itertuples(index=False):
        w, l = row.winner_id, row.loser_id
        w_rank = getattr(row, "winner_rank", None)
        l_rank = getattr(row, "loser_rank", None)

        dq_w = history.setdefault(w, deque(maxlen=window))
        dq_l = history.setdefault(l, deque(maxlen=window))

        # Pre-match averages
        wq_pre_w.append(sum(dq_w) / len(dq_w) if dq_w else 0.5)
        wq_pre_l.append(sum(dq_l) / len(dq_l) if dq_l else 0.5)

        # Winner beat an opponent of quality = _opponent_value(loser_rank)
        dq_w.append(_opponent_value(l_rank))
        # Loser lost to an opponent of quality = _opponent_value(winner_rank)
        # (still records that they faced a tough opponent)
        dq_l.append(_opponent_value(w_rank) * 0.5)  # half credit for a loss

    df = df.copy()
    df["win_quality_pre_w"] = wq_pre_w
    df["win_quality_pre_l"] = wq_pre_l
    return df
```

> **Note:** `winner_rank` and `loser_rank` are already present in the Sackmann
> data and survive the `_clean()` step, so no new data source is needed.

> **Effort:** ~1 hour. **Expected lift:** likely more meaningful than dominance
> because it addresses a genuine gap in `form_diff` — it can distinguish a
> player on a 10-match win streak vs. qualifiers from one on the same streak
> vs. top-20 opponents.

---

## What to Ignore

| ATTR concept | Why it doesn't apply here |
|---|---|
| Career cumulative score | We need per-match features, not career totals |
| Cross-era prestige weighting | `tourney_level` already covers this |
| 1990 start cutoff | ELO is seeded from 1968; no reason to drop earlier data |
| Pre-2000 SPS default of 0.80 | Not relevant — `winner_rank` / `loser_rank` exist in the data |
| Team events exclusion | Already excluded in TML data pipeline |

---

## Implementation Priority

| # | Feature | Effort | Expected Value |
|---|---|---|---|
| 1 | `win_quality_diff` (SPS-inspired) | ~1 hr | Medium-high |
| 2 | `dominance_diff` (Set Multiplier-inspired) | ~1 hr | Low-medium |

Both can be added to `features.py` as standalone `_compute_*` functions and
wired into `FEATURE_COLS` without touching anything else. Retrain after adding
to measure actual accuracy delta.
