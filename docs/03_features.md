# Feature Engineering

## Overview

Features are computed **chronologically** — for each match we use only information available *before* that match. This prevents data leakage and mirrors how the model will be used at prediction time.

---

## 1. ELO Rating System

ELO is the backbone feature. Every player starts at **1500**. After each match, ratings adjust based on the outcome and the expected probability.

### Formula

```
Expected = 1 / (1 + 10^((opponent_rating - player_rating) / 400))
New Rating = Old Rating + K × (Actual − Expected)
```

- **K-factor = 32** (controls volatility)
- **Actual** = 1 for winner, 0 for loser

### Implementation

```python
from collections import defaultdict
from typing import Dict, List, Tuple


class EloRatingSystem:
    """
    ELO rating system with overall and surface-specific ratings.
    """

    def __init__(self, k_factor: int = 32, initial_rating: int = 1500):
        self.k_factor = k_factor
        self.initial_rating = initial_rating

        # Overall ELO
        self.ratings: Dict[int, float] = {}

        # Surface-specific ELO (Clay, Grass, Hard, Carpet)
        self.surface_ratings: Dict[str, Dict[int, float]] = {
            "Clay": {},
            "Grass": {},
            "Hard": {},
            "Carpet": {},
        }

        # History for visualisation
        self.rating_history: Dict[int, List[Tuple[str, float]]] = {}
        self.surface_history = {s: {} for s in self.surface_ratings}

    # ------------------------------------------------------------------
    def expected_score(self, player: float, opponent: float) -> float:
        return 1.0 / (1.0 + 10 ** ((opponent - player) / 400))

    # ------------------------------------------------------------------
    def update_ratings(
        self, winner_id: int, loser_id: int, surface: str, date: str
    ) -> Tuple[float, float, float, float]:
        """
        Update ratings after a match.
        Returns (winner_elo_before, loser_elo_before,
                 winner_surface_elo_before, loser_surface_elo_before)
        """
        # --- Overall ---
        w_elo = self.ratings.get(winner_id, self.initial_rating)
        l_elo = self.ratings.get(loser_id, self.initial_rating)

        exp_w = self.expected_score(w_elo, l_elo)
        exp_l = self.expected_score(l_elo, w_elo)

        self.ratings[winner_id] = w_elo + self.k_factor * (1 - exp_w)
        self.ratings[loser_id] = l_elo + self.k_factor * (0 - exp_l)

        # --- Surface-specific ---
        surf = surface if surface in self.surface_ratings else "Hard"
        w_surf = self.surface_ratings[surf].get(winner_id, self.initial_rating)
        l_surf = self.surface_ratings[surf].get(loser_id, self.initial_rating)

        exp_ws = self.expected_score(w_surf, l_surf)
        exp_ls = self.expected_score(l_surf, w_surf)

        self.surface_ratings[surf][winner_id] = w_surf + self.k_factor * (1 - exp_ws)
        self.surface_ratings[surf][loser_id] = l_surf + self.k_factor * (0 - exp_ls)

        # Record history
        for pid, rating in [
            (winner_id, self.ratings[winner_id]),
            (loser_id, self.ratings[loser_id]),
        ]:
            self.rating_history.setdefault(pid, []).append((date, rating))

        return w_elo, l_elo, w_surf, l_surf

    # ------------------------------------------------------------------
    def get_rating(self, player_id: int) -> float:
        return self.ratings.get(player_id, self.initial_rating)

    def get_surface_rating(self, player_id: int, surface: str) -> float:
        surf = surface if surface in self.surface_ratings else "Hard"
        return self.surface_ratings[surf].get(player_id, self.initial_rating)

    def get_top_players(self, n: int = 20):
        return sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)[:n]
```

---

## 2. Full Feature List

| # | Feature | Description | Source |
|---|---------|-------------|--------|
| 1 | `elo_diff` | Winner ELO − Loser ELO (before match) | ELO system |
| 2 | `surface_elo_diff` | Surface-specific ELO difference | ELO system |
| 3 | `total_elo_diff` | `(overall + surface)` ELO difference | Composite |
| 4 | `h2h_diff` | Head-to-head wins difference | H2H tracker |
| 5 | `form_diff` | Recent form difference (win rate, last 50) | Form tracker |
| 6 | `rank_diff` | ATP ranking difference (inverted: lower = better) | Match data |
| 7 | `height_diff` | Height difference (cm) | Player data |
| 8 | `age_diff` | Age difference (years) | Match data |

### Feature Columns Used for Training

```python
FEATURE_COLS = [
    "elo_diff",
    "surface_elo_diff",
    "total_elo_diff",
    "h2h_diff",
    "form_diff",
    "rank_diff",
    "height_diff",
    "age_diff",
]
```

---

## 3. Feature Engineering Pipeline

```python
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class FeatureEngineer:
    """
    Calculate all prediction features chronologically.
    """

    def __init__(self):
        self.elo_system = EloRatingSystem(k_factor=32)
        self.h2h: Dict[Tuple[int, int], int] = {}      # (p1, p2) -> wins for p1
        self.recent_matches: Dict[int, List[int]] = {}  # player -> [1=win, 0=loss]
        self.recent_window = 50

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Iterate chronologically and compute features BEFORE each match.
        """
        print("\n--- Feature Engineering ---")
        features = []
        total = len(df)

        for idx, row in df.iterrows():
            if idx % 10_000 == 0:
                print(f"  {idx}/{total} ({idx / total * 100:.1f}%)")

            wid = row["winner_id"]
            lid = row["loser_id"]
            surface = row["surface"]
            date = str(row["tourney_date"])

            # ---- ELO (returns pre-match values, then updates internally) ----
            w_elo, l_elo, w_surf, l_surf = self.elo_system.update_ratings(
                wid, lid, surface, date
            )

            # ---- Head-to-head (BEFORE update) ----
            h2h_w = self.h2h.get((wid, lid), 0)
            h2h_l = self.h2h.get((lid, wid), 0)
            self.h2h[(wid, lid)] = h2h_w + 1

            # ---- Recent form (BEFORE update) ----
            w_recent = self.recent_matches.get(wid, [])[-self.recent_window :]
            l_recent = self.recent_matches.get(lid, [])[-self.recent_window :]
            w_form = sum(w_recent) / len(w_recent) if w_recent else 0.5
            l_form = sum(l_recent) / len(l_recent) if l_recent else 0.5
            self.recent_matches.setdefault(wid, []).append(1)
            self.recent_matches.setdefault(lid, []).append(0)

            # ---- Ranking (lower number = better) ----
            w_rank = row["winner_rank"] if pd.notna(row["winner_rank"]) else 500
            l_rank = row["loser_rank"] if pd.notna(row["loser_rank"]) else 500

            # ---- Physical ----
            w_ht = row["winner_ht"] if pd.notna(row["winner_ht"]) else 183
            l_ht = row["loser_ht"] if pd.notna(row["loser_ht"]) else 183
            w_age = row["winner_age"] if pd.notna(row["winner_age"]) else 25
            l_age = row["loser_age"] if pd.notna(row["loser_age"]) else 25

            features.append(
                {
                    "match_idx": idx,
                    "winner_id": wid,
                    "loser_id": lid,
                    "surface": surface,
                    "tourney_level": row["tourney_level"],
                    "best_of": row["best_of"],
                    "year": row["year"],
                    "tourney_name": row["tourney_name"],
                    # ELO
                    "winner_elo": w_elo,
                    "loser_elo": l_elo,
                    "elo_diff": w_elo - l_elo,
                    "winner_surface_elo": w_surf,
                    "loser_surface_elo": l_surf,
                    "surface_elo_diff": w_surf - l_surf,
                    "winner_total_elo": w_elo + w_surf,
                    "loser_total_elo": l_elo + l_surf,
                    "total_elo_diff": (w_elo + w_surf) - (l_elo + l_surf),
                    # H2H
                    "h2h_winner": h2h_w,
                    "h2h_loser": h2h_l,
                    "h2h_diff": h2h_w - h2h_l,
                    # Form
                    "winner_form": w_form,
                    "loser_form": l_form,
                    "form_diff": w_form - l_form,
                    # Ranking
                    "winner_rank": w_rank,
                    "loser_rank": l_rank,
                    "rank_diff": l_rank - w_rank,  # positive = winner ranked higher
                    # Physical
                    "winner_height": w_ht,
                    "loser_height": l_ht,
                    "height_diff": w_ht - l_ht,
                    "winner_age": w_age,
                    "loser_age": l_age,
                    "age_diff": w_age - l_age,
                }
            )

        feature_df = pd.DataFrame(features)
        print(f"\nFeature engineering complete. Shape: {feature_df.shape}")
        return feature_df

    def get_elo_system(self) -> EloRatingSystem:
        return self.elo_system
```

---

## 4. Balanced Dataset Construction

Since every row has the winner as Player 1, we randomly swap perspectives for 50% of rows so the model doesn't learn a positional bias.

```python
def prepare_training_data(
    feature_df: pd.DataFrame,
    feature_cols: list,
    test_year: int = 2025,
):
    """
    Split into train/test and create balanced labels.
    """
    train_df = feature_df[feature_df["year"] < test_year].copy()
    test_df  = feature_df[feature_df["year"] >= test_year].copy()

    def create_balanced(df):
        np.random.seed(42)
        X, y = [], []
        for _, row in df.iterrows():
            if np.random.random() < 0.5:
                X.append([row[c] for c in feature_cols])
                y.append(1)   # Player 1 (winner) wins
            else:
                X.append([-row[c] for c in feature_cols])  # negate diffs
                y.append(0)   # Player 1 (loser) loses
        return np.array(X), np.array(y)

    X_train, y_train = create_balanced(train_df)
    X_test,  y_test  = create_balanced(test_df)

    return X_train, y_train, X_test, y_test, train_df, test_df
```

---

## 5. Possible Extensions

| Feature | Notes |
|---------|-------|
| **Serve stats** | Ace rate, DF rate, 1st serve %, break-point save % (available in Sackmann data) |
| **Fatigue index** | Days since last match, matches in last 14 days |
| **Tournament-specific ELO** | Separate ratings per tournament (e.g., clay-court specialists at Roland Garros) |
| **Momentum** | Streak length (consecutive wins/losses) |
| **Fast ELO** | Higher K-factor (e.g., 64) for capturing recent form more aggressively |
| **Win-rate by round** | Performance in QF/SF/F specifically |
