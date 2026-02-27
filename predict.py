"""
predict.py
----------
MatchPredictor and TournamentSimulator for live inference.

The trained model is loaded from data_files/tennis_predictor.pkl.
Player ELO / form are resolved via name lookup in the features parquet.

Usage:
    from predict import load_model, MatchPredictor, TournamentSimulator
    
    data = load_model()
    predictor = MatchPredictor.from_model_dict(data, features_df)
    winner, prob = predictor.predict_by_name("Carlos Alcaraz", "Jannik Sinner", "Clay")
"""

import os
import pickle
from typing import Any

import numpy as np
import pandas as pd

MODEL_PATH = os.path.join("data_files", "tennis_predictor.pkl")


# ── Section 3 — load helper ───────────────────────────────────────────────────

def load_model(path: str = MODEL_PATH) -> dict:
    """Load the pickled model dict produced by train.py."""
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Section 4 — MatchPredictor ────────────────────────────────────────────────

class MatchPredictor:
    """
    Use a trained model + features parquet to predict match outcomes.

    ELO, form, and H2H are resolved by player *name* lookup in the parquet,
    so callers don't need to track internal player IDs.
    """

    def __init__(self, model: Any, feature_names: list[str], features_df: pd.DataFrame):
        self.model         = model
        self.feature_names = feature_names
        self._df           = features_df  # full features parquet

    # ------------------------------------------------------------------
    @classmethod
    def from_model_dict(cls, model_dict: dict, features_df: pd.DataFrame) -> "MatchPredictor":
        return cls(model_dict["model"], model_dict["feature_names"], features_df)

    # ------------------------------------------------------------------
    def _latest_elo(self, name: str) -> float:
        """Most recent overall pre-match ELO for a player (by name)."""
        df = self._df
        mask = (df["winner_name"] == name) | (df["loser_name"] == name)
        sub  = df[mask]
        if sub.empty:
            return 1500.0
        row = sub.iloc[-1]
        return float(row["elo_pre_w"] if row["winner_name"] == name else row["elo_pre_l"])

    def _latest_surface_elo(self, name: str, surface: str) -> float:
        """Most recent surface-specific pre-match ELO for a player."""
        df   = self._df
        surf = surface.capitalize()
        mask = (
            ((df["winner_name"] == name) | (df["loser_name"] == name))
            & (df["surface"].str.capitalize() == surf)
        )
        sub = df[mask]
        if sub.empty:
            return 1500.0
        row = sub.iloc[-1]
        return float(row["elo_surf_pre_w"] if row["winner_name"] == name else row["elo_surf_pre_l"])

    def _h2h(self, p1: str, p2: str) -> tuple[int, int]:
        """Historical H2H wins (p1 vs p2, p2 vs p1) from the parquet."""
        df   = self._df
        mask = (
            ((df["winner_name"] == p1) & (df["loser_name"] == p2))
            | ((df["winner_name"] == p2) & (df["loser_name"] == p1))
        )
        sub = df[mask]
        if sub.empty:
            return 0, 0
        p1_wins = int((sub["winner_name"] == p1).sum())
        p2_wins = int((sub["winner_name"] == p2).sum())
        return p1_wins, p2_wins

    def _recent_form(self, name: str, window: int = 20) -> float:
        """Win rate over the last `window` matches from the parquet."""
        df   = self._df
        mask = (df["winner_name"] == name) | (df["loser_name"] == name)
        sub  = df[mask].tail(window)
        if sub.empty:
            return 0.5
        wins = float((sub["winner_name"] == name).sum())
        return wins / len(sub)

    # ------------------------------------------------------------------
    def build_feature_vector(
        self,
        p1: str,
        p2: str,
        surface: str,
        rank1: int | None = None,
        rank2: int | None = None,
        height1: float | None = None,
        height2: float | None = None,
        age1: float | None = None,
        age2: float | None = None,
    ) -> np.ndarray:
        """
        Build the FEATURE_COLS vector for a future match.
        rank, height, age default to 0 (neutral difference) when unknown.
        """
        elo1   = self._latest_elo(p1)
        elo2   = self._latest_elo(p2)
        surf1  = self._latest_surface_elo(p1, surface)
        surf2  = self._latest_surface_elo(p2, surface)
        h2h1, h2h2 = self._h2h(p1, p2)
        f1     = self._recent_form(p1)
        f2     = self._recent_form(p2)

        r1 = rank1 or 500
        r2 = rank2 or 500
        ht1 = height1 or 183.0
        ht2 = height2 or 183.0
        a1  = age1 or 25.0
        a2  = age2 or 25.0

        vec = {
            "elo_diff":         elo1 - elo2,
            "surface_elo_diff": surf1 - surf2,
            "total_elo_diff":   (elo1 + surf1) - (elo2 + surf2),
            "h2h_diff":         h2h1 - h2h2,
            "form_diff":        f1 - f2,
            "rank_diff":        r2 - r1,        # positive = p1 ranked higher
            "height_diff":      ht1 - ht2,
            "age_diff":         a1 - a2,
        }
        return np.array([vec[f] for f in self.feature_names], dtype=float).reshape(1, -1)

    # ------------------------------------------------------------------
    def predict(
        self,
        p1: str,
        p2: str,
        surface: str,
        **kwargs,
    ) -> tuple[str, float]:
        """
        Predict winner and P1's win probability.
        Returns (predicted_winner_name, p1_win_probability).
        """
        X = self.build_feature_vector(p1, p2, surface, **kwargs)
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)[0]
            # label 1 = p1 wins (positive diff = p1 better)
            p1_prob = float(proba[1])
        else:
            pred = int(self.model.predict(X)[0])
            p1_prob = 1.0 if pred == 1 else 0.0

        winner = p1 if p1_prob >= 0.5 else p2
        return winner, p1_prob

    def predict_prob(self, p1: str, p2: str, surface: str, **kwargs) -> float:
        """Convenience: just return P1 win probability (0–1)."""
        _, p1_prob = self.predict(p1, p2, surface, **kwargs)
        return p1_prob


# ── Section 6 — TournamentSimulator ───────────────────────────────────────────

class TournamentSimulator:
    """Simulate a single-elimination bracket."""

    def __init__(self, predictor: MatchPredictor):
        self.predictor = predictor

    ROUND_NAMES = ["R1", "R2", "R3", "QF", "SF", "Final"]

    def simulate(
        self, draw: list[tuple[str, str]], surface: str
    ) -> tuple[list[dict], tuple[str, str]]:
        """
        Simulate a single-elimination bracket.

        Parameters
        ----------
        draw : list of (player_id_or_name, display_name) tuples, in seeded order.
               Length must be a power of 2.
        surface : court surface for all matches.

        Returns
        -------
        results  : list of dicts with keys round, match, winner, confidence
        champion : (id_or_name, display_name) tuple
        """
        current    = list(draw)
        results    = []
        round_num  = 0

        while len(current) > 1:
            rnd       = self.ROUND_NAMES[min(round_num, len(self.ROUND_NAMES) - 1)]
            next_round: list[tuple[str, str]] = []

            for i in range(0, len(current), 2):
                p1_id, p1_name = current[i]
                p2_id, p2_name = current[i + 1]
                winner_name, conf = self.predictor.predict(p1_name, p2_name, surface)
                winner_id   = p1_id if winner_name == p1_name else p2_id
                next_round.append((winner_id, winner_name))
                results.append({
                    "round":      rnd,
                    "match":      f"{p1_name} vs {p2_name}",
                    "winner":     winner_name,
                    "confidence": f"{max(conf, 1 - conf):.1%}",
                })

            current   = next_round
            round_num += 1

        champion = current[0]
        return results, champion

    def print_bracket(self, results: list[dict], champion: tuple[str, str]) -> None:
        """Pretty-print simulated bracket results."""
        print("\n" + "=" * 60)
        print("BRACKET SIMULATION RESULTS")
        print("=" * 60)
        current_round = None
        for r in results:
            if r["round"] != current_round:
                current_round = r["round"]
                print(f"\n  [{current_round}]")
            print(f"    {r['match'][:45]:45s}  → {r['winner']}  ({r['confidence']})")
        print(f"\n🏆 Champion: {champion[1]}")
        print("=" * 60)


# ── Quick sanity check when run directly ──────────────────────────────────────

if __name__ == "__main__":
    import sys

    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH} — run `python train.py` first.")
        sys.exit(1)

    parquet = os.path.join("data_files", "features_2020_present.parquet")
    print(f"Loading model from {MODEL_PATH} …")
    data = load_model()
    print(f"  Model: {data['model_name']}  accuracy: {data['accuracy']:.2%}")
    print(f"  Features: {data['feature_names']}")

    print(f"\nLoading features from {parquet} …")
    df = pd.read_parquet(parquet)

    predictor = MatchPredictor.from_model_dict(data, df)

    # Sample prediction — two top-10 players
    test_matches = [
        ("Jannik Sinner",    "Carlos Alcaraz", "Hard"),
        ("Novak Djokovic",   "Rafael Nadal",   "Clay"),
        ("Carlos Alcaraz",   "Jannik Sinner",  "Clay"),
    ]
    print("\nSample predictions:")
    print(f"  {'Match':45s}  {'P1 prob':>8s}  {'Predicted winner'}")
    print("  " + "-" * 75)
    for p1, p2, surf in test_matches:
        winner, prob = predictor.predict(p1, p2, surf)
        elo1 = predictor._latest_elo(p1)
        elo2 = predictor._latest_elo(p2)
        print(
            f"  {p1} vs {p2} ({surf}):{'':5s}"
            f"  {prob:>7.1%}    {winner}"
            f"   [ELO {elo1:.0f} vs {elo2:.0f}]"
        )
