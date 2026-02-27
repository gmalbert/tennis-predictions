# Modeling

## Overview

We train several classifiers and compare them against an ELO-only baseline. The best model (typically XGBoost at ~70%) is serialized with the ELO state for serving in the Streamlit app.

### Accuracy Benchmarks (from reference repo)

| Model | Accuracy |
|---|---|
| ELO-only baseline | 64.0% |
| Decision Tree (sklearn) | 65.2% |
| Random Forest | 69.8% |
| **XGBoost** | **70.0%** |
| Neural Network | 69.1% |

---

## 1. ELO-Only Baseline

The simplest predictor: the player with the higher ELO wins.

```python
import numpy as np

# X_test[:, 0] is elo_diff
elo_pred = (X_test[:, 0] > 0).astype(int)
elo_acc = np.mean(elo_pred == y_test)
print(f"ELO-only baseline: {elo_acc:.2%}")
```

---

## 2. Scikit-Learn Models

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report


FEATURE_NAMES = [
    "elo_diff", "surface_elo_diff", "total_elo_diff",
    "h2h_diff", "form_diff", "rank_diff",
    "height_diff", "age_diff",
]


def train_all_models(X_train, y_train, X_test, y_test):
    results = {}

    # ---- Decision Tree ----
    dt = DecisionTreeClassifier(max_depth=10, min_samples_split=5, random_state=42)
    dt.fit(X_train, y_train)
    dt_acc = accuracy_score(y_test, dt.predict(X_test))
    results["Decision Tree"] = {"model": dt, "accuracy": dt_acc}
    print(f"Decision Tree: {dt_acc:.2%}")

    # ---- Random Forest ----
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test))
    results["Random Forest"] = {"model": rf, "accuracy": rf_acc}
    print(f"Random Forest: {rf_acc:.2%}")

    # Feature importance
    print("\nFeature Importance (RF):")
    for feat, imp in sorted(
        zip(FEATURE_NAMES, rf.feature_importances_), key=lambda x: -x[1]
    ):
        print(f"  {feat}: {imp:.4f}")

    # ---- XGBoost ----
    try:
        import xgboost as xgb

        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
        )
        xgb_model.fit(X_train, y_train)
        xgb_acc = accuracy_score(y_test, xgb_model.predict(X_test))
        results["XGBoost"] = {"model": xgb_model, "accuracy": xgb_acc}
        print(f"XGBoost: {xgb_acc:.2%}")
    except ImportError:
        print("XGBoost not installed — pip install xgboost")

    # ---- Neural Network ----
    nn = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=500,
        random_state=42,
        early_stopping=True,
    )
    nn.fit(X_train, y_train)
    nn_acc = accuracy_score(y_test, nn.predict(X_test))
    results["Neural Network"] = {"model": nn, "accuracy": nn_acc}
    print(f"Neural Network: {nn_acc:.2%}")

    # ---- Summary ----
    print("\n" + "=" * 50)
    print("MODEL COMPARISON")
    print("=" * 50)
    for name, data in results.items():
        print(f"  {name:20s}: {data['accuracy']:.2%}")

    return results
```

---

## 3. Model Serialization

Save the best model along with all state needed for prediction.

```python
import pickle
import os


def save_best_model(results, feature_engineer, feature_names, player_names, output_dir="."):
    best_name = max(results, key=lambda k: results[k]["accuracy"])
    best_model = results[best_name]["model"]

    save_data = {
        "model": best_model,
        "model_name": best_name,
        "feature_engineer": feature_engineer,
        "feature_names": feature_names,
        "player_names": player_names,
        "elo_system": feature_engineer.get_elo_system(),
    }

    path = os.path.join(output_dir, "tennis_predictor.pkl")
    with open(path, "wb") as f:
        pickle.dump(save_data, f)

    print(f"Saved {best_name} ({results[best_name]['accuracy']:.2%}) → {path}")
```

### Loading for Prediction

```python
def load_model(path="tennis_predictor.pkl"):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data  # dict with model, elo_system, feature_engineer, player_names, etc.
```

---

## 4. Prediction at Inference Time

At prediction time, we compute the same feature vector from current ELO state.

```python
import numpy as np


class MatchPredictor:
    """Use a trained model + ELO state to predict a match."""

    def __init__(self, model, feature_engineer, feature_names):
        self.model = model
        self.fe = feature_engineer
        self.feature_names = feature_names

    def predict(self, p1_id: int, p2_id: int, surface: str):
        """
        Returns (winner_id, win_probability).
        """
        elo = self.fe.elo_system

        # ELO features
        p1_elo = elo.get_rating(p1_id)
        p2_elo = elo.get_rating(p2_id)
        p1_surf = elo.get_surface_rating(p1_id, surface)
        p2_surf = elo.get_surface_rating(p2_id, surface)

        # H2H
        h2h_p1 = self.fe.h2h.get((p1_id, p2_id), 0)
        h2h_p2 = self.fe.h2h.get((p2_id, p1_id), 0)

        # Form
        r1 = self.fe.recent_matches.get(p1_id, [])[-50:]
        r2 = self.fe.recent_matches.get(p2_id, [])[-50:]
        f1 = sum(r1) / len(r1) if r1 else 0.5
        f2 = sum(r2) / len(r2) if r2 else 0.5

        features = np.array([
            p1_elo - p2_elo,                                  # elo_diff
            p1_surf - p2_surf,                                # surface_elo_diff
            (p1_elo + p1_surf) - (p2_elo + p2_surf),         # total_elo_diff
            h2h_p1 - h2h_p2,                                  # h2h_diff
            f1 - f2,                                           # form_diff
            0,                                                 # rank_diff (neutral)
            0,                                                 # height_diff (neutral)
            0,                                                 # age_diff (neutral)
        ]).reshape(1, -1)

        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(features)[0]
            p1_wins_prob = proba[1]
        else:
            pred = self.model.predict(features)[0]
            p1_wins_prob = 1.0 if pred == 1 else 0.0

        if p1_wins_prob >= 0.5:
            return p1_id, p1_wins_prob
        else:
            return p2_id, 1 - p1_wins_prob
```

---

## 5. Full Training Script Skeleton (`tennis_predictor.py`)

```python
"""
tennis_predictor.py — Full training pipeline.
"""
import os
from setup_data import setup
from features import FeatureEngineer, prepare_training_data, FEATURE_COLS
from data_loader import load_atp_data, load_player_data, clean_data
from models import train_all_models, save_best_model


def main():
    DATA_DIR = os.path.join(os.path.dirname(__file__), "tennis_atp")

    # 1. Ensure data exists
    setup()

    # 2. Load & clean
    matches = load_atp_data(DATA_DIR, start_year=1991, end_year=2025)
    players = load_player_data(DATA_DIR)
    player_names = dict(
        zip(players["player_id"], players["name_first"] + " " + players["name_last"])
    )
    matches = clean_data(matches)

    # 3. Feature engineering
    fe = FeatureEngineer()
    feature_df = fe.calculate_features(matches)

    # 4. Train/test split + balanced labels
    X_train, y_train, X_test, y_test, _, _ = prepare_training_data(
        feature_df, FEATURE_COLS, test_year=2025
    )

    # 5. Train models
    results = train_all_models(X_train, y_train, X_test, y_test)

    # 6. Save best model
    save_best_model(results, fe, FEATURE_COLS, player_names)


if __name__ == "__main__":
    main()
```

---

## 6. Tournament Bracket Simulation

After training, we can simulate any tournament draw.

```python
class TournamentSimulator:
    """Simulate a single-elimination bracket."""

    def __init__(self, predictor: MatchPredictor, player_names: dict):
        self.predictor = predictor
        self.player_names = player_names

    def simulate(self, player_ids: list, surface: str):
        """
        player_ids: list of (player_id, player_name) tuples.
        Returns list of round results and champion.
        """
        current = list(player_ids)
        results = []
        round_num = 0
        round_names = ["R1", "R2", "R3", "QF", "SF", "Final"]

        while len(current) > 1:
            rnd = round_names[min(round_num, len(round_names) - 1)]
            next_round = []

            for i in range(0, len(current), 2):
                p1_id, p1_name = current[i]
                p2_id, p2_name = current[i + 1]
                winner_id, conf = self.predictor.predict(p1_id, p2_id, surface)
                winner_name = p1_name if winner_id == p1_id else p2_name
                next_round.append((winner_id, winner_name))
                results.append({
                    "round": rnd,
                    "match": f"{p1_name} vs {p2_name}",
                    "winner": winner_name,
                    "confidence": conf,
                })

            current = next_round
            round_num += 1

        champion = current[0]
        return results, champion
```

---

## 7. Hyperparameter Tuning Tips

| Model | Key Hyperparameters | Suggested Range |
|---|---|---|
| Random Forest | `n_estimators`, `max_depth` | 100-500, 10-20 |
| XGBoost | `n_estimators`, `max_depth`, `learning_rate`, `subsample` | 100-300, 4-8, 0.05-0.2, 0.7-0.9 |
| Neural Network | `hidden_layer_sizes`, `learning_rate_init` | (64,32), (128,64,32), 0.001 |

Use `sklearn.model_selection.GridSearchCV` or `RandomizedSearchCV` for tuning.
