"""
train.py
--------
Full model training pipeline.

Reads:   data_files/features_2020_present.parquet
Writes:  data_files/tennis_predictor.pkl

Usage:
    python train.py            # train all models, save best
    python train.py --no-nn    # skip slow Neural Network
    python train.py --test-year 2024   # change test cutoff
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from features import FEATURE_COLS, prepare_training_data

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR   = "data_files"
PARQUET    = os.path.join(DATA_DIR, "features_2020_present.parquet")
MODEL_PATH = os.path.join(DATA_DIR, "tennis_predictor.pkl")


# ── Section 1 — ELO-only baseline ─────────────────────────────────────────────

def elo_baseline(X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Predict the player with the higher overall ELO wins (elo_diff > 0)."""
    elo_col = FEATURE_COLS.index("elo_diff")
    proba   = (X_test[:, elo_col] > 0).astype(float)   # 1.0 or 0.0
    preds   = proba.astype(int)
    acc     = float(np.mean(preds == y_test))
    auc     = float(roc_auc_score(y_test, proba))
    brier   = float(brier_score_loss(y_test, proba))
    ll      = float(log_loss(y_test, np.clip(proba, 1e-7, 1 - 1e-7)))
    print(f"ELO-only baseline : acc={acc:.2%}  auc={auc:.3f}  brier={brier:.4f}  logloss={ll:.4f}")
    return {"accuracy": acc, "auc": auc, "brier": brier, "log_loss": ll}


# ── Section 2 — Scikit-learn + XGBoost models ─────────────────────────────────

def _score_model(model, X_test, y_test) -> dict:
    """Compute all evaluation metrics for a fitted model."""
    preds = model.predict(X_test)
    acc   = float(accuracy_score(y_test, preds))
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)[:, 1]
    else:
        proba = preds.astype(float)
    auc    = float(roc_auc_score(y_test, proba))
    brier  = float(brier_score_loss(y_test, proba))
    ll     = float(log_loss(y_test, proba))
    return {"accuracy": acc, "auc": auc, "brier": brier, "log_loss": ll, "_proba": proba}


def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    feature_names: list[str] = FEATURE_COLS,
    train_nn: bool = True,
) -> dict:
    """
    Train Decision Tree, Random Forest, XGBoost, and (optionally) Neural Network.
    Returns a dict: {model_name: {"model": ..., "accuracy": float}}
    """
    results: dict = {}

    # ---- Decision Tree ----
    dt = DecisionTreeClassifier(max_depth=10, min_samples_split=5, random_state=42)
    dt.fit(X_train, y_train)
    dt_scores = _score_model(dt, X_test, y_test)
    results["Decision Tree"] = {"model": dt, **dt_scores}
    print(f"Decision Tree     : acc={dt_scores['accuracy']:.2%}  auc={dt_scores['auc']:.3f}  brier={dt_scores['brier']:.4f}")

    # ---- Random Forest ----
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_scores = _score_model(rf, X_test, y_test)
    results["Random Forest"] = {"model": rf, **rf_scores}
    print(f"Random Forest     : acc={rf_scores['accuracy']:.2%}  auc={rf_scores['auc']:.3f}  brier={rf_scores['brier']:.4f}")

    print("\n  Feature importances (RF):")
    for feat, imp in sorted(
        zip(feature_names, rf.feature_importances_), key=lambda x: -x[1]
    ):
        print(f"    {feat:22s}: {imp:.4f}")

    # ---- XGBoost ----
    try:
        import xgboost as xgb  # noqa: PLC0415

        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
        )
        xgb_model.fit(X_train, y_train)
        xgb_scores = _score_model(xgb_model, X_test, y_test)
        results["XGBoost"] = {"model": xgb_model, **xgb_scores}
        print(f"\nXGBoost           : acc={xgb_scores['accuracy']:.2%}  auc={xgb_scores['auc']:.3f}  brier={xgb_scores['brier']:.4f}")
    except ImportError:
        print("XGBoost not installed — pip install xgboost")

    # ---- Neural Network (optional — slow) ----
    if train_nn:
        nn = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=500,
            random_state=42,
            early_stopping=True,
            verbose=False,
        )
        nn.fit(X_train, y_train)
        nn_scores = _score_model(nn, X_test, y_test)
        results["Neural Network"] = {"model": nn, **nn_scores}
        print(f"Neural Network    : acc={nn_scores['accuracy']:.2%}  auc={nn_scores['auc']:.3f}  brier={nn_scores['brier']:.4f}")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print(f"{'MODEL':<20} {'ACC':>7} {'AUC':>7} {'BRIER':>8} {'LOGLOSS':>9}")
    print("-" * 60)
    best_acc = max(v["accuracy"] for v in results.values())
    for name, data in results.items():
        marker = " ◀" if data["accuracy"] == best_acc else ""
        print(
            f"  {name:<20} {data['accuracy']:>6.2%} {data['auc']:>7.3f}"
            f" {data['brier']:>8.4f} {data['log_loss']:>9.4f}{marker}"
        )
    print()
    return results


# ── Section 3 — Model serialization ───────────────────────────────────────────

def save_best_model(
    results: dict,
    feature_names: list[str],
    y_test: np.ndarray | None = None,
    meta: dict | None = None,
    output_path: str = MODEL_PATH,
) -> str:
    """Persist the best model (by accuracy) + full evaluation data to a pickle file."""
    best_name = max(results, key=lambda k: results[k]["accuracy"])
    best      = results[best_name]
    best_model = best["model"]

    # ---- Feature importances (for RF / XGBoost / tree-based) ----
    feat_imp = None
    if hasattr(best_model, "feature_importances_"):
        feat_imp = [
            {"feature": f, "importance": float(i)}
            for f, i in sorted(
                zip(feature_names, best_model.feature_importances_),
                key=lambda x: -x[1],
            )
        ]

    # ---- Calibration curve (best model) ----
    cal_data = None
    if y_test is not None and best.get("_proba") is not None:
        frac_pos, mean_pred = calibration_curve(y_test, best["_proba"], n_bins=10)
        cal_data = {
            "fraction_of_positives": frac_pos.tolist(),
            "mean_predicted_value":  mean_pred.tolist(),
        }

    # ---- ROC curve (best model) ----
    roc_data = None
    if y_test is not None and best.get("_proba") is not None:
        fpr, tpr, _ = roc_curve(y_test, best["_proba"])
        # downsample to ≤200 points so pkl stays small
        step = max(1, len(fpr) // 200)
        roc_data = {
            "fpr": fpr[::step].tolist(),
            "tpr": tpr[::step].tolist(),
            "auc": best["auc"],
        }

    # ---- All metrics (strip internal _proba from results) ----
    all_metrics = {
        k: {m: v for m, v in d.items() if not m.startswith("_") and m != "model"}
        for k, d in results.items()
    }

    save_data = {
        "model":          best_model,
        "model_name":     best_name,
        "feature_names":  feature_names,
        "accuracy":       best["accuracy"],
        "auc":            best["auc"],
        "brier":          best["brier"],
        "log_loss":       best["log_loss"],
        "all_metrics":    all_metrics,
        "feature_importances": feat_imp,
        "calibration_data":    cal_data,
        "roc_data":            roc_data,
        **(meta or {}),
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(save_data, f)

    print(f"Saved: {output_path}")
    print(f"  Best model : {best_name}  acc={best['accuracy']:.2%}  auc={best['auc']:.3f}  brier={best['brier']:.4f}")
    return output_path


def load_model(path: str = MODEL_PATH) -> dict:
    """Load the pickled model dict.  Keys: model, model_name, feature_names, accuracy."""
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(test_year: int = 2025, train_nn: bool = True) -> dict:
    print(f"Loading features from {PARQUET} …")
    df = pd.read_parquet(PARQUET)
    print(f"  Shape: {df.shape}")

    # Prepare balanced train / test arrays
    X_train, y_train, X_test, y_test, tr_df, te_df = prepare_training_data(
        df, FEATURE_COLS, test_year=test_year
    )

    print(f"\nRunning ELO-only baseline on {len(y_test):,} test rows …")
    elo_metrics = elo_baseline(X_test, y_test)

    print(f"\nTraining models (train ≤{test_year-1}, test ≥{test_year}) …")
    results = train_all_models(X_train, y_train, X_test, y_test, FEATURE_COLS, train_nn)

    meta = {
        "elo_baseline_acc": elo_metrics["accuracy"],
        "elo_baseline":     elo_metrics,
        "test_year":        test_year,
        "train_rows":       len(y_train),
        "test_rows":        len(y_test),
        "parquet_path":     PARQUET,
        "train_date":       pd.Timestamp.now().isoformat(),
    }
    save_best_model(results, FEATURE_COLS, y_test=y_test, meta=meta)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train tennis prediction models.")
    parser.add_argument("--test-year", type=int, default=2025,
                        help="First year used as test set (default: 2025)")
    parser.add_argument("--no-nn", action="store_true",
                        help="Skip the slower Neural Network model")
    args = parser.parse_args()
    main(test_year=args.test_year, train_nn=not args.no_nn)
