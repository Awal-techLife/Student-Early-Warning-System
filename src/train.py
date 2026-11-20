"""Train, evaluate and save the early-warning models.

Usage:
    python -m src.train

What it does, for each version (A, A_absences, B):
  1. Hold out 20% of students as a test set (stratified, fixed seed). Never used for choices.
  2. On the other 80%, with stratified 5-fold CV:
       - evaluate simple baselines (dummy, one-line rules),
       - tune 3 models (logistic regression, random forest, gradient boosting),
       - pick each model's decision threshold from out-of-fold predictions.
  3. Refit on the 80%, then score the untouched test set once.
  4. Save the best model per version to models/ and all results to reports/.
"""

from __future__ import annotations

import json
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split,
)
from sklearn.pipeline import Pipeline

from src.data import PROJECT_ROOT, load_clean
from src.features import VERSION_LABELS, all_features, make_preprocessor

SEED = 42
TEST_SIZE = 0.20
N_FOLDS = 5

# Decision policy: catch at least 80% of struggling students; among thresholds that
# do, choose the one with the fewest false alarms (highest precision).
TARGET_RECALL = 0.80

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

CV = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# Small grids on purpose: 520 training students is not enough to tune many knobs
# without overfitting the validation folds.
MODEL_SPECS = {
    "Logistic Regression": (
        LogisticRegression(class_weight="balanced", max_iter=5000),
        {"model__C": [0.01, 0.1, 1.0, 10.0]},
    ),
    "Random Forest": (
        RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample",
                               random_state=SEED, n_jobs=-1),
        {"model__max_depth": [4, 8, None], "model__min_samples_leaf": [1, 5, 10]},
    ),
    "Gradient Boosting": (
        HistGradientBoostingClassifier(class_weight="balanced", max_iter=200,
                                       early_stopping=False, random_state=SEED),
        {"model__learning_rate": [0.03, 0.1], "model__max_depth": [2, 3],
         "model__min_samples_leaf": [10, 20]},
    ),
}

# One-line rules a teacher could apply without any model. Score = 1 means "flag".
RULES = {
    "Rule: failures >= 1": lambda X: (X["failures"] >= 1).astype(float),
    "Rule: G1 < 10": lambda X: (X["G1"] < 10).astype(float),
}
RULES_FOR_VERSION = {
    "A": ["Rule: failures >= 1"],
    "A_absences": ["Rule: failures >= 1"],
    "B": ["Rule: failures >= 1", "Rule: G1 < 10"],
}


# ----------------------------------------------------------------------------- helpers
def get_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic stratified 80/20 split. Reused by later phases."""
    return train_test_split(df, test_size=TEST_SIZE, stratify=df["at_risk"],
                            random_state=SEED)


def choose_threshold(y_true, proba, target_recall: float = TARGET_RECALL) -> float:
    """Highest-precision threshold whose recall is still >= target_recall."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    precision, recall = precision[:-1], recall[:-1]  # align with thresholds
    ok = recall >= target_recall
    if not ok.any():
        return float(thresholds.min())
    best = np.flatnonzero(ok)[np.argmax(precision[ok])]
    return float(thresholds[best])


def metrics(y_true, proba, threshold: float) -> dict[str, float]:
    y_true = np.asarray(y_true)
    pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # precision is undefined when nothing is flagged
        return {
            "recall": recall_score(y_true, pred, zero_division=0),
            "precision": precision_score(y_true, pred, zero_division=0),
            "f1": f1_score(y_true, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, proba),
            "avg_precision": average_precision_score(y_true, proba),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        }


def per_fold(y, proba, threshold) -> dict[str, float]:
    """Mean and std of metrics across the CV folds (from out-of-fold predictions)."""
    rows = [metrics(y.iloc[val], proba[val], threshold) for _, val in CV.split(np.zeros(len(y)), y)]
    out = {}
    for k in ["recall", "precision", "f1", "roc_auc", "avg_precision"]:
        vals = [r[k] for r in rows]
        out[f"cv_{k}"], out[f"cv_{k}_std"] = float(np.mean(vals)), float(np.std(vals))
    return out


def row(version, name, cv_stats, test_stats, threshold, params=None) -> dict:
    return {"version": version, "model": name, "threshold": threshold,
            **cv_stats, **{f"test_{k}": v for k, v in test_stats.items()},
            "params": json.dumps(params or {})}


# ----------------------------------------------------------------------------- training
def run_version(version: str, train: pd.DataFrame, test: pd.DataFrame) -> tuple[list[dict], dict]:
    feats = all_features(version)
    X_tr, y_tr = train[feats], train["at_risk"]
    X_te, y_te = test[feats], test["at_risk"]
    rows, fitted = [], {}

    # Baseline 1: dummy that predicts the class prior (flags nobody at threshold 0.5).
    dummy = DummyClassifier(strategy="prior")
    oof = cross_val_predict(dummy, X_tr, y_tr, cv=CV, method="predict_proba")[:, 1]
    dummy.fit(X_tr, y_tr)
    rows.append(row(version, "Dummy (always 'pass')", per_fold(y_tr, oof, 0.5),
                    metrics(y_te, dummy.predict_proba(X_te)[:, 1], 0.5), 0.5))

    # Baseline 2: one-line rules (nothing to fit, so CV = evaluate on each fold).
    for rule in RULES_FOR_VERSION[version]:
        score_tr = RULES[rule](train).to_numpy()
        rows.append(row(version, rule, per_fold(y_tr, score_tr, 0.5),
                        metrics(y_te, RULES[rule](test).to_numpy(), 0.5), 0.5))

    # The three ML models.
    for name, (estimator, grid) in MODEL_SPECS.items():
        pipe = Pipeline([("prep", make_preprocessor(version)), ("model", estimator)])
        search = GridSearchCV(pipe, grid, cv=CV, scoring="average_precision", n_jobs=-1)
        search.fit(X_tr, y_tr)
        best = search.best_estimator_

        # Threshold chosen from out-of-fold probabilities: the test set is never consulted.
        oof = cross_val_predict(best, X_tr, y_tr, cv=CV, method="predict_proba")[:, 1]
        thr = choose_threshold(y_tr, oof)
        test_proba = best.predict_proba(X_te)[:, 1]

        rows.append(row(version, name, per_fold(y_tr, oof, thr),
                        metrics(y_te, test_proba, thr), thr, search.best_params_))
        fitted[name] = (best, thr, test_proba)
        print(f"  {name:<20} best {search.best_params_}  threshold={thr:.2f}")

    return rows, fitted


def results_markdown(res: pd.DataFrame) -> str:
    """README-ready table: CV mean ± std, then held-out test."""
    pm = lambda m, s: f"{m:.2f} ± {s:.2f}"
    lines = [
        "| Version | Model | CV recall | CV precision | CV ROC-AUC | Test recall | Test precision | Test F1 | Test ROC-AUC |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in res.iterrows():
        lines.append(
            f"| {VERSION_LABELS[r.version]} | {r.model} | {pm(r.cv_recall, r.cv_recall_std)} | "
            f"{pm(r.cv_precision, r.cv_precision_std)} | {pm(r.cv_roc_auc, r.cv_roc_auc_std)} | "
            f"{r.test_recall:.2f} | {r.test_precision:.2f} | {r.test_f1:.2f} | {r.test_roc_auc:.2f} |"
        )
    return "\n".join(lines)


def main() -> None:
    df = load_clean("por")
    train, test = get_split(df)
    print(f"Train {len(train)} ({train.at_risk.sum()} at risk) | "
          f"Test {len(test)} ({test.at_risk.sum()} at risk)\n")

    all_rows, robustness = [], []
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    for version in ["A", "A_absences", "B"]:
        print(f"Version {VERSION_LABELS[version]}")
        rows, fitted = run_version(version, train, test)
        all_rows += rows

        # Pick the best ML model by CV average precision (threshold-free, suits imbalance).
        ml = [r for r in rows if r["model"] in MODEL_SPECS]
        winner = max(ml, key=lambda r: r["cv_avg_precision"])
        pipe, thr, test_proba = fitted[winner["model"]]
        print(f"  -> selected: {winner['model']}\n")

        # Robustness: does performance hold without the 15 probable dropouts (G3 = 0)?
        keep = (test["no_final_exam"] == 0).to_numpy()
        m_all = metrics(test["at_risk"], test_proba, thr)
        m_sub = metrics(test["at_risk"][keep], test_proba[keep], thr)
        robustness.append({"version": version, "model": winner["model"],
                           "recall_all": m_all["recall"], "recall_without_g3_zero": m_sub["recall"],
                           "precision_all": m_all["precision"],
                           "precision_without_g3_zero": m_sub["precision"],
                           "n_g3_zero_in_test": int((~keep).sum())})

        if version in ("A", "B"):  # the two versions the app offers
            joblib.dump({
                "pipeline": pipe, "threshold": thr, "version": version,
                "version_label": VERSION_LABELS[version], "model_name": winner["model"],
                "features": all_features(version), "target_recall": TARGET_RECALL,
                "cv_metrics": {k: v for k, v in winner.items() if k.startswith("cv_")},
                "test_metrics": {k: v for k, v in winner.items() if k.startswith("test_")},
                "sklearn_version": sklearn.__version__, "seed": SEED,
            }, MODELS_DIR / f"version_{version}.joblib")

    res = pd.DataFrame(all_rows)
    res.to_csv(REPORTS_DIR / "results.csv", index=False)
    rob = pd.DataFrame(robustness)
    rob.to_csv(REPORTS_DIR / "robustness_g3_zero.csv", index=False)
    (REPORTS_DIR / "results.md").write_text(results_markdown(res), encoding="utf-8")

    print(results_markdown(res))
    print("\nRobustness (selected models, test set, with vs without G3 = 0 students):")
    print(rob.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
