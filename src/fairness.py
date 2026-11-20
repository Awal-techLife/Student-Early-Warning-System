"""Fairness audit and the sensitive-feature test.

Usage:
    python -m src.fairness

1. Group audit: recall, precision, false-positive rate and flag rate for each group of
   `sex`, `address` and `school`. Uses 5-fold out-of-fold predictions on ALL 649 students:
   the test set has only 20 at-risk students, too few to split into groups.
   Recall gets a bootstrap 95% interval because some groups are small.
2. Sensitive-feature test: retrain each version without `sex`, `Dalc`, `Walc`, `romantic`
   (and additionally without `school`) and compare performance and fairness gaps.
   Same model type and hyperparameters; threshold re-chosen the same way.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import cross_val_predict

from src.data import load_clean
from src.features import VERSION_LABELS, all_features, make_preprocessor
from src.train import CV, MODELS_DIR, REPORTS_DIR, choose_threshold, get_split, metrics, per_fold

SENSITIVE = ["sex", "Dalc", "Walc", "romantic"]
GROUP_COLS = ["sex", "address", "school"]
VARIANTS = {
    "All features": [],
    "No sensitive features": SENSITIVE,
    "No sensitive + no school": SENSITIVE + ["school"],
}


def oof_proba(pipeline, X: pd.DataFrame, y) -> np.ndarray:
    return cross_val_predict(clone(pipeline), X, y, cv=CV, method="predict_proba")[:, 1]


def group_metrics(df: pd.DataFrame, proba: np.ndarray, threshold: float, col: str,
                  n_boot: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Per-group error rates for one attribute, with a bootstrap CI on recall."""
    rng = np.random.default_rng(seed)
    pred = proba >= threshold
    rows = []
    for g, idx in df.groupby(col).indices.items():
        y, p = df["at_risk"].to_numpy()[idx] == 1, pred[idx]
        hits = p[y].astype(float)  # 1 if an at-risk student was flagged
        boots = rng.choice(hits, size=(n_boot, len(hits)), replace=True).mean(axis=1)
        rows.append({
            "attribute": col, "group": g, "n": len(idx), "n_at_risk": int(y.sum()),
            "base_rate": y.mean(), "flag_rate": p.mean(),
            "recall": hits.mean(), "recall_lo": np.percentile(boots, 2.5),
            "recall_hi": np.percentile(boots, 97.5),
            "precision": (p & y).sum() / max(p.sum(), 1),
            "fpr": (p & ~y).sum() / max((~y).sum(), 1),
        })
    return pd.DataFrame(rows)


def recall_gap(df, proba, threshold, col) -> float:
    g = group_metrics(df, proba, threshold, col, n_boot=1)
    return float(g.recall.max() - g.recall.min())


def build_variant(bundle: dict, drop: list[str]):
    """Same model + hyperparameters as the saved bundle, different feature set."""
    pipe = clone(bundle["pipeline"])
    pipe.steps[0] = ("prep", make_preprocessor(bundle["version"], drop))
    return pipe, all_features(bundle["version"], drop)


def main() -> None:
    df = load_clean("por")
    train, test = get_split(df)
    sens_rows, group_tables = [], []

    for version in ["A", "B"]:
        bundle = joblib.load(MODELS_DIR / f"version_{version}.joblib")
        print(f"Version {VERSION_LABELS[version]} ({bundle['model_name']})")
        for variant, drop in VARIANTS.items():
            pipe, feats = build_variant(bundle, drop)

            # Same protocol as train.py: threshold from out-of-fold predictions on train.
            oof_tr = oof_proba(pipe, train[feats], train["at_risk"])
            thr = choose_threshold(train["at_risk"], oof_tr)
            cv = per_fold(train["at_risk"], oof_tr, thr)
            pipe.fit(train[feats], train["at_risk"])
            te = metrics(test["at_risk"], pipe.predict_proba(test[feats])[:, 1], thr)

            # Fairness on all 649 students via out-of-fold predictions.
            oof_all = oof_proba(pipe, df[feats], df["at_risk"])
            for col in GROUP_COLS:
                group_tables.append(group_metrics(df, oof_all, thr, col)
                                    .assign(version=version, variant=variant))

            sens_rows.append({
                "version": version, "variant": variant, "n_features": len(feats),
                "threshold": thr, "cv_roc_auc": cv["cv_roc_auc"], "cv_roc_auc_std": cv["cv_roc_auc_std"],
                "cv_avg_precision": cv["cv_avg_precision"], "cv_precision": cv["cv_precision"],
                "test_recall": te["recall"], "test_precision": te["precision"],
                "test_roc_auc": te["roc_auc"],
                **{f"recall_gap_{c}": recall_gap(df, oof_all, thr, c) for c in GROUP_COLS},
            })
            print(f"  {variant:<26} CV AUC {cv['cv_roc_auc']:.3f}  test recall {te['recall']:.2f}  "
                  f"precision {te['precision']:.2f}")

            if variant == "No sensitive features":
                joblib.dump({**bundle, "pipeline": pipe, "threshold": thr, "features": feats,
                             "dropped_features": drop,
                             "cv_metrics": cv,
                             "test_metrics": {f"test_{k}": v for k, v in te.items()}},
                            MODELS_DIR / f"version_{version}_reduced.joblib")

    sens = pd.DataFrame(sens_rows)
    groups = pd.concat(group_tables, ignore_index=True)
    sens.to_csv(REPORTS_DIR / "sensitive_feature_test.csv", index=False)
    groups.to_csv(REPORTS_DIR / "fairness_groups.csv", index=False)

    print("\nSensitive-feature test")
    print(sens.drop(columns=["threshold"]).round(3).to_string(index=False))
    print("\nGroup audit (All features vs No sensitive features)")
    show = groups[groups.variant != "No sensitive + no school"]
    print(show[["version", "variant", "attribute", "group", "n", "n_at_risk", "base_rate",
                "flag_rate", "recall", "recall_lo", "recall_hi", "precision", "fpr"]]
          .round(2).to_string(index=False))


if __name__ == "__main__":
    main()
