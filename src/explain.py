"""Explainability helpers: global importance and per-student SHAP reasons.

- permutation_importance_table: how much the model's ranking quality drops when
  one original column is shuffled (model-agnostic, on held-out data).
- logistic_coefficients: signed effects from a logistic regression (odds ratios).
- shap_contributions / explain_student: per-student contributions, with one-hot
  columns summed back into the original feature so a teacher sees "Mother's job",
  not "Mjob_at_home".

SHAP units depend on the model: Random Forest -> probability points,
Gradient Boosting -> log-odds. Signs and ranking mean the same thing in both.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.features import all_features, make_preprocessor

# Plain-English names for the app and charts.
FEATURE_LABELS = {
    "school": "School", "sex": "Sex", "age": "Age", "address": "Home address (urban/rural)",
    "famsize": "Family size", "Pstatus": "Parents living together", "Medu": "Mother's education",
    "Fedu": "Father's education", "Mjob": "Mother's job", "Fjob": "Father's job",
    "reason": "Reason for choosing school", "guardian": "Guardian",
    "traveltime": "Travel time to school", "studytime": "Weekly study time",
    "failures": "Past class failures", "schoolsup": "Extra school support",
    "famsup": "Family educational support", "paid": "Paid extra classes",
    "activities": "Extra-curricular activities", "nursery": "Attended nursery school",
    "higher": "Wants higher education", "internet": "Internet at home",
    "romantic": "In a relationship", "famrel": "Family relationship quality",
    "freetime": "Free time after school", "goout": "Going out with friends",
    "Dalc": "Weekday alcohol use", "Walc": "Weekend alcohol use", "health": "Health status",
    "absences": "Absences", "G1": "First-period grade (G1)",
}


def label(col: str) -> str:
    return FEATURE_LABELS.get(col, col)


# ----------------------------------------------------------------------------- global
def permutation_importance_table(pipeline, X: pd.DataFrame, y, n_repeats: int = 50,
                                 seed: int = 42) -> pd.DataFrame:
    """Drop in average precision when each ORIGINAL column is shuffled."""
    r = permutation_importance(pipeline, X, y, scoring="average_precision",
                               n_repeats=n_repeats, random_state=seed)
    return (pd.DataFrame({"feature": X.columns, "importance": r.importances_mean,
                          "std": r.importances_std})
            .assign(label=lambda d: d.feature.map(label))
            .sort_values("importance", ascending=False, ignore_index=True))


def logistic_coefficients(version: str, X: pd.DataFrame, y, C: float = 0.1,
                          drop: list[str] | None = None) -> pd.DataFrame:
    """Fit a balanced logistic regression; return coefficients and odds ratios.

    Numerics are standardised, so a numeric coefficient is the effect of +1 standard
    deviation; a categorical one is the effect of that category vs the reference.
    """
    pipe = Pipeline([("prep", make_preprocessor(version, drop)),
                     ("model", LogisticRegression(C=C, class_weight="balanced", max_iter=5000))])
    pipe.fit(X, y)
    names = pipe.named_steps["prep"].get_feature_names_out()
    coef = pipe.named_steps["model"].coef_[0]
    return (pd.DataFrame({"term": names, "coef": coef, "odds_ratio": np.exp(coef)})
            .assign(abs_coef=lambda d: d.coef.abs())
            .sort_values("abs_coef", ascending=False, ignore_index=True)
            .drop(columns="abs_coef"))


# ----------------------------------------------------------------------------- SHAP
def _column_groups(prep) -> dict[str, list[int]]:
    """Map each original column to the indices of its transformed columns."""
    groups: dict[str, list[int]] = {}
    i = 0
    for name, trans, cols in prep.transformers_:
        if name == "remainder" or trans == "drop":
            continue
        if hasattr(trans, "categories_"):  # one-hot encoder
            for col, cats, drop_idx in zip(cols, trans.categories_,
                                           trans.drop_idx_ if trans.drop_idx_ is not None
                                           else [None] * len(cols)):
                width = len(cats) - (0 if drop_idx is None else 1)
                groups[col] = list(range(i, i + width))
                i += width
        else:
            for col in cols:
                groups[col] = [i]
                i += 1
    return groups


def shap_contributions(bundle: dict, X: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """SHAP contributions per ORIGINAL feature (rows = students), and the base value."""
    prep = bundle["pipeline"].named_steps["prep"]
    model = bundle["pipeline"].named_steps["model"]
    Xt = prep.transform(X[bundle["features"]])
    explainer = shap.TreeExplainer(model)
    sv = explainer(Xt)
    values = sv.values[..., 1] if sv.values.ndim == 3 else sv.values
    base = np.ravel(explainer.expected_value)
    base = float(base[1] if base.size == 2 else base[0])
    grouped = {col: values[:, idx].sum(axis=1) for col, idx in _column_groups(prep).items()}
    return pd.DataFrame(grouped, index=X.index), base


def explain_student(bundle: dict, student: pd.DataFrame, top_k: int = 3) -> list[dict]:
    """Top reasons pushing ONE student's risk up: [{feature, label, value, contribution}]."""
    contrib, _ = shap_contributions(bundle, student.iloc[[0]])
    row = contrib.iloc[0].sort_values(ascending=False)
    reasons = []
    for feat, c in row.items():
        if c <= 0 or len(reasons) == top_k:
            break
        reasons.append({"feature": feat, "label": label(feat),
                        "value": student.iloc[0][feat], "contribution": float(c)})
    return reasons


def global_shap_importance(bundle: dict, X: pd.DataFrame) -> pd.DataFrame:
    """Mean |SHAP| per original feature: average impact on individual predictions."""
    contrib, _ = shap_contributions(bundle, X)
    return (contrib.abs().mean().rename("mean_abs_shap").reset_index()
            .rename(columns={"index": "feature"})
            .assign(label=lambda d: d.feature.map(label))
            .sort_values("mean_abs_shap", ascending=False, ignore_index=True))


__all__ = [
    "FEATURE_LABELS", "label", "permutation_importance_table", "logistic_coefficients",
    "shap_contributions", "explain_student", "global_shap_importance", "all_features",
]
