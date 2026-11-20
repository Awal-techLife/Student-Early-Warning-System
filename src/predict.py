"""Scoring logic shared by the Streamlit app and the tests.

- load_bundle(version): the reduced model (no sex / alcohol / relationship data).
- validate_input(df, bundle): check a teacher's data before scoring.
- score(df, bundle): risk score, risk band, top reasons and suggested actions per student.

Risk bands (scores are NOT probabilities; see notebook 03):
    High    score >= threshold         flagged; catches ~80% (A) / ~90% (B) of at-risk students
    Medium  score >= watch_threshold   "watch list"; together with High catches ~95%
    Low     below both
"""

from __future__ import annotations

from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from src.data import load_clean
from src.explain import label, shap_contributions
from src.train import MODELS_DIR

# Reasons that describe the setting rather than the student are not shown as "reasons".
# (A teacher at school MS already knows the student attends MS.)
HIDDEN_REASONS = {"school"}

BANDS = ["High", "Medium", "Low"]

# A reason is shown only if it is a real driver for THIS student:
#   - at least MIN_SHARE of the student's largest reason, and
#   - above a small floor (SHAP units differ: RF = probability, GB = log-odds).
# Tiny contributions are mostly tree-model noise (e.g. "family relationship: excellent"
# nudging risk up by 1 point) and would confuse a teacher.
MIN_SHARE = 0.25
MIN_CONTRIBUTION = {"Random Forest": 0.02, "Gradient Boosting": 0.10}
NO_SINGLE_REASON = "No single strong factor: the score comes from several small ones."

VALUE_LABELS = {
    "Medu": {0: "none", 1: "primary (to 4th grade)", 2: "5th–9th grade", 3: "secondary", 4: "higher education"},
    "Fedu": {0: "none", 1: "primary (to 4th grade)", 2: "5th–9th grade", 3: "secondary", 4: "higher education"},
    "studytime": {1: "< 2 h/week", 2: "2–5 h/week", 3: "5–10 h/week", 4: "> 10 h/week"},
    "traveltime": {1: "< 15 min", 2: "15–30 min", 3: "30–60 min", 4: "> 1 hour"},
    "failures": {0: "none", 1: "1", 2: "2", 3: "3 or more"},
    "famrel": {1: "1 – very bad", 2: "2", 3: "3", 4: "4", 5: "5 – excellent"},
    "freetime": {1: "1 – very low", 2: "2", 3: "3", 4: "4", 5: "5 – very high"},
    "goout": {1: "1 – very low", 2: "2", 3: "3", 4: "4", 5: "5 – very high"},
    "health": {1: "1 – very bad", 2: "2", 3: "3", 4: "4", 5: "5 – very good"},
    "address": {"U": "urban", "R": "rural"},
    "famsize": {"LE3": "3 or fewer", "GT3": "more than 3"},
    "Pstatus": {"T": "living together", "A": "apart"},
    "school": {"GP": "Gabriel Pereira (GP)", "MS": "Mousinho da Silveira (MS)"},
    "Mjob": {"at_home": "at home", "health": "health care", "services": "civil services",
             "teacher": "teacher", "other": "other"},
    "Fjob": {"at_home": "at home", "health": "health care", "services": "civil services",
             "teacher": "teacher", "other": "other"},
    "reason": {"home": "close to home", "reputation": "school reputation",
               "course": "course preference", "other": "other"},
}

# Generic, low-stakes teacher actions linked to the main reason. Deliberately supportive.
ACTIONS = {
    "failures": "Review gaps from previous years and agree a short catch-up plan.",
    "higher": "Have a conversation about goals and future options (careers, further study).",
    "studytime": "Help the student set up a weekly study routine; suggest homework club.",
    "goout": "Talk about balancing social time with study time.",
    "freetime": "Talk about how free time is used; suggest structured activities or study sessions.",
    "G1": "Go through the first test together and plan targeted revision.",
    "famrel": "Check in on wellbeing; involve the form tutor or counsellor if appropriate.",
    "health": "Check in on wellbeing; refer to the school nurse or counsellor if needed.",
    "famsup": "Contact the family to discuss how they can support learning at home.",
    "Medu": "Contact the family to explain how they can support homework and revision.",
    "Fedu": "Contact the family to explain how they can support homework and revision.",
    "schoolsup": "Consider the school's extra-support programme.",
    "paid": "Consider free school-based extra classes as an alternative.",
    "traveltime": "Check whether travel time is limiting homework or attendance.",
    "internet": "Make sure the student can access materials offline or on school computers.",
    "activities": "Encourage joining a club or extra-curricular activity.",
    "age": "Check for previous repeated years and any gaps they left.",
    "absences": "Check attendance records and follow up on patterns.",
}
DEFAULT_ACTION = "Have a short one-to-one check-in to understand how the student is doing."


@lru_cache(maxsize=None)
def load_bundle(version: str) -> dict:
    return joblib.load(MODELS_DIR / f"version_{version}_reduced.joblib")


@lru_cache(maxsize=1)
def valid_values() -> dict:
    """Allowed values per column, taken from the training data."""
    df = load_clean("por")
    out = {}
    for col in df.columns:
        if df[col].dtype.kind in "iuf":
            out[col] = (int(df[col].min()), int(df[col].max()))
        else:
            out[col] = sorted(df[col].unique().tolist())
    out["G1"] = (0, 20)  # full grade scale, not just the observed range
    return out


def pretty(col: str, value) -> str:
    try:
        return VALUE_LABELS.get(col, {}).get(value, str(value))
    except TypeError:
        return str(value)


def validate_input(df: pd.DataFrame, bundle: dict) -> tuple[pd.DataFrame, list[str]]:
    """Return (cleaned frame with only the model's features, list of error messages)."""
    errors = []
    missing = [c for c in bundle["features"] if c not in df.columns]
    if missing:
        return df, [f"Missing columns: {', '.join(missing)}"]
    if df.empty:
        return df, ["The file has no rows."]
    X = df[bundle["features"]].copy()
    allowed = valid_values()
    for col in bundle["features"]:
        rule = allowed[col]
        if isinstance(rule, tuple):
            vals = pd.to_numeric(X[col], errors="coerce")
            bad = vals.isna() | ~vals.between(*rule)
            if bad.any():
                errors.append(f"`{col}` must be a whole number from {rule[0]} to {rule[1]} "
                              f"(problem in row(s) {_rows(bad)}).")
            else:
                X[col] = vals.astype(int)
        else:
            X[col] = X[col].astype(str).str.strip()
            bad = ~X[col].isin(rule)
            if bad.any():
                errors.append(f"`{col}` must be one of {rule} (problem in row(s) {_rows(bad)}).")
    return X, errors


def _rows(mask: pd.Series) -> str:
    idx = np.flatnonzero(mask.to_numpy()) + 1  # 1-based, as a teacher counts rows
    shown = ", ".join(map(str, idx[:5]))
    return shown + (f" and {len(idx) - 5} more" if len(idx) > 5 else "")


def band(score: float, bundle: dict) -> str:
    if score >= bundle["threshold"]:
        return "High"
    if score >= bundle["watch_threshold"]:
        return "Medium"
    return "Low"


def score(X: pd.DataFrame, bundle: dict, top_k: int = 3) -> pd.DataFrame:
    """Score validated rows. Returns one row per student with band and reasons."""
    X = X[bundle["features"]].reset_index(drop=True)
    proba = bundle["pipeline"].predict_proba(X)[:, 1]
    contrib, _ = shap_contributions(bundle, X)
    contrib = contrib.drop(columns=[c for c in HIDDEN_REASONS if c in contrib.columns])

    floor = MIN_CONTRIBUTION.get(bundle["model_name"], 0.0)
    rows = []
    for i in range(len(X)):
        b = band(proba[i], bundle)
        c = contrib.iloc[i].sort_values(ascending=False)
        top = c[(c >= floor) & (c >= MIN_SHARE * c.iloc[0])].head(top_k) if c.iloc[0] > 0 else c[:0]
        reasons = [] if b == "Low" else [
            {"feature": f, "label": label(f), "value": pretty(f, X.iloc[i][f]),
             "contribution": float(v)} for f, v in top.items()]
        actions = list(dict.fromkeys(ACTIONS.get(r["feature"], DEFAULT_ACTION) for r in reasons))
        if b == "Low":
            summary = ""
        elif reasons:
            summary = "; ".join(f"{r['label']}: {r['value']}" for r in reasons)
        else:
            summary = NO_SINGLE_REASON
        rows.append({
            "risk_score": float(proba[i]),
            "risk_band": b,
            "reasons": reasons,
            "top_reasons": summary,
            "suggested_actions": [] if b == "Low" else (actions or [DEFAULT_ACTION]),
        })
    return pd.DataFrame(rows)
