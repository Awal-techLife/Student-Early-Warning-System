"""Tests for SHAP explanations and fairness metrics (use the saved reduced models)."""

import joblib
import numpy as np
import pandas as pd
import pytest

from src.data import load_clean
from src.explain import explain_student, shap_contributions
from src.fairness import SENSITIVE, group_metrics
from src.train import MODELS_DIR, get_split


@pytest.fixture(scope="module")
def data():
    df = load_clean("por")
    return get_split(df)


@pytest.mark.parametrize("version", ["A", "B"])
def test_reduced_models_exclude_sensitive_features(version):
    b = joblib.load(MODELS_DIR / f"version_{version}_reduced.joblib")
    assert not set(b["features"]) & set(SENSITIVE)


@pytest.mark.parametrize("version", ["A", "B"])
def test_shap_groups_cover_every_feature_and_add_up(version, data):
    _, test = data
    b = joblib.load(MODELS_DIR / f"version_{version}_reduced.joblib")
    X = test.iloc[:10]
    contrib, base = shap_contributions(b, X)
    assert set(contrib.columns) == set(b["features"])
    proba = b["pipeline"].predict_proba(X[b["features"]])[:, 1]
    total = contrib.sum(axis=1).to_numpy() + base
    if b["model_name"] == "Random Forest":       # SHAP in probability space
        np.testing.assert_allclose(total, proba, atol=1e-6)
    else:                                         # SHAP in log-odds space
        np.testing.assert_allclose(1 / (1 + np.exp(-total)), proba, atol=1e-6)


def test_explain_student_returns_positive_reasons(data):
    _, test = data
    b = joblib.load(MODELS_DIR / "version_A_reduced.joblib")
    reasons = explain_student(b, test.iloc[[0]], top_k=3)
    assert len(reasons) <= 3
    assert all(r["contribution"] > 0 for r in reasons)
    contribs = [r["contribution"] for r in reasons]
    assert contribs == sorted(contribs, reverse=True)


def test_group_metrics_counts():
    df = pd.DataFrame({"at_risk": [1, 1, 0, 0, 1, 0], "g": ["a", "a", "a", "b", "b", "b"]})
    proba = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.9])
    t = group_metrics(df, proba, 0.5, "g", n_boot=10).set_index("group")
    assert t.loc["a", "recall"] == 0.5          # 1 of 2 at-risk flagged
    assert t.loc["a", "fpr"] == 1.0             # the 1 fine student flagged
    assert t.loc["b", "recall"] == 1.0
    assert t.loc["b", "fpr"] == 0.5
