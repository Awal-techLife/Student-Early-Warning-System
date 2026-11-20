"""Leakage and consistency tests for feature sets and the training helpers."""

import numpy as np
import pytest

from src.data import LEAKAGE_COLS, load_clean
from src.features import FEATURE_SETS, all_features, make_preprocessor
from src.train import TARGET_RECALL, choose_threshold, get_split


@pytest.mark.parametrize("version", list(FEATURE_SETS))
def test_no_leakage_columns(version):
    assert not set(all_features(version)) & set(LEAKAGE_COLS)


def test_grades_only_in_version_b():
    assert "G1" not in all_features("A")
    assert "G1" not in all_features("A_absences")
    assert "G1" in all_features("B")


def test_absences_not_in_start_of_year_model():
    assert "absences" not in all_features("A")


def test_drop_removes_columns():
    feats = all_features("A", drop=["sex", "Dalc"])
    assert "sex" not in feats and "Dalc" not in feats


def test_preprocessor_runs_on_real_data():
    df = load_clean("por")
    X = make_preprocessor("B").fit_transform(df[all_features("B")])
    assert X.shape[0] == len(df)
    assert not np.isnan(X).any()


def test_split_is_stratified_and_reproducible():
    df = load_clean("por")
    tr1, te1 = get_split(df)
    tr2, te2 = get_split(df)
    assert te1.index.equals(te2.index)
    assert abs(te1.at_risk.mean() - df.at_risk.mean()) < 0.01


def test_choose_threshold_meets_recall_target():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    proba = np.clip(y * 0.3 + rng.uniform(0, 0.7, 300), 0, 1)
    thr = choose_threshold(y, proba)
    recall = ((proba >= thr) & (y == 1)).sum() / y.sum()
    assert recall >= TARGET_RECALL
