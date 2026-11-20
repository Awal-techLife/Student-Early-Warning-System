"""Tests for the app's scoring logic (src/predict.py)."""

import pandas as pd
import pytest

from src.data import PROJECT_ROOT
from src.predict import BANDS, HIDDEN_REASONS, band, load_bundle, score, validate_input

SAMPLE = PROJECT_ROOT / "data" / "sample_class.csv"


@pytest.fixture(scope="module")
def sample():
    return pd.read_csv(SAMPLE)


@pytest.mark.parametrize("version", ["A", "B"])
def test_sample_class_scores_cleanly(sample, version):
    b = load_bundle(version)
    X, errors = validate_input(sample, b)
    assert errors == []
    res = score(X, b)
    assert len(res) == len(sample)
    assert set(res.risk_band) <= set(BANDS)
    assert res.risk_score.between(0, 1).all()


@pytest.mark.parametrize("version", ["A", "B"])
def test_reasons_never_show_school_and_low_band_has_none(sample, version):
    b = load_bundle(version)
    X, _ = validate_input(sample, b)
    res = score(X, b)
    for _, r in res.iterrows():
        assert not {x["feature"] for x in r.reasons} & HIDDEN_REASONS
        if r.risk_band == "Low":
            assert r.reasons == [] and r.top_reasons == ""


def test_band_boundaries():
    b = {"threshold": 0.5, "watch_threshold": 0.2}
    assert band(0.5, b) == "High"
    assert band(0.49, b) == "Medium"
    assert band(0.2, b) == "Medium"
    assert band(0.19, b) == "Low"


def test_missing_column_is_reported(sample):
    b = load_bundle("B")
    _, errors = validate_input(sample.drop(columns=["G1"]), b)
    assert errors and "G1" in errors[0]


def test_bad_values_are_reported_with_row_numbers(sample):
    b = load_bundle("A")
    bad = sample.copy()
    bad.loc[2, "failures"] = 9          # out of range
    bad.loc[4, "higher"] = "maybe"      # not an allowed category
    _, errors = validate_input(bad, b)
    assert any("failures" in e and "row(s) 3" in e for e in errors)
    assert any("higher" in e and "row(s) 5" in e for e in errors)


def test_whitespace_and_text_numbers_are_accepted(sample):
    b = load_bundle("A")
    messy = sample.copy().astype(str)
    messy["higher"] = " " + messy["higher"] + " "
    X, errors = validate_input(messy, b)
    assert errors == []
    assert X["failures"].dtype.kind == "i"


def test_high_risk_profile_is_flagged():
    b = load_bundle("B")
    student = pd.read_csv(SAMPLE).iloc[[0]].copy()
    student["G1"] = 5
    student["failures"] = 2
    X, _ = validate_input(student, b)
    assert score(X, b).risk_band.iloc[0] == "High"
