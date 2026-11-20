"""Basic tests for src/data.py. Run with: python -m pytest"""

import pandas as pd
import pytest

from src.data import (
    EXPECTED_COLUMNS,
    LEAKAGE_COLS,
    PASS_MARK,
    add_target,
    load_clean,
    validate,
)


@pytest.fixture(scope="module")
def por():
    return load_clean("por")


def test_shape_and_columns(por):
    assert len(por) == 649
    assert set(EXPECTED_COLUMNS) <= set(por.columns)


def test_target_matches_pass_mark(por):
    assert ((por["G3"] < PASS_MARK) == por["at_risk"].astype(bool)).all()
    assert por["at_risk"].isin([0, 1]).all()


def test_zero_grade_students_kept_and_flagged(por):
    flagged = por[por["no_final_exam"] == 1]
    assert len(flagged) == 15
    assert (flagged["G3"] == 0).all()
    assert (flagged["at_risk"] == 1).all()


def test_leakage_columns_present_so_they_can_be_excluded(por):
    # features.py removes these; this guards against a rename silently breaking that.
    assert set(LEAKAGE_COLS) <= set(por.columns)


def test_validate_rejects_out_of_range_grade():
    bad = pd.DataFrame([{c: 0 for c in EXPECTED_COLUMNS}])
    bad["G3"] = 25
    with pytest.raises(ValueError, match="G3"):
        validate(bad)


def test_add_target_does_not_modify_input():
    df = pd.DataFrame({"G3": [5, 10, 15]})
    out = add_target(df)
    assert "at_risk" not in df.columns
    assert out["at_risk"].tolist() == [1, 0, 0]
