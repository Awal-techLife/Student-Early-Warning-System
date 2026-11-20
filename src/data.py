"""Load, validate and clean the UCI Student Performance data.

Usage:
    python -m src.data            # prints a sanity-check summary
    from src.data import load_clean
    df = load_clean("por")        # cleaned DataFrame with an `at_risk` column
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

SUBJECT_FILES = {"por": "student-por.csv", "mat": "student-mat.csv"}

PASS_MARK = 10  # Portuguese grading: 0-20, pass at 10
GRADE_COLS = ["G1", "G2", "G3"]

# Columns that reveal the outcome. They must NEVER be model inputs.
# (G1 is allowed, but only in Version B. See src/features.py.)
LEAKAGE_COLS = ["G2", "G3", "at_risk", "no_final_exam"]

# Columns we expect, so a wrong or corrupted file fails loudly instead of silently.
EXPECTED_COLUMNS = [
    "school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu",
    "Mjob", "Fjob", "reason", "guardian", "traveltime", "studytime", "failures",
    "schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet",
    "romantic", "famrel", "freetime", "goout", "Dalc", "Walc", "health",
    "absences", "G1", "G2", "G3",
]


def load_raw(subject: str = "por") -> pd.DataFrame:
    """Read one subject's CSV exactly as published (semicolon-separated)."""
    if subject not in SUBJECT_FILES:
        raise ValueError(f"subject must be one of {list(SUBJECT_FILES)}, got {subject!r}")
    path = RAW_DIR / SUBJECT_FILES[subject]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the zip from "
            "https://archive.ics.uci.edu/dataset/320/student+performance into data/raw/."
        )
    return pd.read_csv(path, sep=";")


def validate(df: pd.DataFrame) -> None:
    """Raise if the data doesn't look like the dataset we expect."""
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if df.isna().any().any():
        raise ValueError("Unexpected missing values in raw data")
    for col in GRADE_COLS:
        if not df[col].between(0, 20).all():
            raise ValueError(f"{col} has values outside 0-20")


def handle_zero_final_grades(df: pd.DataFrame) -> pd.DataFrame:
    """Decide what to do with students whose final grade G3 is exactly 0.

    In student-por.csv there are 15 such students. All 15 have absences == 0,
    14 of the 15 are from school MS, and most had G1 between 4 and 11.
    A 0/20 with zero recorded absences most likely means the student did not
    sit the final exam (dropped out / left), not that they scored zero.
    """
    # Policy: KEEP these students and FLAG them.
    # A student who leaves before the final exam is exactly who an early-warning
    # system should catch, so dropping them would hide the most serious cases and
    # shrink the at-risk class by 15%. The flag lets later phases re-run results
    # without them (a robustness check). The flag is derived from G3, so it is
    # listed in LEAKAGE_COLS and must never be used as a model input.
    out = df.copy()
    out["no_final_exam"] = (out["G3"] == 0).astype(int)
    return out


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add the binary target: 1 if the student fails (G3 < 10), else 0."""
    out = df.copy()
    out["at_risk"] = (out["G3"] < PASS_MARK).astype(int)
    return out


def load_clean(subject: str = "por") -> pd.DataFrame:
    """Full pipeline: load -> validate -> handle G3 == 0 -> add target."""
    df = load_raw(subject)
    validate(df)
    df = handle_zero_final_grades(df)
    return add_target(df)


def summarise(df: pd.DataFrame) -> None:
    """Print a quick sanity-check summary."""
    n = len(df)
    n_risk = int(df["at_risk"].sum())
    print(f"Rows: {n}   Columns: {df.shape[1]}")
    print(f"At risk (G3 < {PASS_MARK}): {n_risk} ({n_risk / n:.1%})")
    n_flag = int(df["no_final_exam"].sum())
    print(f"Flagged no_final_exam (G3 == 0): {n_flag}  -> at risk without them: {n_risk - n_flag}")
    print("\nAt-risk rate by school:")
    print(df.groupby("school")["at_risk"].agg(["count", "mean"]).round(3).to_string())


if __name__ == "__main__":
    summarise(load_clean("por"))
