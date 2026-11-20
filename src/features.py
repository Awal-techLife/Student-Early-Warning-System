"""Feature sets for each model version, plus the shared preprocessing step.

Versions
--------
A           Start of year. Background, family and behaviour survey answers only.
            No grades and no absences: `absences` counts the WHOLE school year,
            so a teacher would not have it on day 1.
A_absences  Version A + `absences`, read as a "prior attendance" proxy.
            A sensitivity check: how much does that (leaky) column add?
B           After the first test. Version A + G1 (first-period grade).

G2, G3 and anything derived from them (see LEAKAGE_COLS) are never used.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import LEAKAGE_COLS

# Text columns (yes/no or a few categories) -> one-hot encoded.
CATEGORICAL = [
    "school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob", "reason",
    "guardian", "schoolsup", "famsup", "paid", "activities", "nursery",
    "higher", "internet", "romantic",
]

# Integer columns. Most are ordinal 1-5 or 0-4 scales; treating them as numbers
# keeps their order and works well for all three model types.
NUMERIC_BASE = [
    "age", "Medu", "Fedu", "traveltime", "studytime", "failures",
    "famrel", "freetime", "goout", "Dalc", "Walc", "health",
]

FEATURE_SETS: dict[str, dict[str, list[str]]] = {
    "A": {"categorical": CATEGORICAL, "numeric": NUMERIC_BASE},
    "A_absences": {"categorical": CATEGORICAL, "numeric": NUMERIC_BASE + ["absences"]},
    "B": {"categorical": CATEGORICAL, "numeric": NUMERIC_BASE + ["G1"]},
}

VERSION_LABELS = {
    "A": "A: start of year",
    "A_absences": "A + absences (proxy)",
    "B": "B: after first test",
}


def feature_columns(version: str, drop: list[str] | None = None) -> dict[str, list[str]]:
    """Return {'categorical': [...], 'numeric': [...]} for a version.

    `drop` removes columns (used in Phase 4 to test without sensitive features).
    """
    if version not in FEATURE_SETS:
        raise ValueError(f"version must be one of {list(FEATURE_SETS)}, got {version!r}")
    drop = set(drop or [])
    cols = {k: [c for c in v if c not in drop] for k, v in FEATURE_SETS[version].items()}
    leaked = set(cols["categorical"] + cols["numeric"]) & set(LEAKAGE_COLS)
    if leaked:  # defensive: should be impossible given the lists above
        raise ValueError(f"Leakage columns in feature set {version}: {sorted(leaked)}")
    return cols


def all_features(version: str, drop: list[str] | None = None) -> list[str]:
    cols = feature_columns(version, drop)
    return cols["categorical"] + cols["numeric"]


def make_preprocessor(version: str, drop: list[str] | None = None) -> ColumnTransformer:
    """One-hot encode categoricals, scale numerics. Fitted inside each CV fold."""
    cols = feature_columns(version, drop)
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore",
                                  sparse_output=False), cols["categorical"]),
            ("num", StandardScaler(), cols["numeric"]),
        ],
        verbose_feature_names_out=False,
    )
