"""Student Early-Warning System: Streamlit app.

Run locally:   streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.explain import label, shap_contributions  # noqa: E402
from src.predict import (  # noqa: E402
    HIDDEN_REASONS, NO_SINGLE_REASON, load_bundle, pretty, score, valid_values, validate_input,
)
from src.data import load_clean  # noqa: E402

FIG_DIR = ROOT / "reports" / "figures"
SAMPLE_CSV = ROOT / "data" / "sample_class.csv"
RISK_C, PASS_C, INK_2 = "#eb6834", "#2a78d6", "#52514e"
BAND_STYLE = {"High": ("#fde3d8", "#9a3412"), "Medium": ("#fdf0cf", "#854d0e"), "Low": ("#e8eef8", "#1e3a8a")}
BAND_TEXT = {
    "High": "Flag for support. Students in this band include most of those who go on to fail.",
    "Medium": "Watch list. Lower risk, but worth an informal check-in.",
    "Low": "No warning signs from the information provided.",
}
VERSIONS = {
    "A": "Start of year (no grades)",
    "B": "After first test (uses G1)",
}

st.set_page_config(page_title="Student Early-Warning", page_icon="🎓", layout="wide")


# ----------------------------------------------------------------------------- data
@st.cache_resource
def get_bundle(version: str) -> dict:
    return load_bundle(version)


@st.cache_data
def defaults() -> dict:
    """Typical value per column (median / most common), used to pre-fill the form."""
    df = load_clean("por")
    return {c: (int(df[c].median()) if df[c].dtype.kind in "iuf" else df[c].mode()[0])
            for c in df.columns}


SECTIONS = {
    "School & background": ["school", "age", "address", "famsize", "Pstatus", "guardian",
                            "reason", "nursery", "traveltime"],
    "Family": ["Medu", "Fedu", "Mjob", "Fjob", "famrel", "famsup"],
    "Study & support": ["studytime", "failures", "schoolsup", "paid", "activities", "higher", "internet"],
    "Free time & wellbeing": ["freetime", "goout", "health"],
    "First test": ["G1"],
}
HELP = {
    "failures": "Number of past class failures (3 means 3 or more).",
    "G1": "First-period grade, 0–20. Pass mark is 10.",
    "higher": "Does the student want to go on to higher education?",
    "schoolsup": "Is the student already receiving extra educational support from the school?",
    "famsup": "Does the family support the student's studies?",
    "paid": "Is the student taking paid extra classes in this subject?",
}


def widget(col: str, rule, default):
    name = label(col)
    if col == "G1":
        return st.slider(name, 0, 20, 10, help=HELP.get(col))
    if col == "age":
        return st.number_input(name, min_value=rule[0], max_value=rule[1], value=default, step=1)
    options = list(range(rule[0], rule[1] + 1)) if isinstance(rule, tuple) else rule
    index = options.index(default) if default in options else 0
    return st.selectbox(name, options, index=index, format_func=lambda v: pretty(col, v),
                        help=HELP.get(col))


def band_badge(band: str, score_value: float, bundle: dict) -> None:
    bg, fg = BAND_STYLE[band]
    st.markdown(
        f"<div style='background:{bg};color:{fg};padding:1rem 1.2rem;border-radius:10px'>"
        f"<div style='font-size:0.85rem;letter-spacing:.04em;text-transform:uppercase'>Risk band</div>"
        f"<div style='font-size:2rem;font-weight:700;line-height:1.2'>{band}</div>"
        f"<div style='margin-top:.3rem'>{BAND_TEXT[band]}</div></div>",
        unsafe_allow_html=True,
    )
    st.caption(f"Risk score {score_value:.2f}. Flag threshold {bundle['threshold']:.2f}, "
               f"watch threshold {bundle['watch_threshold']:.2f}. "
               "Scores rank students; they are not probabilities.")


def contribution_chart(bundle: dict, X: pd.DataFrame):
    contrib, _ = shap_contributions(bundle, X)
    row = contrib.iloc[0].drop(labels=[c for c in HIDDEN_REASONS if c in contrib.columns])
    top = row.reindex(row.abs().sort_values(ascending=False).index).head(6)[::-1]
    fig, ax = plt.subplots(figsize=(6, 2.8))
    ax.barh([f"{label(f)}: {pretty(f, X.iloc[0][f])}" for f in top.index], top.values,
            color=[RISK_C if v > 0 else PASS_C for v in top.values], height=0.6)
    ax.axvline(0, color=INK_2, lw=0.8)
    ax.set_xlabel("←  lowers risk   |   raises risk  →", color=INK_2, fontsize=9)
    ax.tick_params(colors=INK_2, labelsize=9)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.set_xticks([])
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🎓 Early-Warning")
    page = st.radio("Page", ["Predict", "Batch upload", "Insights", "About"], label_visibility="collapsed")
    st.divider()
    version = st.radio("Model version", list(VERSIONS), format_func=VERSIONS.get,
                       help="Version A works on day 1 using background and survey answers only. "
                            "Version B also uses the first-period grade (G1) and is more accurate.")
    bundle = get_bundle(version)
    tm = bundle["test_metrics"]
    st.caption(f"Test set: catches **{tm['test_recall']:.0%}** of at-risk students; "
               f"**{tm['test_precision']:.0%}** of flags are correct.")
    st.divider()
    st.caption("A prompt for a supportive conversation, **not** a judgement about a student. "
               "Never use it for grading, streaming or discipline.")


# ----------------------------------------------------------------------------- pages
if page == "Predict":
    st.header("Check one student")
    st.write(f"**{VERSIONS[version]}.** Fill in what you know; fields start at the most common answer.")
    allowed, dflt = valid_values(), defaults()
    feats = set(bundle["features"])

    with st.form("student"):
        values = {}
        sections = [(t, [c for c in cols if c in feats]) for t, cols in SECTIONS.items()]
        sections = [(t, cols) for t, cols in sections if cols]
        for title, cols in sections:
            st.subheader(title)
            grid = st.columns(3)
            for i, col in enumerate(cols):
                with grid[i % 3]:
                    values[col] = widget(col, allowed[col], dflt[col])
        submitted = st.form_submit_button("Assess risk", type="primary")

    if submitted:
        X, errors = validate_input(pd.DataFrame([values]), bundle)
        if errors:
            st.error("\n".join(errors))
        else:
            r = score(X, bundle).iloc[0]
            left, right = st.columns([2, 3])
            with left:
                band_badge(r.risk_band, r.risk_score, bundle)
            with right:
                if r.risk_band == "Low":
                    st.success("No action needed beyond normal teaching.")
                else:
                    st.markdown("**Main reasons**")
                    if r.reasons:
                        for reason in r.reasons:
                            st.markdown(f"- {reason['label']}: **{reason['value']}**")
                    else:
                        st.markdown(f"- {NO_SINGLE_REASON}")
                    st.markdown("**Suggested next steps**")
                    for a in r.suggested_actions:
                        st.markdown(f"- {a}")
            with st.expander("How each answer moved the score"):
                st.pyplot(contribution_chart(bundle, X), use_container_width=False)
                st.caption("SHAP contributions. School is part of the score but not listed as a reason.")

elif page == "Batch upload":
    st.header("Check a whole class")
    st.write("Upload a CSV with one row per student. The app scores everyone and sorts the list by risk.")
    c1, c2 = st.columns([3, 2])
    with c1:
        upload = st.file_uploader("Class CSV (comma- or semicolon-separated)", type=["csv"])
    with c2:
        st.download_button("Download a sample class (25 students)", SAMPLE_CSV.read_bytes(),
                           "sample_class.csv", "text/csv")
        use_sample = st.checkbox("Or try it with the sample class")

    with st.expander("Which columns does the file need?"):
        allowed = valid_values()
        st.write("Optional: a `student_id` or `name` column. Extra columns are ignored. Required for this version:")
        st.dataframe(pd.DataFrame({
            "column": bundle["features"],
            "meaning": [label(c) for c in bundle["features"]],
            "allowed values": [f"{r[0]}–{r[1]}" if isinstance(r, tuple) else ", ".join(r)
                               for r in (allowed[c] for c in bundle["features"])],
        }), hide_index=True, use_container_width=True)

    source = upload if upload is not None else (SAMPLE_CSV if use_sample else None)
    if source is not None:
        try:
            raw = pd.read_csv(source, sep=None, engine="python")
        except Exception as exc:  # noqa: BLE001 (show any parsing problem to the user)
            st.error(f"Could not read the file: {exc}")
            st.stop()
        X, errors = validate_input(raw, bundle)
        if errors:
            st.error("Please fix the file and upload it again:\n\n" + "\n".join(f"- {e}" for e in errors))
            st.stop()

        id_col = next((c for c in raw.columns if c.lower() in {"student_id", "id", "name", "student"}), None)
        ids = raw[id_col].astype(str) if id_col else pd.Series(range(1, len(raw) + 1), name="row").astype(str)
        with st.spinner("Scoring students..."):
            res = score(X, bundle)
        out = pd.DataFrame({
            "Student": ids.to_numpy(), "Risk band": res.risk_band, "Risk score": res.risk_score,
            "Main reasons": res.top_reasons,
            "Suggested first step": [a[0] if a else "" for a in res.suggested_actions],
        })
        order = {"High": 0, "Medium": 1, "Low": 2}
        out = out.sort_values(["Risk band", "Risk score"], key=lambda s: s.map(order) if s.name == "Risk band" else -s)

        m = out["Risk band"].value_counts()
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Students", len(out))
        k2.metric("High", int(m.get("High", 0)))
        k3.metric("Medium", int(m.get("Medium", 0)))
        k4.metric("Low", int(m.get("Low", 0)))

        styled = out.style.map(lambda b: f"background-color:{BAND_STYLE[b][0]};color:{BAND_STYLE[b][1]};"
                                         "font-weight:600", subset=["Risk band"])
        st.dataframe(styled, hide_index=True, use_container_width=True, column_config={
            "Risk score": st.column_config.ProgressColumn("Risk score", min_value=0.0, max_value=1.0, format="%.2f"),
            "Main reasons": st.column_config.TextColumn(width="large"),
            "Suggested first step": st.column_config.TextColumn(width="large"),
        })
        st.download_button("Download results (CSV)", out.to_csv(index=False).encode("utf-8"),
                           "class_risk_report.csv", "text/csv", type="primary")
        st.caption("Expect false alarms: in testing, about "
                   f"{1 - bundle['test_metrics']['test_precision']:.0%} of High-band students passed anyway. "
                   "Use the list to decide who to talk to first, not who is 'going to fail'.")

elif page == "Insights":
    st.header("What the data and models show")
    a, b = get_bundle("A")["test_metrics"], get_bundle("B")["test_metrics"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Version A recall", f"{a['test_recall']:.0%}", help="Share of at-risk test students flagged High")
    c2.metric("Version A precision", f"{a['test_precision']:.0%}", help="Share of High flags that were correct")
    c3.metric("Version B recall", f"{b['test_recall']:.0%}")
    c4.metric("Version B precision", f"{b['test_precision']:.0%}")
    st.caption("Held-out test set: 130 students, 20 at risk (one student = 5 percentage points of recall).")

    figures = [
        ("01_target_distribution.png", "1 in 6 students fails Portuguese; most failing students score 7–9, just under the pass mark."),
        ("02_failures.png", "Past failures are the strongest single sign, but most at-risk students have none."),
        ("03_school_address.png", "The gap is between the two schools, not between urban and rural homes."),
        ("07_precision_recall.png", "Version A finds far more at-risk students than the simple rule. In Version B, "
                                    "the rule 'G1 < 10' is about as good as the model."),
        ("08_feature_importance.png", "What each model relies on: failures and ambition (A); the first grade (B)."),
        ("11_fairness.png", "Fairness audit: similar results for boys/girls and urban/rural; "
                            "a real gap between schools in Version A."),
    ]
    for i in range(0, len(figures), 2):
        cols = st.columns(2)
        for col, (fname, caption) in zip(cols, figures[i:i + 2]):
            with col:
                st.image(str(FIG_DIR / fname), use_container_width=True)
                st.caption(caption)

    st.subheader("Model comparison")
    res = pd.read_csv(ROOT / "reports" / "results.csv")
    res = res[res.version.isin(["A", "B"])].assign(Version=lambda d: d.version.map(VERSIONS))
    st.dataframe(res[["Version", "model", "cv_roc_auc", "test_recall", "test_precision", "test_f1", "test_roc_auc"]]
                 .rename(columns={"model": "Model", "cv_roc_auc": "CV ROC-AUC", "test_recall": "Test recall",
                                  "test_precision": "Test precision", "test_f1": "Test F1",
                                  "test_roc_auc": "Test ROC-AUC"}).round(2),
                 hide_index=True, use_container_width=True)
    st.caption("All-features models from the comparison stage. The app uses the selected model per version, "
               "retrained without sex, alcohol use and relationship status (no loss in accuracy).")

else:  # About
    st.header("About this tool")
    st.markdown(f"""
**What it does.** Estimates which secondary-school students are at risk of failing Portuguese
(final grade below 10/20) so a teacher can offer support **early**, and shows *why* each student is flagged.

**Two versions**
- **A: start of year.** Uses only background and survey answers (no grades, no attendance), so it works on day 1.
  Catches about {get_bundle('A')['test_metrics']['test_recall']:.0%} of at-risk students, with roughly 2 false alarms per real case.
- **B: after the first test.** Adds the first-period grade (G1). More accurate, but a simple rule
  ("flag anyone with G1 below 10") performs about as well. Its extra value is ranking and explanations.

**What it does not use.** Sex, alcohol use and relationship status were tested and **removed**: they did not improve
accuracy and a school should not need them. Later grades (G2, final grade) are never used.

**Known limitations**
- **Small dataset:** 649 students from **two schools in Portugal** (2005–06). It may not transfer to other countries,
  curricula or years without retraining on local data.
- **School gap:** at the lower-risk school (GP) Version A catches about 69% of at-risk students vs 90% at MS,
  and it flags most students at MS. Per-school thresholds would be the fix.
- **Scores are not probabilities.** They rank students. The bands are set from cross-validated results.
- **Correlation, not causation.** "Low study time" is associated with failing. That doesn't prove more study time alone would prevent it.

**How it should be used:** as a prompt for a supportive conversation. Never for grading, streaming, discipline
or any automated decision. A teacher's knowledge of the student always comes first.

**Data:** P. Cortez and A. Silva. *Using Data Mining to Predict Secondary School Student Performance.*
FUBUTEC 2008, pp. 5–12. UCI Machine Learning Repository, dataset 320 (CC BY 4.0).

**Code:** Python, scikit-learn, SHAP, Streamlit. Built by Awal Abdullah N-don.
""")
