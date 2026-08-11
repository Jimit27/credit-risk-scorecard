"""Interactive scoring app for the credit-risk scorecard.

Run locally:      streamlit run app/streamlit_app.py
Deploy free on:   share.streamlit.io (point it at this file)

The app loads the persisted model bundle and nothing else - the same object the
batch pipeline produced, with the same WoE bins and the same calibrator. There
is no second copy of the preprocessing living in the UI.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from creditrisk.config import load_config  # noqa: E402
from creditrisk.explain import explain_applicant, global_importance, points_table, reason_codes  # noqa: E402
from creditrisk.model import ScorecardModel  # noqa: E402
from creditrisk.serving import EXAMPLE_APPLICANTS, derive_features  # noqa: E402

st.set_page_config(page_title="Credit Risk Scorecard", page_icon="•", layout="wide")

BAND_COLOUR = {"A": "#0ca30c", "B": "#1baf7a", "C": "#eda100", "D": "#ec835a", "E": "#d03b3b"}


@st.cache_resource
def load_artifacts():
    cfg = load_config()
    model = ScorecardModel.load(cfg.path("paths.models") / "scorecard.joblib")
    metrics_path = cfg.path("paths.reports") / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    return cfg, model, metrics


def sidebar_inputs() -> dict[str, float]:
    st.sidebar.header("Applicant")
    preset = st.sidebar.selectbox("Start from a profile", list(EXAMPLE_APPLICANTS), index=1)
    base = EXAMPLE_APPLICANTS[preset]

    st.sidebar.caption("Adjust any field to see the decision move.")
    values = {
        "age": st.sidebar.slider("Age", 18, 95, int(base["age"])),
        "monthly_income": st.sidebar.slider("Monthly income (£)", 500, 15000, int(base["monthly_income"]), step=50),
        "revolving_utilisation": st.sidebar.slider(
            "Revolving credit used", 0.0, 1.5, float(base["revolving_utilisation"]), step=0.01,
            help="Balance as a share of available revolving limit.",
        ),
        "debt_ratio": st.sidebar.slider(
            "Debt-to-income ratio", 0.0, 2.0, float(base["debt_ratio"]), step=0.01,
            help="Monthly debt repayments divided by monthly income.",
        ),
        "open_credit_lines": st.sidebar.slider("Open credit lines", 0, 30, int(base["open_credit_lines"])),
        "real_estate_lines": st.sidebar.slider("Property-secured lines", 0, 10, int(base["real_estate_lines"])),
        "dependents": st.sidebar.slider("Dependants", 0, 8, int(base["dependents"])),
        "times_30_59_dpd": st.sidebar.slider("Times 30-59 days late (2 yrs)", 0, 10, int(base["times_30_59_dpd"])),
        "times_60_89_dpd": st.sidebar.slider("Times 60-89 days late (2 yrs)", 0, 10, int(base["times_60_89_dpd"])),
        "times_90_dpd": st.sidebar.slider("Times 90+ days late (2 yrs)", 0, 10, int(base["times_90_dpd"])),
    }
    return values


def decision_panel(model: ScorecardModel, applicant: pd.DataFrame) -> None:
    decision = model.decide(applicant[model.features]).iloc[0]
    band = str(decision["band"])
    label = {b["name"]: b.get("label", b["name"]) for b in model.bands}.get(band, band)

    left, middle, right = st.columns([1.1, 1, 1])
    left.metric("Scorecard points", f"{int(decision['score'])}")
    middle.metric("Probability of default", f"{decision['probability_of_default'] * 100:.2f}%")
    right.markdown(
        f"<div style='padding-top:6px'><span style='font-size:13px;color:#52514e'>Policy grade</span><br>"
        f"<span style='font-size:34px;font-weight:600;color:{BAND_COLOUR.get(band, '#0b0b0b')}'>{band}</span>"
        f"<span style='font-size:14px;color:#52514e;margin-left:10px'>{label}</span></div>",
        unsafe_allow_html=True,
    )

    st.divider()
    st.subheader("Why this score")
    st.caption(
        "Adverse-action reasons, ranked by distance-to-maximum: the points this applicant gave up "
        "on each factor compared with the best-scoring band for that factor."
    )
    codes = reason_codes(model, applicant.iloc[0], top_n=5)
    if not codes:
        st.success("This applicant sits in the strongest band for every factor in the scorecard.")
    else:
        st.dataframe(
            pd.DataFrame(codes)
            .assign(applicant_value=lambda d: d["applicant_value"].map(lambda v: "not reported" if pd.isna(v) else f"{v:g}"))
            .rename(
                columns={
                    "rank": "#",
                    "reason": "Factor",
                    "applicant_value": "Applicant",
                    "bin": "Band",
                    "points_lost": "Points lost",
                    "bin_default_rate": "Default rate in band",
                }
            )[["#", "Factor", "Applicant", "Band", "Points lost", "Default rate in band"]],
            hide_index=True,
            use_container_width=True,
        )

    with st.expander("Full points breakdown for this applicant"):
        breakdown = explain_applicant(model, applicant.iloc[0])
        st.bar_chart(breakdown.set_index("description")["points"], color="#2a78d6", horizontal=True)
        st.dataframe(breakdown, hide_index=True, use_container_width=True)


def performance_tab(metrics: dict, cfg) -> None:
    if not metrics:
        st.info("Run `python -m creditrisk.pipeline all` to generate the metrics report.")
        return

    champion = metrics["champion"]
    test = metrics["models"][champion]["test"]
    columns = st.columns(4)
    columns[0].metric("Gini", f"{test['gini']:.3f}")
    columns[1].metric("KS", f"{test['ks']:.3f}")
    columns[2].metric("Brier", f"{test['brier']:.4f}")
    columns[3].metric("Test applicants", f"{test['n']:,}")

    st.caption(
        f"Champion: {champion}. Held-out sample, scored once. "
        f"Predicted default rate {test['mean_predicted_rate'] * 100:.2f}% against "
        f"{test['observed_default_rate'] * 100:.2f}% observed."
    )

    figures = cfg.path("paths.figures")
    pairs = [
        ("score_distribution.png", "Score separation"),
        ("calibration.png", "Calibration"),
        ("band_default_rates.png", "Risk by policy grade"),
        ("approval_profit.png", "Approval rate against book profit"),
    ]
    for index in range(0, len(pairs), 2):
        row = st.columns(2)
        for column, (filename, caption) in zip(row, pairs[index : index + 2], strict=False):
            path = figures / filename
            if path.exists():
                column.image(str(path), caption=caption, use_container_width=True)


def scorecard_tab(model: ScorecardModel) -> None:
    st.caption(
        "The entire model, printed. Every applicant's score is the sum of one row per factor, "
        "which is what makes a scorecard defensible to a credit committee."
    )
    st.dataframe(global_importance(model), hide_index=True, use_container_width=True)
    st.subheader("Points by band")
    st.dataframe(
        points_table(model)[["feature", "bin", "count", "bad_rate", "woe", "points"]],
        hide_index=True,
        use_container_width=True,
        height=420,
    )


def openbanking_tab(cfg) -> None:
    summary_path = cfg.path("paths.reports") / "openbanking_summary.json"
    ledger_path = cfg.path("paths.gold") / "openbanking_ledger_sample.csv"

    st.warning(
        "The transaction ledger behind this tab is **simulated**. No public Open Banking dataset "
        "links real current-account transactions to real default outcomes. Nothing here feeds the "
        "scorecard, and no predictive claim is made from it - it demonstrates the categorisation "
        "and affordability engineering, not a result."
    )
    if not summary_path.exists():
        st.info("Run `python -m creditrisk.pipeline openbanking` to generate this section.")
        return

    summary = json.loads(summary_path.read_text())
    columns = st.columns(4)
    columns[0].metric("Applicants", f"{summary['applicants']:,}")
    columns[1].metric("Transactions", f"{summary['transactions']:,}")
    columns[2].metric(
        "Unseen-merchant accuracy",
        f"{summary['unseen_merchant_accuracy_text_plus_behaviour'] * 100:.0f}%",
        delta=f"{(summary['unseen_merchant_accuracy_text_plus_behaviour'] - summary['unseen_merchant_accuracy_text_only']) * 100:+.0f} pts vs text only",
    )
    columns[3].metric(
        "Majority-class baseline",
        f"{summary['majority_class_baseline_accuracy'] * 100:.0f}%",
        help="Always predicting the most common category in the held-out set. The honest bar to clear.",
    )

    st.caption(
        "On a merchant it has never seen, the text-only model lands *below* random guessing - nothing in "
        "'OCTOPUS ENERGY' resembles 'EDF ENERGY' at the character level. Adding transaction behaviour "
        "(amount, sign, recurrence, amount stability) takes it to roughly five times the baseline. Still "
        "hard, which is why commercial providers pair a curated merchant dictionary with a behavioural model."
    )

    if ledger_path.exists():
        st.subheader("Sample ledger")
        st.dataframe(pd.read_csv(ledger_path).head(60), hide_index=True, use_container_width=True)

    st.subheader("Derived affordability features")
    st.write(", ".join(f"`{name}`" for name in summary["features_derived"]))


def main() -> None:
    cfg, model, metrics = load_artifacts()

    st.title("Credit risk scorecard")
    st.caption(
        "A calibrated probability-of-default model with a published points table, "
        "adverse-action reason codes and a business cut-off. Built on the Give Me Some Credit "
        "dataset (150,000 real consumer credit files)."
    )

    values = sidebar_inputs()
    applicant = derive_features(values)

    score_tab, performance, scorecard, openbanking = st.tabs(
        ["Score an applicant", "Model performance", "The scorecard", "Open Banking (simulated)"]
    )
    with score_tab:
        decision_panel(model, applicant)
    with performance:
        performance_tab(metrics, cfg)
    with scorecard:
        scorecard_tab(model)
    with openbanking:
        openbanking_tab(cfg)


if __name__ == "__main__":
    main()
