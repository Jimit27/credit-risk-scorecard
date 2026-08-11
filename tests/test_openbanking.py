"""Tests for the Open Banking cashflow module.

The ledger is simulated, so these tests check that the *engineering* is right -
that the derived features mean what they say - rather than asserting anything
about predictive power, which the simulation cannot establish.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from creditrisk.openbanking import (
    DISCRETIONARY_CATEGORIES,
    ESSENTIAL_CATEGORIES,
    LedgerSpec,
    add_behavioural_features,
    cashflow_features,
    generate_ledger,
)


@pytest.fixture(scope="module")
def small_ledger() -> pd.DataFrame:
    return generate_ledger(LedgerSpec(n_applicants=60, months=6, seed=11))


def test_ledger_generation_is_deterministic():
    first = generate_ledger(LedgerSpec(n_applicants=20, months=3, seed=99))
    second = generate_ledger(LedgerSpec(n_applicants=20, months=3, seed=99))
    pd.testing.assert_frame_equal(first, second)


def test_ledger_has_income_and_spending_for_every_applicant(small_ledger):
    by_applicant = small_ledger.groupby("applicant_id")["true_category"].apply(set)
    assert all("income" in categories for categories in by_applicant)
    assert all(categories & ESSENTIAL_CATEGORIES for categories in by_applicant)


def test_income_is_credit_and_spending_is_debit(small_ledger):
    assert (small_ledger.loc[small_ledger["true_category"] == "income", "amount"] > 0).all()
    spending = small_ledger[small_ledger["true_category"].isin(ESSENTIAL_CATEGORIES | DISCRETIONARY_CATEGORIES)]
    assert (spending["amount"] < 0).all()


def test_behavioural_features_flag_a_regular_committed_outgoing(small_ledger):
    """Rent recurs monthly at a stable amount; eating out does not."""
    frame = add_behavioural_features(small_ledger)
    housing = frame[frame["true_category"] == "housing"]["merchant_is_monthly"].mean()
    eating_out = frame[frame["true_category"] == "eating_out"]["merchant_is_monthly"].mean()
    assert housing > eating_out


def test_cashflow_features_cover_every_applicant_with_no_stray_nulls(small_ledger):
    frame = small_ledger.rename(columns={"true_category": "category"})
    features = cashflow_features(frame, months=6)

    assert len(features) == small_ledger["applicant_id"].nunique()
    # Counts and amounts are zero when nothing happened, never unknown.
    for column in ("ob_fees_monthly", "ob_cash_monthly", "ob_debt_repayment_monthly", "ob_fee_events"):
        assert features[column].notna().all(), f"{column} leaked nulls where zero is the right answer"


def test_surplus_after_essentials_is_income_minus_essentials(small_ledger):
    frame = small_ledger.rename(columns={"true_category": "category"})
    features = cashflow_features(frame, months=6)
    expected = features["ob_income_monthly_median"] - features["ob_essential_spend_monthly"]
    assert np.allclose(features["ob_surplus_after_essentials"], expected, equal_nan=True)


def test_income_volatility_separates_regular_from_irregular_earners(small_ledger):
    frame = small_ledger.rename(columns={"true_category": "category"})
    features = cashflow_features(frame, months=6)
    regular = features[features["ob_income_is_regular"] == 1]["ob_income_volatility"]
    irregular = features[features["ob_income_is_regular"] == 0]["ob_income_volatility"]
    assert regular.max() < irregular.min()


def test_overdraft_days_are_consistent_with_the_balance_series(small_ledger):
    frame = small_ledger.rename(columns={"true_category": "category"})
    features = cashflow_features(frame, months=6).set_index("applicant_id")
    ever_negative = frame[frame["balance_after"] < 0]["applicant_id"].unique()
    assert (features.loc[ever_negative, "ob_days_in_overdraft"] > 0).all()
    never_negative = features.index.difference(ever_negative)
    assert (features.loc[never_negative, "ob_days_in_overdraft"] == 0).all()
