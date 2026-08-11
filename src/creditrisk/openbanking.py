"""Open Banking cashflow features from a transaction ledger.

WHAT IS AND IS NOT REAL HERE
----------------------------
The transaction ledger this module consumes is **simulated**. There is no
public, licensable Open Banking dataset with real current-account transactions
attached to real default outcomes - the data is exactly the kind that cannot be
released - so one is generated from a documented process in
:func:`generate_ledger`.

Consequently **no predictive claim is made from these features**. They are not
merged into the champion scorecard and they contribute nothing to the headline
Gini reported in the README. What this module demonstrates is the engineering
an affordability model actually requires, which is the part that transfers:

1. Turning free-text merchant strings into a spending taxonomy - the messiest
   and most underrated step in any Open Banking product.
2. Deriving affordability signals from a transaction stream: income regularity,
   essential-spend cover, discretionary volatility, overdraft dependence.

The merchant classifier is evaluated on a **held-out set of merchants**, not a
random split of transactions, so it has to generalise from the shape of an
unseen string rather than recognise one it has already memorised.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Merchant taxonomy
# --------------------------------------------------------------------------
# Real bank feeds look like this: truncated, upper-case, store numbers welded
# on, inconsistent spacing. Any categoriser that assumes clean names fails on
# contact with a genuine feed.
MERCHANT_CATALOGUE: dict[str, list[str]] = {
    "groceries": [
        "TESCO STORES {n}", "SAINSBURYS SMKT {n}", "ASDA SUPERSTORE", "ALDI {n}",
        "LIDL GB {n}", "M&S SIMPLY FOOD", "CO-OP GROUP {n}", "WAITROSE {n}",
        "MORRISONS STORE {n}", "ICELAND FOODS",
    ],
    "housing": [
        "RENT PAYMENT REF{n}", "DD HOUSING ASSOC", "MORTGAGE PYMT {n}",
        "LETTINGS LTD RENT", "COUNCIL TAX {n} DD",
    ],
    "utilities": [
        "BRITISH GAS DD", "OCTOPUS ENERGY", "EDF ENERGY {n}", "THAMES WATER DD",
        "VIRGIN MEDIA {n}", "BT GROUP PLC DD", "EE LIMITED {n}", "VODAFONE UK",
    ],
    "transport": [
        "TFL TRAVEL CH", "TRAINLINE {n}", "UBER *TRIP {n}", "SHELL SERVICE STN",
        "BP EXPRESS {n}", "NCP CAR PARK", "GWR TICKET OFFICE",
    ],
    "subscriptions": [
        "NETFLIX.COM", "SPOTIFY UK {n}", "AMAZON PRIME {n}", "APPLE.COM/BILL",
        "DISNEY PLUS", "PURE GYM LTD {n}", "AUDIBLE UK",
    ],
    "eating_out": [
        "DELIVEROO {n}", "JUST EAT {n}", "PRET A MANGER {n}", "GREGGS PLC {n}",
        "NANDOS {n}", "COSTA COFFEE {n}", "UBER *EATS {n}", "WAGAMAMA {n}",
    ],
    "shopping": [
        "AMZNMktplace {n}", "ARGOS RETAIL {n}", "JD SPORTS {n}", "ASOS.COM",
        "PRIMARK STORES", "IKEA LTD {n}", "BOOTS THE CHEMIST {n}", "ZARA UK {n}",
    ],
    "cash": ["CASH WDL ATM {n}", "LINK ATM WDL {n}", "POST OFFICE CNTR"],
    "debt_repayment": [
        "KLARNA* PYMT {n}", "CLEARPAY {n}", "LOAN REPAYMT REF{n}",
        "CREDIT CARD PYMT", "PAYPAL CREDIT {n}",
    ],
    "fees": [
        "UNPAID DD FEE", "OVERDRAFT INT CHG", "RETURNED ITEM FEE", "ARRANGED OD FEE",
    ],
    "income": [
        "SALARY {n} BACS", "PAYROLL CREDIT {n}", "WAGES BACS {n}",
        "HMRC CREDIT {n}", "UC PAYMENT DWP",
    ],
    "transfers": ["TFR TO SAVINGS", "FASTER PYMT OUT {n}", "TFR FROM SAVINGS"],
}

# These are tuples, not sets, and that matters. Python randomises string
# hashing per process, so iterating a set of category names yields a different
# order in every run. RECURRING_CATEGORIES is iterated while drawing each
# applicant's fixed merchants, so a set made the whole simulation irreproducible
# - identical transaction counts, different contents, and reported accuracies
# that drifted by a point between runs of the same committed code.
#
# Spend categories a lender treats as non-negotiable when testing affordability.
ESSENTIAL_CATEGORIES = ("groceries", "housing", "transport", "utilities")
DISCRETIONARY_CATEGORIES = ("eating_out", "shopping", "subscriptions")

# Categories where a person deals with the *same* counterparty every month: one
# landlord, one energy supplier, one employer. Drawing a fresh merchant each
# month would destroy the recurrence structure that makes these transactions
# recognisable, which is precisely the signal the behavioural features use.
RECURRING_CATEGORIES = ("debt_repayment", "housing", "income", "subscriptions", "utilities")


@dataclass(frozen=True)
class LedgerSpec:
    """Parameters of the simulated ledger."""

    n_applicants: int = 3000
    months: int = 6
    seed: int = 7


def generate_ledger(spec: LedgerSpec) -> pd.DataFrame:
    """Simulate a current-account transaction ledger.

    The generative process, stated plainly so the output is not mistaken for
    real data: each applicant draws a log-normal monthly income, a pay rhythm
    (monthly or four-weekly), a housing cost as a share of income, and a latent
    "financial pressure" level that raises discretionary spending, cash usage
    and the chance of overdraft fees. Transaction dates and amounts are jittered
    around those parameters.
    """
    rng = np.random.default_rng(spec.seed)
    rows: list[dict] = []

    for applicant in range(spec.n_applicants):
        income = float(np.clip(rng.lognormal(mean=7.8, sigma=0.45), 900, 12000))
        four_weekly = rng.random() < 0.25
        pressure = float(np.clip(rng.beta(2, 5), 0, 1))  # 0 comfortable, 1 stretched
        housing_share = float(np.clip(rng.normal(0.32 + 0.15 * pressure, 0.08), 0.05, 0.75))
        irregular_income = rng.random() < (0.10 + 0.35 * pressure)

        # One landlord, one energy supplier, one employer - fixed for this
        # applicant, reference number and all.
        fixed_merchants = {
            category: _render(str(rng.choice(MERCHANT_CATALOGUE[category])), rng)
            for category in RECURRING_CATEGORIES
        }
        # A household holds a handful of subscriptions, each billed monthly at
        # the same amount, rather than a fresh random service every month.
        subscriptions = [
            (_render(str(template), rng), round(float(rng.choice([4.99, 7.99, 9.99, 12.99, 15.99, 24.99, 34.99])), 2))
            for template in rng.choice(MERCHANT_CATALOGUE["subscriptions"], size=int(rng.integers(1, 5)), replace=False)
        ]
        # A loan or card repayment is a fixed commitment, not a fresh draw.
        has_credit_commitment = rng.random() < 0.35 + 0.4 * pressure
        commitment_amount = round(income * float(rng.uniform(0.03, 0.18)), 2)

        day = 0
        pay_dates = []
        while day < spec.months * 30:
            pay_dates.append(day)
            day += 28 if four_weekly else 30

        balance = income * float(rng.uniform(0.05, 0.9)) * (1 - 0.7 * pressure)

        for pay_day in pay_dates:
            # --- income ---------------------------------------------------
            variation = rng.normal(1.0, 0.30 if irregular_income else 0.02)
            amount = round(max(income * variation, 50.0), 2)
            balance += amount
            rows.append(_row(applicant, pay_day, "income", amount, rng, balance, fixed_merchants))

            # --- essentials -----------------------------------------------
            for category, share in (
                ("housing", housing_share),
                ("utilities", 0.07),
                ("groceries", 0.14 + 0.03 * pressure),
                ("transport", 0.06),
            ):
                budget = income * share
                n_txn = 1 if category in {"housing", "utilities"} else int(rng.integers(4, 12))
                for _ in range(n_txn):
                    value = round(budget / n_txn * float(rng.uniform(0.7, 1.3)), 2)
                    balance -= value
                    offset = pay_day + int(rng.integers(0, 28))
                    rows.append(_row(applicant, offset, category, -value, rng, balance, fixed_merchants))

            # --- discretionary, scaled by financial pressure ---------------
            for category, base in (("eating_out", 0.05), ("shopping", 0.06)):
                budget = income * base * (1 + 0.8 * pressure)
                for _ in range(int(rng.integers(2, 10))):
                    value = round(budget / 6 * float(rng.uniform(0.4, 1.8)), 2)
                    balance -= value
                    rows.append(_row(applicant, pay_day + int(rng.integers(0, 28)), category, -value, rng, balance, fixed_merchants))

            # --- subscriptions: same services, same amounts, every month ----
            for merchant, value in subscriptions:
                balance -= value
                rows.append(
                    _row(applicant, pay_day + int(rng.integers(0, 28)), "subscriptions", -value, rng, balance,
                         {"subscriptions": merchant})
                )

            # --- cash, credit repayments, fees ------------------------------
            for _ in range(int(rng.poisson(0.5 + 3.0 * pressure))):
                value = round(float(rng.choice([20, 40, 50, 100, 200])), 2)
                balance -= value
                rows.append(_row(applicant, pay_day + int(rng.integers(0, 28)), "cash", -value, rng, balance, fixed_merchants))

            if has_credit_commitment:
                balance -= commitment_amount
                rows.append(
                    _row(applicant, pay_day + int(rng.integers(0, 28)), "debt_repayment", -commitment_amount,
                         rng, balance, fixed_merchants)
                )

            # Fees are the consequence of the simulated balance actually going
            # negative, not an independent coin flip - which is what makes the
            # derived overdraft features behave like real ones.
            if balance < 0 and rng.random() < 0.6:
                for _ in range(int(rng.integers(1, 3))):
                    value = round(float(rng.choice([8.0, 12.0, 15.0, 25.0])), 2)
                    balance -= value
                    rows.append(_row(applicant, pay_day + int(rng.integers(0, 28)), "fees", -value, rng, balance, fixed_merchants))

            if rng.random() < 0.4 * (1 - pressure):
                value = round(income * float(rng.uniform(0.02, 0.15)), 2)
                balance -= value
                rows.append(_row(applicant, pay_day + int(rng.integers(0, 28)), "transfers", -value, rng, balance, fixed_merchants))

    ledger = pd.DataFrame(rows).sort_values(["applicant_id", "day"]).reset_index(drop=True)
    LOGGER.info("Simulated %s transactions for %s applicants", len(ledger), spec.n_applicants)
    return ledger


def _render(template: str, rng: np.random.Generator) -> str:
    """Substitute the store/reference number that makes each descriptor unique."""
    return template.replace("{n}", str(int(rng.integers(100, 99999))))


def _row(
    applicant: int,
    day: int,
    category: str,
    amount: float,
    rng: np.random.Generator,
    balance: float,
    fixed_merchants: dict[str, str] | None = None,
) -> dict:
    fixed = (fixed_merchants or {}).get(category)
    description = fixed if fixed else _render(str(rng.choice(MERCHANT_CATALOGUE[category])), rng)
    return {
        "applicant_id": applicant,
        "day": int(day),
        "description": description,
        "amount": float(amount),
        "balance_after": round(float(balance), 2),
        "true_category": category,  # generator ground truth, for evaluation only
    }


# --------------------------------------------------------------------------
# Merchant categorisation
# --------------------------------------------------------------------------
BEHAVIOURAL_FEATURES = [
    "abs_amount_log",
    "is_credit",
    "day_of_month",
    "merchant_frequency",
    "merchant_amount_cv",
    "merchant_is_monthly",
    "share_of_month_outflow",
]


def add_behavioural_features(ledger: pd.DataFrame) -> pd.DataFrame:
    """Attach the transaction-shape features a categoriser needs to generalise.

    A merchant name the model has never seen tells it nothing. How the
    transaction *behaves* still does: rent is large, monthly and near-identical
    each time; a subscription is small, monthly and near-identical; groceries
    are frequent and variable. These features carry that structure.
    """
    frame = ledger.copy()
    frame["abs_amount_log"] = np.log1p(frame["amount"].abs())
    frame["is_credit"] = (frame["amount"] > 0).astype(float)
    frame["day_of_month"] = frame["day"] % 30
    frame["merchant_stem"] = frame["description"].str.replace(r"\d+", "", regex=True).str.strip()

    # Per applicant-merchant recurrence and amount stability.
    key = ["applicant_id", "merchant_stem"]
    grouped = frame.groupby(key)["amount"]
    counts = grouped.transform("size")
    mean_amount = grouped.transform("mean").abs()
    std_amount = grouped.transform("std").fillna(0.0)
    frame["merchant_frequency"] = counts
    frame["merchant_amount_cv"] = (std_amount / mean_amount.replace(0, np.nan)).fillna(0.0)
    # "Roughly once a month, in a stable amount" - the signature of a
    # committed outgoing rather than a discretionary one.
    frame["merchant_is_monthly"] = ((counts.between(4, 8)) & (frame["merchant_amount_cv"] < 0.25)).astype(float)

    month_outflow = (
        frame.assign(month=frame["day"] // 30, outflow=frame["amount"].where(frame["amount"] < 0, 0).abs())
        .groupby(["applicant_id", "month"])["outflow"]
        .transform("sum")
    )
    frame["share_of_month_outflow"] = (frame["amount"].abs() / month_outflow.replace(0, np.nan)).fillna(0.0)
    return frame


def train_merchant_classifier(ledger: pd.DataFrame, seed: int = 7) -> dict:
    """Categorise transactions, and measure what it takes to generalise.

    Evaluated on **held-out merchants**: whole merchant stems are removed from
    training, so at test time the model meets strings it has never seen. A
    random transaction split would score far higher and mean far less, because
    the same merchant would sit on both sides of it.

    Two models are fitted deliberately, because the comparison is the point:

    * **text only** - character n-grams over the description. This is what most
      transaction-categorisation demos do, and on an unseen merchant it
      collapses, because nothing in "OCTOPUS ENERGY" resembles "EDF ENERGY".
    * **text + behaviour** - the same n-grams alongside amount, sign,
      recurrence and amount stability. A merchant the model has never seen is
      still recognisable as a monthly fixed commitment.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    frame = add_behavioural_features(ledger)

    held_out: list[str] = []
    for _, group in frame.groupby("true_category"):
        stems = sorted(group["merchant_stem"].unique())
        if len(stems) > 2:
            held_out.extend(rng.choice(stems, size=max(1, len(stems) // 4), replace=False).tolist())

    is_test = frame["merchant_stem"].isin(held_out)
    train, test = frame[~is_test], frame[is_test]

    def text_block() -> TfidfVectorizer:
        # Character n-grams, not words: bank descriptors are full of
        # truncations and run-together tokens that word features miss.
        return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)

    text_only = Pipeline([("tfidf", text_block()), ("clf", LogisticRegression(max_iter=1500, C=5.0))])
    text_only.fit(train["description"], train["true_category"])
    text_only_accuracy = (
        float((text_only.predict(test["description"]) == test["true_category"]).mean()) if len(test) else float("nan")
    )

    hybrid = Pipeline(
        [
            (
                "features",
                ColumnTransformer(
                    [
                        ("text", text_block(), "description"),
                        ("behaviour", StandardScaler(), BEHAVIOURAL_FEATURES),
                    ]
                ),
            ),
            ("clf", LogisticRegression(max_iter=2000, C=5.0)),
        ]
    )
    feature_columns = ["description", *BEHAVIOURAL_FEATURES]
    hybrid.fit(train[feature_columns], train["true_category"])
    hybrid_accuracy = (
        float((hybrid.predict(test[feature_columns]) == test["true_category"]).mean()) if len(test) else float("nan")
    )

    # The honest baseline is the majority class of the *test* set, not a uniform
    # guess over the taxonomy. The held-out transactions are heavily skewed
    # towards groceries and shopping, so "1/n_categories" flatters every model
    # measured against it.
    majority_baseline = float(test["true_category"].value_counts(normalize=True).max()) if len(test) else float("nan")

    LOGGER.info(
        "Unseen-merchant accuracy - text only %.3f, text + behaviour %.3f "
        "(majority-class baseline %.3f, %s held-out merchants)",
        text_only_accuracy,
        hybrid_accuracy,
        majority_baseline,
        len(held_out),
    )

    return {
        "pipeline": hybrid,
        "text_only_pipeline": text_only,
        "feature_columns": feature_columns,
        "held_out_merchants": held_out,
        "text_only_accuracy": text_only_accuracy,
        "majority_class_baseline": majority_baseline,
        "unseen_merchant_accuracy": hybrid_accuracy,
        "report": classification_report(test["true_category"], hybrid.predict(test[feature_columns]), zero_division=0)
        if len(test)
        else "",
    }


def categorise(ledger: pd.DataFrame, classifier, feature_columns: list[str] | None = None) -> pd.DataFrame:
    """Attach a predicted category to every transaction."""
    out = add_behavioural_features(ledger)
    columns = feature_columns or ["description", *BEHAVIOURAL_FEATURES]
    out["category"] = classifier.predict(out[columns])
    return out


# --------------------------------------------------------------------------
# Affordability features
# --------------------------------------------------------------------------
def cashflow_features(ledger: pd.DataFrame, months: int = 6, category_column: str = "category") -> pd.DataFrame:
    """Derive per-applicant affordability signals from a categorised ledger.

    These are the questions an Open Banking affordability check asks that a
    bureau file cannot answer: is the income regular, does it cover the
    essentials, and is the account living in its overdraft?
    """
    frame = ledger.copy()
    frame["is_income"] = frame[category_column] == "income"
    frame["outflow"] = frame["amount"].where(frame["amount"] < 0, 0.0).abs()
    frame["month"] = (frame["day"] // 30).clip(upper=months - 1)

    grouped = frame.groupby("applicant_id")
    features = pd.DataFrame(index=grouped.size().index)

    income = frame[frame["is_income"]].groupby("applicant_id")["amount"]
    features["ob_income_monthly_median"] = income.median()
    features["ob_income_payment_count"] = income.size()
    # Coefficient of variation: a delivery rider and a salaried nurse can have
    # the same mean income and completely different affordability.
    features["ob_income_volatility"] = (income.std() / income.mean()).replace([np.inf, -np.inf], np.nan)
    features["ob_income_is_regular"] = (features["ob_income_volatility"] < 0.15).astype(float)

    total_out = grouped["outflow"].sum()
    for label, categories in (("essential", ESSENTIAL_CATEGORIES), ("discretionary", DISCRETIONARY_CATEGORIES)):
        spend = frame[frame[category_column].isin(categories)].groupby("applicant_id")["outflow"].sum()
        features[f"ob_{label}_spend_monthly"] = spend / months
        features[f"ob_{label}_spend_ratio"] = (spend / total_out).fillna(0.0)

    for category in ("debt_repayment", "cash", "fees"):
        spend = frame[frame[category_column] == category].groupby("applicant_id")["outflow"].sum()
        # Reindex before filling: an applicant with no transactions in this
        # category is absent from the groupby entirely, and their true value is
        # zero rather than unknown.
        features[f"ob_{category}_monthly"] = (spend.reindex(features.index) / months).fillna(0.0)

    fee_events = frame[frame[category_column] == "fees"].groupby("applicant_id").size()
    features["ob_fee_events"] = fee_events.reindex(features.index).fillna(0.0)

    balance = grouped["balance_after"]
    features["ob_min_balance"] = balance.min()
    features["ob_mean_balance"] = balance.mean()
    negative_days = frame[frame["balance_after"] < 0].groupby("applicant_id")["day"].nunique()
    features["ob_days_in_overdraft"] = negative_days.reindex(features.index).fillna(0.0)

    # The headline affordability number: what is left each month once income
    # has covered every non-negotiable cost.
    features["ob_surplus_after_essentials"] = (
        features["ob_income_monthly_median"].fillna(0.0) - features["ob_essential_spend_monthly"].fillna(0.0)
    )
    features["ob_essential_cover_ratio"] = (
        features["ob_income_monthly_median"] / features["ob_essential_spend_monthly"].replace(0, np.nan)
    )

    # Month-to-month spending volatility, a proxy for how predictable the
    # household's outgoings are.
    monthly_out = frame.groupby(["applicant_id", "month"])["outflow"].sum().unstack(fill_value=0.0)
    features["ob_spend_volatility"] = (monthly_out.std(axis=1) / monthly_out.mean(axis=1)).replace(
        [np.inf, -np.inf], np.nan
    )

    return features.reset_index().fillna({"ob_income_is_regular": 0.0})


def build_openbanking_features(cfg) -> tuple[pd.DataFrame, dict]:
    """Full Open Banking track: simulate, categorise, aggregate, persist."""
    spec = LedgerSpec(
        n_applicants=int(cfg.get("openbanking.n_applicants", 3000)),
        months=int(cfg.get("openbanking.months_of_history", 6)),
        seed=int(cfg.get("openbanking.seed", 7)),
    )
    ledger = generate_ledger(spec)
    trained = train_merchant_classifier(ledger, seed=spec.seed)
    categorised = categorise(ledger, trained["pipeline"], trained["feature_columns"])
    features = cashflow_features(categorised, months=spec.months)

    gold = cfg.path("paths.gold")
    gold.mkdir(parents=True, exist_ok=True)
    ledger.head(500).to_csv(gold / "openbanking_ledger_sample.csv", index=False)
    features.to_parquet(gold / "openbanking_features.parquet", index=False)

    agreement = float((categorised["category"] == categorised["true_category"]).mean())
    summary = {
        "simulated": True,
        "applicants": spec.n_applicants,
        "transactions": int(len(ledger)),
        "months_of_history": spec.months,
        "held_out_merchants": len(trained["held_out_merchants"]),
        "n_categories": int(ledger["true_category"].nunique()),
        # Two baselines, because the uniform one is too easy to beat and
        # quoting it alone would overstate the result. The majority-class
        # figure is the one the accuracies below should be judged against.
        "uniform_baseline_accuracy": 1.0 / ledger["true_category"].nunique(),
        "majority_class_baseline_accuracy": trained["majority_class_baseline"],
        "unseen_merchant_accuracy_text_only": trained["text_only_accuracy"],
        "unseen_merchant_accuracy_text_plus_behaviour": trained["unseen_merchant_accuracy"],
        # In-sample: most of these merchants were in training. Reported for
        # context on what a familiar-merchant feed looks like, not as a result.
        "full_ledger_agreement_in_sample": agreement,
        "features_derived": [c for c in features.columns if c.startswith("ob_")],
    }
    return features, summary


def main() -> None:  # pragma: no cover - CLI entry point
    import json

    from creditrisk.config import load_config
    from creditrisk.logging_utils import setup_logging

    setup_logging()
    cfg = load_config()
    _, summary = build_openbanking_features(cfg)
    path = cfg.path("paths.reports") / "openbanking_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOGGER.info("Open Banking summary written to %s", path)


if __name__ == "__main__":  # pragma: no cover
    main()
