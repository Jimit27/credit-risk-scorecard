"""Train, calibrate and compare the champion scorecard and the GBM challenger.

The deliberate design choice here is that the *interpretable* model is the
champion. A WoE logistic scorecard is what a lender can defend to a regulator,
and the gradient-boosted challenger exists to quantify what that defensibility
costs in discrimination. If the gap is small, interpretability is free and the
scorecard ships. Publishing both, rather than only the winner, is the honest
version of a model-selection section.

Calibration is fitted on the validation split and the test split is scored
exactly once, at the end.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from creditrisk.config import Config, load_config
from creditrisk.features import MODEL_FEATURES, TARGET
from creditrisk.metrics import (
    approval_curve,
    calibration_table,
    evaluate,
    gains_table,
    population_stability_index,
    psi_verdict,
)
from creditrisk.model import ScorecardModel
from creditrisk.scorecard import ScoreScaling
from creditrisk.woe import MISSING_LABEL, WoETransformer, iv_strength

LOGGER = logging.getLogger(__name__)


def load_gold(cfg: Config) -> pd.DataFrame:
    """Read the gold table with pandas.

    Spark builds the lake; it is not needed to read it. Keeping training free
    of a Spark dependency is what lets CI and the Streamlit app run on a plain
    Python install.
    """
    gold_dir = cfg.path("paths.gold") / "features"
    frame = pd.read_parquet(gold_dir)
    if "split" not in frame.columns:
        raise ValueError("Gold table is missing the 'split' column")
    frame["split"] = frame["split"].astype(str)
    return frame


def _splits(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {name: frame[frame["split"] == name].reset_index(drop=True) for name in ("train", "valid", "test")}


def _fit_calibrator(y_true: np.ndarray, y_prob: np.ndarray, method: str) -> Any:
    """Fit a post-hoc calibrator on the validation split."""
    if method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        calibrator.fit(y_prob, y_true)
        return calibrator
    if method == "platt":
        platt = LogisticRegression(max_iter=1000)
        platt.fit(y_prob.reshape(-1, 1), y_true)

        class _PlattWrapper:
            def __init__(self, model: LogisticRegression) -> None:
                self.model = model

            def predict(self, p: np.ndarray) -> np.ndarray:
                return self.model.predict_proba(np.asarray(p).reshape(-1, 1))[:, 1]

        return _PlattWrapper(platt)
    raise ValueError(f"Unknown calibration method: {method}")


def train(cfg: Config | None = None) -> dict[str, Any]:
    """Run the full training and evaluation cycle; return the metrics report."""
    cfg = cfg or load_config()
    reports_dir = cfg.path("paths.reports")
    models_dir = cfg.path("paths.models")
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    frame = load_gold(cfg)
    parts = _splits(frame)
    LOGGER.info(
        "Split sizes - train %s / valid %s / test %s",
        len(parts["train"]),
        len(parts["valid"]),
        len(parts["test"]),
    )

    features = [c for c in MODEL_FEATURES if c in frame.columns]
    y = {name: part[TARGET].to_numpy(dtype=int) for name, part in parts.items()}
    X = {name: part[features] for name, part in parts.items()}

    # --- Weight of Evidence, fitted on train only ---------------------------
    woe = WoETransformer(
        max_bins=int(cfg.get("woe.max_bins", 8)),
        min_bin_fraction=float(cfg.get("woe.min_bin_fraction", 0.05)),
        smoothing=float(cfg.get("woe.laplace_smoothing", 0.5)),
    )
    woe.fit(X["train"], pd.Series(y["train"]))

    iv = woe.information_values()
    min_iv = float(cfg.get("woe.min_information_value", 0.02))
    # iv is already sorted descending, so the survivor of a correlated pair is
    # always the stronger feature.
    above_floor = [f for f in iv.index if f in features and iv[f] >= min_iv]

    # A feature whose observed values all fall in one bin cannot separate
    # anybody: its only discrimination comes from the Missing bin, which for the
    # arrears counters is a couple of hundred sentinel-coded records. It clears
    # the IV floor on the strength of that handful and then sits in the points
    # table looking like a major driver. Require at least two populated bins
    # among applicants who actually have a value.
    degenerate = {}
    for feature in list(above_floor):
        finite_bins = (woe.bins_[feature].stats["bin"] != MISSING_LABEL).sum()
        if finite_bins < 2:
            degenerate[feature] = "single populated bin - no separation among observed values"
            above_floor.remove(feature)
    if degenerate:
        LOGGER.info("Dropped %s degenerate feature(s): %s", len(degenerate), ", ".join(degenerate))

    # Several engineered delinquency features are near-restatements of each
    # other. Left in, they split one signal across correlated coefficients and
    # make the published points table unreadable, so the lower-IV member of any
    # highly correlated pair is dropped.
    selected, dropped_correlated = _prune_correlated(
        woe.transform(X["train"][above_floor]),
        ranking=above_floor,
        threshold=float(cfg.get("woe.max_pairwise_correlation", 0.85)),
    )
    LOGGER.info(
        "Selected %s of %s features (IV >= %s, then %s dropped as correlated)",
        len(selected),
        len(features),
        min_iv,
        len(dropped_correlated),
    )

    iv_frame = iv.reset_index()
    iv_frame.columns = ["feature", "information_value"]
    iv_frame["strength"] = iv_frame["information_value"].map(iv_strength)
    iv_frame["selected"] = iv_frame["feature"].isin(selected)
    exclusion_reasons = {**degenerate, **dropped_correlated}
    iv_frame["exclusion_reason"] = [
        "" if f in selected else exclusion_reasons.get(f, "IV below floor") for f in iv_frame["feature"]
    ]
    iv_frame.to_csv(reports_dir / "information_values.csv", index=False)
    woe.bin_table().to_csv(reports_dir / "woe_bin_table.csv", index=False)

    # Refit the transformer on the selected features only, so the persisted
    # bundle carries no bins it does not use.
    woe_selected = WoETransformer(
        max_bins=int(cfg.get("woe.max_bins", 8)),
        min_bin_fraction=float(cfg.get("woe.min_bin_fraction", 0.05)),
        smoothing=float(cfg.get("woe.laplace_smoothing", 0.5)),
    ).fit(X["train"][selected], pd.Series(y["train"]))

    woe_matrices = {name: woe_selected.transform(part[selected]) for name, part in parts.items()}

    # --- Champion: WoE logistic scorecard ------------------------------------
    logistic = LogisticRegression(
        C=float(cfg.get("model.logistic_woe.C", 1.0)),
        max_iter=int(cfg.get("model.logistic_woe.max_iter", 2000)),
        solver="lbfgs",
    )
    logistic.fit(woe_matrices["train"], y["train"])

    # --- Challenger: gradient boosting on raw features ----------------------
    gbm = _fit_gbm(cfg, X["train"][selected], y["train"], X["valid"][selected], y["valid"])

    scaling = ScoreScaling(
        base_score=float(cfg.get("scorecard.base_score", 600)),
        base_odds=float(cfg.get("scorecard.base_odds", 50)),
        pdo=float(cfg.get("scorecard.pdo", 20)),
    )
    bands = list(cfg.get("scorecard.bands", []))
    calibration_method = str(cfg.get("model.calibration.method", "isotonic"))

    candidates: dict[str, ScorecardModel] = {}

    logistic_valid_raw = logistic.predict_proba(woe_matrices["valid"])[:, 1]
    candidates["logistic_woe"] = ScorecardModel(
        kind="logistic_woe",
        estimator=logistic,
        features=selected,
        scaling=scaling,
        bands=bands,
        woe=woe_selected,
        calibrator=_fit_calibrator(y["valid"], logistic_valid_raw, calibration_method),
        metadata={"n_features": len(selected)},
    )

    if gbm is not None:
        gbm_valid_raw = gbm.predict_proba(X["valid"][selected])[:, 1]
        candidates["gbm"] = ScorecardModel(
            kind="gbm",
            estimator=gbm,
            features=selected,
            scaling=scaling,
            bands=bands,
            woe=None,
            calibrator=_fit_calibrator(y["valid"], gbm_valid_raw, calibration_method),
            metadata={"n_features": len(selected)},
        )

    # --- Evaluate every candidate on every split ----------------------------
    report: dict[str, Any] = {
        "dataset": {
            "rows": int(len(frame)),
            "features_available": len(features),
            "features_selected": len(selected),
            "default_rate": float(frame[TARGET].mean()),
            "split_sizes": {name: int(len(part)) for name, part in parts.items()},
        },
        "models": {},
    }

    for name, model in candidates.items():
        model_report: dict[str, Any] = {}
        for split_name, part in parts.items():
            probability = model.predict_proba(part[selected])
            model_report[split_name] = evaluate(y[split_name], probability)
        report["models"][name] = model_report
        LOGGER.info(
            "%-14s test Gini %.4f | KS %.4f | Brier %.5f",
            name,
            model_report["test"]["gini"],
            model_report["test"]["ks"],
            model_report["test"]["brier"],
        )

    champion_name = str(cfg.get("model.champion", "logistic_woe"))
    if champion_name not in candidates:
        champion_name = next(iter(candidates))
    champion = candidates[champion_name]
    report["champion"] = champion_name

    # --- Champion-only diagnostics ------------------------------------------
    train_probability = champion.predict_proba(X["train"][selected])
    test_probability = champion.predict_proba(X["test"][selected])

    calibration_table(y["test"], test_probability).to_csv(reports_dir / "calibration_table.csv", index=False)
    gains = gains_table(y["test"], test_probability)
    gains.to_csv(reports_dir / "gains_table.csv", index=False)

    # --- Cut-off: chosen on validation, reported on test --------------------
    # Picking the cut-off by maximising profit on the test set would be
    # selecting a hyperparameter on the holdout and then quoting the holdout as
    # an unbiased estimate. The PD threshold is chosen on validation and then
    # applied, unchanged, to test.
    revenue_per_good = float(cfg.get("economics.revenue_per_good", 450.0))
    loss_per_bad = float(cfg.get("economics.loss_per_bad", 2800.0))

    valid_probability = champion.predict_proba(X["valid"][selected])
    valid_curve = approval_curve(y["valid"], valid_probability, revenue_per_good, loss_per_bad)
    chosen = valid_curve.loc[valid_curve["expected_profit"].idxmax()]
    pd_cutoff = float(chosen["score_cutoff_probability"])

    test_curve = approval_curve(y["test"], test_probability, revenue_per_good, loss_per_bad)
    test_curve.to_csv(reports_dir / "approval_curve.csv", index=False)
    valid_curve.to_csv(reports_dir / "approval_curve_validation.csv", index=False)

    accepted = test_probability <= pd_cutoff
    accepted_bads = int(y["test"][accepted].sum())
    accepted_goods = int(accepted.sum() - accepted_bads)
    all_bads = int(y["test"].sum())
    profit_approve_all = (len(y["test"]) - all_bads) * revenue_per_good - all_bads * loss_per_bad

    report["business"] = {
        "cutoff_chosen_on": "validation",
        "pd_cutoff": pd_cutoff,
        "score_cutoff": float(np.min(champion.score(X["test"][selected])[accepted])) if accepted.any() else None,
        "approval_rate_on_validation": float(chosen["approval_rate"]),
        "approval_rate_on_test": float(accepted.mean()),
        "bad_rate_at_cutoff_on_test": float(accepted_bads / max(accepted.sum(), 1)),
        "expected_profit_at_cutoff_on_test": float(accepted_goods * revenue_per_good - accepted_bads * loss_per_bad),
        "expected_profit_approve_all_on_test": float(profit_approve_all),
    }
    report["business"]["profit_uplift_vs_approve_all"] = (
        report["business"]["expected_profit_at_cutoff_on_test"] - profit_approve_all
    )

    # Realised risk per policy grade on the held-out sample. A grade table whose
    # bad rate is not monotonic means the boundaries are wrong, so this gets
    # written out and checked rather than assumed.
    band_table = _band_table(champion, parts["test"][selected], y["test"])
    band_table.to_csv(reports_dir / "band_table.csv", index=False)
    report["bands"] = band_table.to_dict(orient="records")
    # Rows run from the lowest-scoring grade upwards, so a well-formed grade
    # table has a strictly falling default rate.
    report["bands_monotonic"] = bool(band_table["observed_default_rate"].is_monotonic_decreasing)

    psi_value = population_stability_index(
        train_probability, test_probability, bins=int(cfg.get("monitoring.psi_bins", 10))
    )
    report["stability"] = {
        "psi_train_vs_test": psi_value,
        "verdict": psi_verdict(
            psi_value,
            float(cfg.get("monitoring.psi_thresholds.stable", 0.10)),
            float(cfg.get("monitoring.psi_thresholds.investigate", 0.25)),
        ),
    }

    # Segment-level performance. A model with a good headline Gini can still be
    # materially worse for one group of applicants, and a lender is expected to
    # know that before it goes live rather than after.
    report["segments"] = _segment_report(parts["test"], y["test"], test_probability)

    champion_path = champion.save(models_dir / "scorecard.joblib")
    for name, model in candidates.items():
        model.save(models_dir / f"model_{name}.joblib")

    report["artifacts"] = {"champion_model": str(champion_path.relative_to(cfg.root))}
    (reports_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    LOGGER.info("Champion '%s' saved to %s", champion_name, champion_path)
    return report


def _prune_correlated(
    woe_matrix: pd.DataFrame, ranking: list[str], threshold: float
) -> tuple[list[str], dict[str, str]]:
    """Greedily drop features correlated above ``threshold`` with a stronger one.

    ``ranking`` is in descending IV order, so the survivor of each correlated
    pair is always the more informative feature. Correlation is measured on the
    WoE-transformed matrix, which is what the logistic actually sees.
    """
    correlation = woe_matrix.corr().abs()
    keep: list[str] = []
    dropped: dict[str, str] = {}
    for feature in ranking:
        column = f"woe_{feature}"
        if column not in correlation.columns:
            continue
        conflict = next(
            (k for k in keep if correlation.loc[column, f"woe_{k}"] >= threshold),
            None,
        )
        if conflict is None:
            keep.append(feature)
        else:
            dropped[feature] = f"correlated {correlation.loc[column, f'woe_{conflict}']:.2f} with {conflict}"
    return keep, dropped


def _fit_gbm(cfg: Config, X_train: pd.DataFrame, y_train: np.ndarray, X_valid: pd.DataFrame, y_valid: np.ndarray):
    """Fit the gradient-boosted challenger, if xgboost is installed."""
    try:
        from xgboost import XGBClassifier
    except ImportError:  # pragma: no cover - optional dependency
        LOGGER.warning("xgboost not installed; skipping the GBM challenger")
        return None

    params = cfg.get("model.gbm", {}) or {}
    model = XGBClassifier(
        n_estimators=int(params.get("n_estimators", 400)),
        max_depth=int(params.get("max_depth", 4)),
        learning_rate=float(params.get("learning_rate", 0.05)),
        subsample=float(params.get("subsample", 0.9)),
        colsample_bytree=float(params.get("colsample_bytree", 0.8)),
        min_child_weight=float(params.get("min_child_weight", 20)),
        reg_lambda=float(params.get("reg_lambda", 2.0)),
        eval_metric="auc",
        early_stopping_rounds=40,
        tree_method="hist",
        random_state=cfg.seed,
        n_jobs=4,
    )
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    return model


def _band_table(model: ScorecardModel, X: pd.DataFrame, y_true: np.ndarray) -> pd.DataFrame:
    """Population share and realised default rate for each policy grade."""
    decisions = model.decide(X)
    decisions["y"] = y_true
    order = [b["name"] for b in sorted(model.bands, key=lambda b: float(b["min_score"]))]
    labels = {str(b["name"]): str(b.get("label", b["name"])) for b in model.bands}
    grouped = (
        decisions.groupby("band")
        .agg(
            applicants=("y", "size"),
            observed_default_rate=("y", "mean"),
            min_score=("score", "min"),
            max_score=("score", "max"),
        )
        .reindex(order)
        .dropna(subset=["applicants"])
        .reset_index()
    )
    grouped["label"] = grouped["band"].map(labels)
    grouped["population_share"] = grouped["applicants"] / grouped["applicants"].sum()
    return grouped[["band", "label", "min_score", "max_score", "applicants", "population_share", "observed_default_rate"]]


def _segment_report(test_frame: pd.DataFrame, y_true: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    """Gini and calibration within each age band."""
    from sklearn.metrics import roc_auc_score

    out: dict[str, Any] = {}
    if "age_band" not in test_frame.columns:
        return out
    for band, index in test_frame.groupby("age_band").groups.items():
        positions = test_frame.index.get_indexer(index)
        y_segment = y_true[positions]
        p_segment = probability[positions]
        if len(np.unique(y_segment)) < 2 or len(y_segment) < 200:
            continue
        out[str(band)] = {
            "n": int(len(y_segment)),
            "gini": float(2 * roc_auc_score(y_segment, p_segment) - 1),
            "observed_default_rate": float(y_segment.mean()),
            "mean_predicted_rate": float(p_segment.mean()),
        }
    return out


def main() -> None:  # pragma: no cover - CLI entry point
    from creditrisk.logging_utils import setup_logging

    setup_logging()
    train()


if __name__ == "__main__":  # pragma: no cover
    main()
