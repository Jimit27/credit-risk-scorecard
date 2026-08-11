"""Credit-risk evaluation metrics.

AUC alone does not tell a lender whether a model is usable. A scorecard is
judged on three separate questions, and this module answers all three:

* **Does it rank?** Gini and the KS statistic.
* **Are the probabilities true?** Brier score, log loss and a decile-level
  calibration table. A model that ranks perfectly but says 3% when the real
  rate is 9% will price the whole book wrong.
* **Is it stable?** Population Stability Index between the development sample
  and whatever is being scored now.

The gains table and the profit curve then turn those into the numbers a credit
committee actually decides on: at a given approval rate, what is the bad rate,
and where does the book stop making money.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


def gini(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Somers' D / accuracy ratio: ``2 * AUC - 1``.

    Credit teams quote Gini, not AUC. A retail scorecard in the 0.40-0.60 band
    is doing an ordinary-to-good job.
    """
    return 2.0 * roc_auc_score(y_true, y_score) - 1.0


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Kolmogorov-Smirnov separation between the good and bad score distributions.

    Ties are evaluated only at the end of each tie group. This matters here more
    than in most places: a WoE scorecard emits a finite set of scores, so ties
    are the norm rather than the exception, and taking the running maximum
    *within* a tie group reads an ordering out of applicants the model scored
    identically - inflating KS with an ordering the model never expressed.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    order = np.argsort(-y_score, kind="mergesort")
    labels = y_true[order]
    sorted_scores = y_score[order]

    total_bad = labels.sum()
    total_good = len(labels) - total_bad
    if total_bad == 0 or total_good == 0:
        return 0.0

    cum_bad = np.cumsum(labels) / total_bad
    cum_good = np.cumsum(1 - labels) / total_good

    # Keep only the last position of each run of equal scores.
    boundaries = np.append(sorted_scores[:-1] != sorted_scores[1:], True)
    return float(np.max(np.abs(cum_bad - cum_good)[boundaries]))


def population_stability_index(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """PSI between a development distribution and a current one.

    Convention: below 0.10 stable, 0.10-0.25 investigate, above 0.25 the
    population has moved far enough that the model should be revalidated.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    quantiles = np.linspace(0, 100, bins + 1)
    edges = np.unique(np.percentile(expected, quantiles))
    if len(edges) < 3:
        # Quantile edges collapse whenever a variable is heavily concentrated on
        # one value - which describes every arrears counter in a credit file,
        # where 95% of applicants sit at zero. Returning 0.0 here would report
        # "no drift" on precisely the features whose drift matters most, so fall
        # back to comparing the distributions over their distinct values.
        return _discrete_psi(expected, actual)
    edges[0], edges[-1] = -np.inf, np.inf

    expected_share = np.histogram(expected, bins=edges)[0] / len(expected)
    actual_share = np.histogram(actual, bins=edges)[0] / len(actual)

    # Floor the shares so an empty bin gives a large-but-finite contribution
    # instead of an infinity that hides every other bin's movement.
    floor = 1e-4
    expected_share = np.clip(expected_share, floor, None)
    actual_share = np.clip(actual_share, floor, None)
    return float(np.sum((actual_share - expected_share) * np.log(actual_share / expected_share)))


def _discrete_psi(expected: np.ndarray, actual: np.ndarray, max_levels: int = 20) -> float:
    """PSI over distinct values, for variables too concentrated to bin by quantile.

    The most common ``max_levels`` values in the reference sample become bins;
    everything else is pooled into a final "other" bin, so an actual sample that
    introduces entirely new values still registers.
    """
    values, counts = np.unique(expected, return_counts=True)
    levels = values[np.argsort(-counts)][:max_levels]

    def shares(sample: np.ndarray) -> np.ndarray:
        counted = np.array([(sample == level).sum() for level in levels], dtype=float)
        other = len(sample) - counted.sum()
        return np.append(counted, other) / max(len(sample), 1)

    floor = 1e-4
    expected_share = np.clip(shares(expected), floor, None)
    actual_share = np.clip(shares(actual), floor, None)
    return float(np.sum((actual_share - expected_share) * np.log(actual_share / expected_share)))


def psi_verdict(value: float, stable: float = 0.10, investigate: float = 0.25) -> str:
    if value < stable:
        return "stable"
    if value < investigate:
        return "investigate"
    return "revalidate"


def calibration_table(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Predicted versus observed default rate, by predicted-risk decile."""
    frame = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_prob)})
    frame["decile"] = pd.qcut(frame["p"].rank(method="first"), bins, labels=False) + 1
    table = (
        frame.groupby("decile")
        .agg(count=("y", "size"), predicted_rate=("p", "mean"), observed_rate=("y", "mean"))
        .reset_index()
    )
    table["difference"] = table["predicted_rate"] - table["observed_rate"]
    return table


def gains_table(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Cumulative bad capture by risk decile - the classic scorecard gains chart."""
    frame = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_prob)}).sort_values("p", ascending=False)
    frame["decile"] = np.minimum((np.arange(len(frame)) / len(frame) * bins).astype(int) + 1, bins)
    grouped = frame.groupby("decile").agg(count=("y", "size"), bads=("y", "sum")).reset_index()
    grouped["bad_rate"] = grouped["bads"] / grouped["count"]
    grouped["cumulative_bads"] = grouped["bads"].cumsum()
    grouped["cumulative_bad_capture"] = grouped["cumulative_bads"] / grouped["bads"].sum()
    grouped["cumulative_population"] = grouped["count"].cumsum() / grouped["count"].sum()
    grouped["lift"] = grouped["cumulative_bad_capture"] / grouped["cumulative_population"]
    return grouped


def approval_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    revenue_per_good: float,
    loss_per_bad: float,
    steps: int = 100,
) -> pd.DataFrame:
    """Book economics as the cut-off moves from strict to permissive.

    For each approval rate this returns the resulting bad rate and the expected
    profit of the accepted book. The profit-maximising row is the cut-off the
    business would actually choose - which is rarely the one that maximises AUC.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    order = np.argsort(y_prob)  # safest applicants first
    labels = y_true[order]

    rows = []
    n = len(labels)
    # The sweep runs to step == steps, i.e. a 100% approval rate, so the final
    # row really is "approve everyone" and can be used as the do-nothing
    # baseline. Stopping at 99% and calling it approve-all overstates the
    # baseline and understates the model's contribution.
    for step in range(1, steps + 1):
        approval_rate = step / steps
        cutoff_index = max(1, int(round(approval_rate * n)))
        accepted = labels[:cutoff_index]
        bads = int(accepted.sum())
        goods = len(accepted) - bads
        rows.append(
            {
                "approval_rate": cutoff_index / n,
                "accepted": cutoff_index,
                "bad_rate": bads / cutoff_index,
                "expected_profit": goods * revenue_per_good - bads * loss_per_bad,
                "score_cutoff_probability": float(y_prob[order][cutoff_index - 1]),
            }
        )
    return pd.DataFrame(rows)


def evaluate(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """The headline metric set for a single model on a single sample."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    auc = float(roc_auc_score(y_true, y_prob))
    return {
        "auc": auc,
        "gini": 2.0 * auc - 1.0,
        "ks": ks_statistic(y_true, y_prob),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "observed_default_rate": float(y_true.mean()),
        "mean_predicted_rate": float(y_prob.mean()),
        "n": int(len(y_true)),
    }
