"""Weight-of-Evidence binning and Information Value.

WoE is the transform that makes a credit scorecard auditable. Each feature is
cut into a handful of bins, and each bin is replaced by

    WoE = ln( (share of goods in bin) / (share of bads in bin) )

so a positive WoE means "this bin is safer than the book average". Two things
fall out of it that a raw continuous feature cannot give you:

* **Monotonicity.** Bins are merged until the bad rate moves in one direction
  across the feature. A scorecard where more missed payments can *lower* your
  risk is one no credit committee will sign off, however good its AUC.
* **A published bin table.** ``bin_table()`` produces exactly the artefact a
  model-risk reviewer asks for - counts, bad rate, WoE and IV per bin.

Missing values are given their own bin rather than imputed. In credit data,
"we have no record" is information, not an inconvenience.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MISSING_LABEL = "Missing"


@dataclass
class FeatureBins:
    """Fitted bins for a single feature."""

    name: str
    edges: np.ndarray  # inner cut points; bins are (-inf, e0], (e0, e1], ... (en, inf)
    woe: np.ndarray  # WoE per finite bin, aligned with ``edges``
    missing_woe: float
    information_value: float
    stats: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)

    def transform(self, values: pd.Series) -> np.ndarray:
        """Map raw values onto their bin's WoE."""
        numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
        out = np.full(numeric.shape, self.missing_woe, dtype=float)
        present = ~np.isnan(numeric)
        if present.any():
            idx = np.searchsorted(self.edges, numeric[present], side="left")
            out[present] = self.woe[idx]
        return out


def _event_rate_direction(bin_index: np.ndarray, bad_rate: np.ndarray) -> int:
    """Sign of the trend between bin order and bad rate (+1 rising, -1 falling)."""
    if len(bad_rate) < 2:
        return 1
    if np.allclose(bad_rate, bad_rate[0]):
        return 1
    corr = np.corrcoef(bin_index.astype(float), bad_rate)[0, 1]
    if np.isnan(corr):
        return 1
    return 1 if corr >= 0 else -1


def _merge_at(counts: list[list[float]], edges: list[float], position: int) -> tuple[list[list[float]], list[float]]:
    """Merge bin ``position`` into ``position + 1``, dropping the edge between them."""
    counts[position + 1][0] += counts[position][0]
    counts[position + 1][1] += counts[position][1]
    del counts[position]
    del edges[position]
    return counts, edges


def fit_feature_bins(
    values: pd.Series,
    target: pd.Series,
    *,
    max_bins: int = 8,
    min_bin_fraction: float = 0.05,
    smoothing: float = 0.5,
    name: str | None = None,
) -> FeatureBins:
    """Fit monotonic WoE bins for one feature.

    ``target`` is 1 for a bad (default) and 0 for a good.
    """
    name = name or (values.name if values.name is not None else "feature")
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    y = target.to_numpy(dtype=int)

    present = ~np.isnan(numeric)
    x_present, y_present = numeric[present], y[present]

    total_bad = float(y.sum())
    total_good = float(len(y) - total_bad)
    if total_bad == 0 or total_good == 0:
        raise ValueError(f"Feature '{name}': target has only one class, WoE is undefined")

    # --- initial cut points, on quantiles of the observed distribution -------
    if len(np.unique(x_present)) <= 1:
        edges: list[float] = []
    else:
        quantiles = np.linspace(0, 1, max_bins + 1)[1:-1]
        edges = sorted(set(np.quantile(x_present, quantiles).round(10).tolist()))

    def bin_counts(edge_list: list[float]) -> list[list[float]]:
        idx = np.searchsorted(np.array(edge_list), x_present, side="left") if edge_list else np.zeros_like(x_present, dtype=int)
        counts = []
        for b in range(len(edge_list) + 1):
            mask = idx == b
            bad = float(y_present[mask].sum())
            counts.append([float(mask.sum()) - bad, bad])  # [good, bad]
        return counts

    counts = bin_counts(edges)

    # --- enforce a minimum bin size -----------------------------------------
    min_count = max(1.0, min_bin_fraction * len(x_present))
    changed = True
    while changed and len(counts) > 1:
        changed = False
        for position in range(len(counts)):
            if sum(counts[position]) < min_count:
                merge_position = position if position < len(counts) - 1 else position - 1
                counts, edges = _merge_at(counts, edges, merge_position)
                changed = True
                break

    # --- enforce monotonic bad rate -----------------------------------------
    def bad_rates(cs: list[list[float]]) -> np.ndarray:
        return np.array([b / max(g + b, 1.0) for g, b in cs])

    direction = _event_rate_direction(np.arange(len(counts)), bad_rates(counts))
    # Loop while more than one bin remains, not two: with exactly two bins a
    # violation is still a violation, and stopping early would ship a
    # non-monotonic feature while advertising monotonicity.
    while len(counts) > 1:
        rates = bad_rates(counts)
        deltas = np.diff(rates) * direction
        violations = np.where(deltas < 0)[0]
        if len(violations) == 0:
            break
        # Merge the pair that violates monotonicity most severely.
        worst = int(violations[np.argmin(deltas[violations])])
        counts, edges = _merge_at(counts, edges, worst)

    # --- WoE and IV ----------------------------------------------------------
    rows = []
    woe_values = []
    iv = 0.0

    def woe_of(good: float, bad: float) -> tuple[float, float]:
        share_good = (good + smoothing) / (total_good + smoothing * 2)
        share_bad = (bad + smoothing) / (total_bad + smoothing * 2)
        w = float(np.log(share_good / share_bad))
        return w, (share_good - share_bad) * w

    labels = _bin_labels(edges)
    for (good, bad), label in zip(counts, labels, strict=True):
        w, contribution = woe_of(good, bad)
        woe_values.append(w)
        iv += contribution
        rows.append(
            {
                "feature": name,
                "bin": label,
                "count": int(good + bad),
                "goods": int(good),
                "bads": int(bad),
                "bad_rate": bad / max(good + bad, 1.0),
                "woe": w,
                "iv_contribution": contribution,
            }
        )

    missing_good = float((~present & (y == 0)).sum())
    missing_bad = float((~present & (y == 1)).sum())
    if missing_good + missing_bad > 0:
        missing_woe, contribution = woe_of(missing_good, missing_bad)
        iv += contribution
        rows.append(
            {
                "feature": name,
                "bin": MISSING_LABEL,
                "count": int(missing_good + missing_bad),
                "goods": int(missing_good),
                "bads": int(missing_bad),
                "bad_rate": missing_bad / max(missing_good + missing_bad, 1.0),
                "woe": missing_woe,
                "iv_contribution": contribution,
            }
        )
    else:
        # No missing values in training. Neutral WoE keeps scoring safe if one
        # turns up in production rather than throwing at inference time.
        missing_woe = 0.0

    return FeatureBins(
        name=str(name),
        edges=np.array(edges, dtype=float),
        woe=np.array(woe_values, dtype=float),
        missing_woe=float(missing_woe),
        information_value=float(iv),
        stats=pd.DataFrame(rows),
    )


def _bin_labels(edges: list[float]) -> list[str]:
    """Human-readable interval labels for the fitted cut points."""
    if not edges:
        return ["(-inf, inf)"]
    labels = [f"(-inf, {edges[0]:g}]"]
    labels += [f"({edges[i]:g}, {edges[i + 1]:g}]" for i in range(len(edges) - 1)]
    labels.append(f"({edges[-1]:g}, inf)")
    return labels


class WoETransformer:
    """Fit and apply WoE bins across a feature matrix."""

    def __init__(self, max_bins: int = 8, min_bin_fraction: float = 0.05, smoothing: float = 0.5) -> None:
        self.max_bins = max_bins
        self.min_bin_fraction = min_bin_fraction
        self.smoothing = smoothing
        self.bins_: dict[str, FeatureBins] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> WoETransformer:
        self.bins_ = {
            column: fit_feature_bins(
                X[column],
                y,
                max_bins=self.max_bins,
                min_bin_fraction=self.min_bin_fraction,
                smoothing=self.smoothing,
                name=column,
            )
            for column in X.columns
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the columns of ``X`` that have fitted bins.

        Iteration follows ``X``, not the fitted dictionary, so a caller may
        transform a subset of the fitted features - which is what feature
        selection needs - without refitting.
        """
        unknown = [c for c in X.columns if c not in self.bins_]
        if unknown:
            raise KeyError(f"No fitted WoE bins for: {unknown}")
        return pd.DataFrame(
            {f"woe_{column}": self.bins_[column].transform(X[column]) for column in X.columns},
            index=X.index,
        )

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def information_values(self) -> pd.Series:
        """IV per feature, strongest first."""
        return pd.Series(
            {name: binning.information_value for name, binning in self.bins_.items()},
            name="information_value",
        ).sort_values(ascending=False)

    def bin_table(self) -> pd.DataFrame:
        """The full published bin table across every feature."""
        frames = [binning.stats for binning in self.bins_.values() if not binning.stats.empty]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def select_features(self, min_iv: float) -> list[str]:
        """Features whose IV clears the configured floor."""
        iv = self.information_values()
        return iv[iv >= min_iv].index.tolist()


def iv_strength(value: float) -> str:
    """The conventional Siddiqi reading of an Information Value."""
    if value < 0.02:
        return "unpredictive"
    if value < 0.10:
        return "weak"
    if value < 0.30:
        return "medium"
    if value < 0.50:
        return "strong"
    return "suspiciously strong - check for leakage"
