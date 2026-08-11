"""Report figures.

One visual system across every chart: a single surface colour, recessive grid
and axes, thin marks, and colour assigned by the job it is doing rather than by
whatever came next in a cycle. Categorical series take fixed hue slots;
magnitude uses one hue light-to-dark. Nothing here uses a second y-axis - where
two measures share an x, they get stacked panels instead.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

LOGGER = logging.getLogger(__name__)

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # blue, orange, aqua - fixed order
# One hue, light to dark, for ordered categories. The lightest step is held at
# a level that still reads against the surface rather than dissolving into it.
SEQUENTIAL = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
STATUS_CRITICAL = "#d03b3b"
STATUS_GOOD = "#0ca30c"


def apply_style() -> None:
    """Chart chrome: recessive everything, so the data is the only loud thing."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "600",
            "axes.titlecolor": INK_PRIMARY,
            "axes.labelcolor": INK_SECONDARY,
            "axes.labelsize": 9.5,
            "axes.edgecolor": BASELINE,
            "axes.linewidth": 1.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "figure.dpi": 150,
        }
    )


def _finish(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Wrote %s", path)
    return path


def score_distribution(scores: np.ndarray, y_true: np.ndarray, path: Path) -> Path:
    """Score distributions for accounts that performed and accounts that defaulted."""
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    # A WoE scorecard emits a finite set of scores, so a fine binning turns
    # into a comb of artefacts. Fewer, wider bins over the central range show
    # the separation rather than the quantisation.
    bins = np.linspace(np.percentile(scores, 0.5), np.percentile(scores, 99.5), 30)

    for label, mask, colour in (
        ("Repaid", y_true == 0, SERIES[0]),
        ("Defaulted", y_true == 1, SERIES[1]),
    ):
        ax.hist(
            scores[mask], bins=bins, density=True, histtype="stepfilled",
            color=colour, alpha=0.28, linewidth=0,
        )
        ax.hist(scores[mask], bins=bins, density=True, histtype="step", color=colour, linewidth=2.0, label=label)

    ax.set_xlabel("Scorecard points")
    ax.set_ylabel("Share of applicants")
    ax.set_title("Score separation between repaid and defaulted accounts")
    ax.set_yticks([])
    ax.grid(axis="x", alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left")
    return _finish(fig, path)


def calibration(table: pd.DataFrame, path: Path) -> Path:
    """Predicted against observed default rate, by predicted-risk decile."""
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    limit = max(table["predicted_rate"].max(), table["observed_rate"].max()) * 1.12

    ax.plot([0, limit], [0, limit], color=BASELINE, linewidth=1.5, linestyle=(0, (4, 3)), zorder=1)
    # Label placed in the empty region below the diagonal: the fitted line hugs
    # the diagonal, so anything near it collides.
    ax.annotate(
        "perfect calibration",
        xy=(limit * 0.55, limit * 0.55), xytext=(limit * 0.60, limit * 0.26),
        color=INK_MUTED, fontsize=8.5,
        arrowprops={"arrowstyle": "-", "color": INK_MUTED, "linewidth": 0.9},
    )
    ax.plot(
        table["predicted_rate"], table["observed_rate"],
        color=SERIES[0], linewidth=2.0, marker="o", markersize=8,
        markerfacecolor=SERIES[0], markeredgecolor=SURFACE, markeredgewidth=2, zorder=3,
    )

    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_xlabel("Predicted default rate")
    ax.set_ylabel("Observed default rate")
    ax.set_title("Calibration on the held-out sample")
    ax.grid(alpha=0.7)
    ax.set_axisbelow(True)
    return _finish(fig, path)


def gains(table: pd.DataFrame, path: Path) -> Path:
    """Share of all defaults captured as the riskiest deciles are excluded."""
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    x = table["cumulative_population"] * 100
    y = table["cumulative_bad_capture"] * 100

    ax.plot([0, 100], [0, 100], color=BASELINE, linewidth=1.5, linestyle=(0, (4, 3)), zorder=1)
    ax.text(78, 68, "no model", color=INK_MUTED, fontsize=8.5, rotation=32, va="bottom")
    ax.plot(x, y, color=SERIES[0], linewidth=2.2, marker="o", markersize=7,
            markerfacecolor=SERIES[0], markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)

    # One direct label, at the decile a credit policy would actually cite.
    highlight = table.iloc[1]
    ax.annotate(
        f"{highlight['cumulative_bad_capture'] * 100:.0f}% of defaults\nsit in the riskiest 20%",
        xy=(highlight["cumulative_population"] * 100, highlight["cumulative_bad_capture"] * 100),
        xytext=(28, 62), color=INK_SECONDARY, fontsize=9,
        arrowprops={"arrowstyle": "-", "color": INK_MUTED, "linewidth": 1},
    )

    ax.set_xlabel("Applicants ranked riskiest first (%)")
    ax.set_ylabel("Defaults captured (%)")
    ax.set_title("Cumulative default capture")
    ax.grid(alpha=0.7)
    ax.set_axisbelow(True)
    return _finish(fig, path)


def approval_and_profit(curve: pd.DataFrame, path: Path) -> Path:
    """Bad rate and book profit across the approval range.

    Two panels rather than two y-axes: a shared x, independent scales, and no
    invitation to read a crossing point that does not exist.
    """
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7.0, 5.6), sharex=True, gridspec_kw={"hspace": 0.16})
    x = curve["approval_rate"] * 100
    best = curve.loc[curve["expected_profit"].idxmax()]
    best_x = best["approval_rate"] * 100

    top.plot(x, curve["bad_rate"] * 100, color=SERIES[0], linewidth=2.2)
    top.axvline(best_x, color=INK_MUTED, linewidth=1.2, linestyle=(0, (4, 3)))
    top.set_ylabel("Bad rate of\naccepted book (%)")
    top.set_title("Where the book stops paying for itself")
    top.grid(alpha=0.7)
    top.set_axisbelow(True)

    bottom.plot(x, curve["expected_profit"] / 1e6, color=SERIES[1], linewidth=2.2)
    bottom.axvline(best_x, color=INK_MUTED, linewidth=1.2, linestyle=(0, (4, 3)))
    bottom.plot([best_x], [best["expected_profit"] / 1e6], marker="o", markersize=9,
                color=SERIES[1], markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
    bottom.annotate(
        f"optimum: approve {best_x:.0f}%\nbad rate {best['bad_rate'] * 100:.1f}%",
        xy=(best_x, best["expected_profit"] / 1e6),
        xytext=(best_x - 42, best["expected_profit"] / 1e6 * 0.55),
        color=INK_SECONDARY, fontsize=9,
        arrowprops={"arrowstyle": "-", "color": INK_MUTED, "linewidth": 1},
    )
    bottom.set_ylabel("Expected profit\n(£m)")
    bottom.set_xlabel("Approval rate (%)")
    bottom.grid(alpha=0.7)
    bottom.set_axisbelow(True)
    return _finish(fig, path)


def band_default_rates(table: pd.DataFrame, path: Path) -> Path:
    """Realised default rate per policy grade - one hue, ordered by magnitude."""
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ordered = table.iloc[::-1].reset_index(drop=True)  # safest grade at the top
    # Darker means riskier: colour tracks the magnitude being shown, so the
    # bar length and the bar colour say the same thing.
    colours = SEQUENTIAL[: len(ordered)]

    positions = np.arange(len(ordered))
    bars = ax.barh(positions, ordered["observed_default_rate"] * 100, height=0.62, color=colours, linewidth=0)
    for bar_patch, (_, row) in zip(bars, ordered.iterrows(), strict=True):
        ax.text(
            bar_patch.get_width() + 0.5, bar_patch.get_y() + bar_patch.get_height() / 2,
            f"{row['observed_default_rate'] * 100:.1f}%   ({row['population_share'] * 100:.0f}% of book)",
            va="center", fontsize=9, color=INK_SECONDARY,
        )

    ax.set_yticks(positions)
    ax.set_yticklabels([f"{row['band']}  {row['label']}" for _, row in ordered.iterrows()], color=INK_SECONDARY)
    ax.set_xlabel("Observed default rate (%)")
    ax.set_title("Realised risk by policy grade, held-out sample")
    ax.set_xlim(0, ordered["observed_default_rate"].max() * 100 * 1.42)
    ax.grid(axis="x", alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    return _finish(fig, path)


def information_values(table: pd.DataFrame, path: Path, top_n: int = 14) -> Path:
    """Information Value per feature, split by whether it made the final model."""
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    frame = table.head(top_n).iloc[::-1].reset_index(drop=True)
    colours = [SERIES[0] if bool(selected) else BASELINE for selected in frame["selected"]]

    positions = np.arange(len(frame))
    ax.barh(positions, frame["information_value"], height=0.62, color=colours, linewidth=0)
    ax.set_yticks(positions)
    ax.set_yticklabels(frame["feature"], fontsize=8.5, color=INK_SECONDARY)
    ax.set_xscale("log")
    ax.set_xlabel("Information Value (log scale)")
    ax.set_title("Feature strength, and what survived selection")

    handles = [
        plt.Line2D([], [], marker="s", linestyle="", markersize=9, color=SERIES[0], label="In final model"),
        plt.Line2D([], [], marker="s", linestyle="", markersize=9, color=BASELINE, label="Excluded"),
    ]
    ax.legend(handles=handles, loc="lower right")
    ax.grid(axis="x", alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    return _finish(fig, path)


def build_all(cfg) -> list[Path]:
    """Regenerate every report figure from the persisted artefacts."""
    from creditrisk.features import TARGET
    from creditrisk.model import ScorecardModel
    from creditrisk.train import load_gold

    apply_style()
    reports = cfg.path("paths.reports")
    figures = cfg.path("paths.figures")
    model = ScorecardModel.load(cfg.path("paths.models") / "scorecard.joblib")

    gold = load_gold(cfg)
    test = gold[gold["split"] == "test"].reset_index(drop=True)
    y_true = test[TARGET].to_numpy(dtype=int)
    scores = model.score(test[model.features])

    written = [
        score_distribution(scores, y_true, figures / "score_distribution.png"),
        calibration(pd.read_csv(reports / "calibration_table.csv"), figures / "calibration.png"),
        gains(pd.read_csv(reports / "gains_table.csv"), figures / "gains.png"),
        approval_and_profit(pd.read_csv(reports / "approval_curve.csv"), figures / "approval_profit.png"),
        band_default_rates(pd.read_csv(reports / "band_table.csv"), figures / "band_default_rates.png"),
        information_values(pd.read_csv(reports / "information_values.csv"), figures / "information_values.png"),
    ]
    return written


def main() -> None:  # pragma: no cover - CLI entry point
    from creditrisk.config import load_config
    from creditrisk.logging_utils import setup_logging

    setup_logging()
    build_all(load_config())


if __name__ == "__main__":  # pragma: no cover
    main()
