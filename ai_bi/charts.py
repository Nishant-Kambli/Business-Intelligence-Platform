"""Small reusable charts for workforce intelligence pages."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

plt.style.use("default")


def _finish(fig, ax, x_label: str = ""):
    fig.set_facecolor("white")
    ax.set_facecolor("white")
    ax.tick_params(colors="#1f2937")
    ax.xaxis.label.set_color("#1f2937")
    ax.yaxis.label.set_color("#1f2937")
    ax.title.set_color("#1f2937")
    ax.set_xlabel(x_label)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.grid(axis="x", color="#e2e8f0", alpha=1)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def group_bar_chart(summary: pd.DataFrame, title: str, x_label: str = "Value", value_column: str = "Value"):
    """Create a sorted horizontal bar chart from Group and value columns."""
    plot = summary.sort_values(value_column) if value_column in summary else summary
    fig, ax = plt.subplots(figsize=(8, max(3, 0.42 * len(plot))))
    if not plot.empty and "Group" in plot and value_column in plot:
        ax.barh(plot["Group"].astype(str), plot[value_column], color="#087e8b")
    ax.set_title(title)
    return _finish(fig, ax, x_label)


def attrition_bar_chart(summary: pd.DataFrame, title: str = "Historical attrition rate by group"):
    """Plot historical attrition rates by group."""
    return group_bar_chart(summary, title, "Attrition rate (%)", "Attrition rate")


def metric_comparison_chart(metrics: pd.DataFrame):
    """Compare held-out model metrics when evaluation artifacts are available."""
    metric_columns = [name for name in ("Accuracy", "Precision", "Recall", "F1-score") if name in metrics]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if not metrics.empty and metric_columns and "Model" in metrics:
        metrics.set_index("Model")[metric_columns].plot(kind="bar", ax=ax, color=["#0b3954", "#087e8b", "#bfd7ea", "#ff5a5f"][:len(metric_columns)])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Held-out test metrics")
    ax.tick_params(axis="x", rotation=0)
    fig.set_facecolor("white")
    ax.set_facecolor("white")
    ax.tick_params(colors="#1f2937")
    ax.xaxis.label.set_color("#1f2937")
    ax.yaxis.label.set_color("#1f2937")
    ax.title.set_color("#1f2937")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.grid(axis="y", color="#e2e8f0", alpha=1)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def feature_importance_chart(features: pd.DataFrame, limit: int = 15):
    """Plot cached model feature importance when present."""
    selected = features.nlargest(limit, "importance").sort_values("importance") if "importance" in features else features.head(0)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.34 * len(selected))))
    if not selected.empty and "feature" in selected:
        ax.barh(selected["feature"], selected["importance"], color="#087e8b")
    ax.set_xlabel("Random Forest importance")
    ax.set_title(f"Top {len(selected)} Random Forest features")
    fig.set_facecolor("white")
    ax.set_facecolor("white")
    ax.tick_params(colors="#1f2937")
    ax.xaxis.label.set_color("#1f2937")
    ax.yaxis.label.set_color("#1f2937")
    ax.title.set_color("#1f2937")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color="#e2e8f0", alpha=1)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def histogram(frame: pd.DataFrame, field: str, title: str, bins: int = 20):
    """Return a compact histogram for a numeric dataset field."""
    values = pd.to_numeric(frame[field], errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(8, 4))
    if not values.empty:
        ax.hist(values, bins=bins, color="#087e8b", edgecolor="white")
    ax.set_title(title)
    ax.set_ylabel("Employees")
    fig.set_facecolor("white")
    ax.set_facecolor("white")
    ax.tick_params(colors="#1f2937")
    ax.xaxis.label.set_color("#1f2937")
    ax.yaxis.label.set_color("#1f2937")
    ax.title.set_color("#1f2937")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e2e8f0", alpha=1)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig
