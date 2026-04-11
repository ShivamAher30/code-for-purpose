"""
visualize.py — Smart Visualization Engine for Talk-to-Data
Features: Intelligent chart type selection, dark theme styling,
          comparison charts, anomaly highlighting.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np
import io
import re

# ──────────────────────────────────────────────
# THEME & STYLING
# ──────────────────────────────────────────────

# Premium color palette
COLORS = [
    "#6C63FF",  # Indigo
    "#E94560",  # Rose
    "#0ABAB5",  # Teal
    "#FF6B35",  # Orange
    "#A855F7",  # Purple
    "#14B8A6",  # Emerald
    "#F59E0B",  # Amber
    "#EC4899",  # Pink
    "#3B82F6",  # Blue
    "#10B981",  # Green
]

GRADIENT_COLORS = [
    "#6C63FF", "#8B7CF6", "#A78BFA", "#C4B5FD",  # Indigo gradient
]

BG_COLOR = "#0F0F1A"
CARD_BG = "#1A1A2E"
TEXT_COLOR = "#E8E8F0"
GRID_COLOR = "#2A2A3E"
ACCENT = "#6C63FF"


def _apply_dark_theme(fig, ax):
    """Apply premium dark theme to any matplotlib figure."""
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(CARD_BG)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID_COLOR)
    ax.spines["bottom"].set_color(GRID_COLOR)
    ax.grid(True, alpha=0.15, color=GRID_COLOR, linestyle="--")
    # Rotate x labels if too many
    if hasattr(ax, "get_xticklabels"):
        labels = ax.get_xticklabels()
        if len(labels) > 6:
            plt.setp(labels, rotation=35, ha="right", fontsize=8)


def _format_number(val):
    """Format large numbers with K/M/B suffixes."""
    if abs(val) >= 1e9:
        return f"{val/1e9:.1f}B"
    elif abs(val) >= 1e6:
        return f"{val/1e6:.1f}M"
    elif abs(val) >= 1e3:
        return f"{val/1e3:.1f}K"
    return f"{val:.1f}"


# ──────────────────────────────────────────────
# SAFE EXECUTION (unchanged from original)
# ──────────────────────────────────────────────

def execute_pandas_code_safely(code: str, df: pd.DataFrame):
    """
    Safely executes generated pandas code.
    Blocks any string with imports, os, sys, direct evals, dunders, etc.
    """
    # 1. Reject fundamentally unsafe patterns
    unsafe_patterns = ["import", "os", "sys", "eval", "exec", "__", "open", "read", "write"]
    for pattern in unsafe_patterns:
        if pattern in code:
            raise ValueError(f"Security Alert: Unsafe operation detected ({pattern}). Blocked execution.")
            
    # 2. Strict evaluation dict
    local_vars = {}
    
    # 3. Execute in sandbox explicitly mapping pd and df
    try:
        # Try evaluating as a pure expression first (e.g. df.sort_values(by='Customer Id', ascending=False))
        result = eval(code, {"pd": pd, "df": df}, local_vars)
        return result, code, None
    except SyntaxError:
        # If it's a statement (like an assignment rule), execute it
        try:
            exec(code, {"pd": pd, "df": df}, local_vars)
            
            # Retrieve the most recently assigned custom variable
            assigned_vars = [v for k, v in local_vars.items() if not k.startswith('_')]
            if assigned_vars:
                return assigned_vars[-1], code, None
            
            # If no variable was assigned, assume an inplace modification happened and return df
            return df, code, None
        except Exception as e:
            return None, code, f"Execution error: {str(e)}"
    except Exception as e:
        return None, code, f"Evaluation error: {str(e)}"


# ──────────────────────────────────────────────
# SMART CHART TYPE SELECTION (Feature 5)
# ──────────────────────────────────────────────

def _detect_chart_type(result_data, query_hint=""):
    """
    Intelligently determine the best chart type based on data characteristics.
    Returns: 'bar', 'line', 'pie', 'scatter', 'heatmap', 'horizontal_bar', 'stacked_bar'
    """
    query_lower = query_hint.lower() if query_hint else ""

    # Explicit user intent overrides
    if any(k in query_lower for k in ["pie", "breakdown", "share", "proportion", "distribution of"]):
        return "pie"
    if any(k in query_lower for k in ["scatter", "correlation", "relationship"]):
        return "scatter"
    if any(k in query_lower for k in ["heatmap", "heat map", "matrix"]):
        return "heatmap"
    if any(k in query_lower for k in ["trend", "over time", "timeline", "monthly", "daily", "weekly", "yearly"]):
        return "line"

    if isinstance(result_data, pd.Series):
        n = len(result_data)
        if n <= 7 and n >= 2:
            # Good candidate for pie if all positive
            if (result_data > 0).all():
                return "pie"
        if n <= 20:
            return "bar"
        return "line"

    elif isinstance(result_data, pd.DataFrame):
        numeric_cols = result_data.select_dtypes(include="number").columns
        n_rows = len(result_data)
        n_numeric = len(numeric_cols)

        # Scatter: 2+ numeric columns, reasonable row count
        if n_numeric >= 2 and n_rows > 5 and "scatter" in query_lower:
            return "scatter"

        # Heatmap: pivot-like structure (many cols and rows)
        if n_rows >= 3 and n_numeric >= 3:
            return "heatmap"

        # Stacked bar: multiple numeric columns with categories
        if n_numeric >= 2 and n_rows <= 15:
            return "stacked_bar"

        if n_rows <= 20:
            return "bar"
        return "line"

    return "bar"


# ──────────────────────────────────────────────
# CHART GENERATORS
# ──────────────────────────────────────────────

def _create_bar_chart(ax, data, title=""):
    """Create a styled bar chart."""
    if isinstance(data, pd.Series):
        bars = ax.bar(
            range(len(data)), data.values,
            color=COLORS[:len(data)], width=0.6, 
            edgecolor="none", alpha=0.9
        )
        ax.set_xticks(range(len(data)))
        ax.set_xticklabels([str(x)[:20] for x in data.index], fontsize=8)
        # Add value labels on bars
        for bar, val in zip(bars, data.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01 * data.max(),
                _format_number(val), ha="center", va="bottom",
                color=TEXT_COLOR, fontsize=8, fontweight="bold"
            )
    elif isinstance(data, pd.DataFrame):
        numeric_cols = data.select_dtypes(include="number").columns
        data[numeric_cols].plot(
            kind="bar", ax=ax, color=COLORS[:len(numeric_cols)],
            width=0.7, edgecolor="none", alpha=0.9
        )
        ax.legend(facecolor=CARD_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=8)
    ax.set_title(title or "Data Comparison", color=TEXT_COLOR, fontsize=12, fontweight="bold", pad=15)


def _create_line_chart(ax, data, title=""):
    """Create a styled line chart with gradient fill."""
    if isinstance(data, pd.Series):
        ax.plot(range(len(data)), data.values, color=ACCENT, linewidth=2.5, marker="o", markersize=4)
        ax.fill_between(range(len(data)), data.values, alpha=0.15, color=ACCENT)
        ax.set_xticks(range(0, len(data), max(1, len(data)//10)))
        tick_labels = [str(data.index[i])[:15] for i in range(0, len(data), max(1, len(data)//10))]
        ax.set_xticklabels(tick_labels, fontsize=7)
    elif isinstance(data, pd.DataFrame):
        numeric_cols = data.select_dtypes(include="number").columns
        for i, col in enumerate(numeric_cols):
            color = COLORS[i % len(COLORS)]
            ax.plot(range(len(data)), data[col].values, color=color, linewidth=2, label=col, marker="o", markersize=3)
            ax.fill_between(range(len(data)), data[col].values, alpha=0.08, color=color)
        ax.legend(facecolor=CARD_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=8)
    ax.set_title(title or "Trend Analysis", color=TEXT_COLOR, fontsize=12, fontweight="bold", pad=15)


def _create_pie_chart(ax, data, title=""):
    """Create a styled pie chart with modern aesthetics."""
    if isinstance(data, pd.DataFrame):
        numeric_cols = data.select_dtypes(include="number").columns
        if len(numeric_cols) > 0:
            data = data[numeric_cols[0]]
        else:
            return

    # Limit to top 8 slices, group rest as "Other"
    if len(data) > 8:
        top = data.nlargest(7)
        other = pd.Series({"Other": data.sum() - top.sum()})
        data = pd.concat([top, other])

    labels = [str(x)[:18] for x in data.index]
    colors_used = COLORS[:len(data)]

    wedges, texts, autotexts = ax.pie(
        data.values, labels=labels, colors=colors_used,
        autopct="%1.1f%%", pctdistance=0.75, startangle=140,
        textprops={"color": TEXT_COLOR, "fontsize": 9},
        wedgeprops={"edgecolor": BG_COLOR, "linewidth": 2},
    )
    for autotext in autotexts:
        autotext.set_fontsize(8)
        autotext.set_fontweight("bold")
    ax.set_title(title or "Breakdown", color=TEXT_COLOR, fontsize=12, fontweight="bold", pad=15)


def _create_scatter_chart(ax, data, title=""):
    """Create a styled scatter plot."""
    if isinstance(data, pd.DataFrame):
        numeric_cols = data.select_dtypes(include="number").columns
        if len(numeric_cols) >= 2:
            ax.scatter(
                data[numeric_cols[0]], data[numeric_cols[1]],
                c=ACCENT, alpha=0.7, s=50, edgecolors="white", linewidth=0.5
            )
            ax.set_xlabel(numeric_cols[0], fontsize=10)
            ax.set_ylabel(numeric_cols[1], fontsize=10)
    ax.set_title(title or "Correlation", color=TEXT_COLOR, fontsize=12, fontweight="bold", pad=15)


def _create_heatmap(ax, data, title=""):
    """Create a styled heatmap."""
    if isinstance(data, pd.DataFrame):
        numeric_data = data.select_dtypes(include="number")
        if numeric_data.empty:
            return
        im = ax.imshow(numeric_data.values, cmap="RdYlBu_r", aspect="auto")
        ax.set_xticks(range(len(numeric_data.columns)))
        ax.set_xticklabels([str(c)[:12] for c in numeric_data.columns], fontsize=7)
        ax.set_yticks(range(len(numeric_data)))
        y_labels = [str(data.index[i])[:12] for i in range(len(numeric_data))]
        ax.set_yticklabels(y_labels, fontsize=7)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(colors=TEXT_COLOR, labelsize=7)
    ax.set_title(title or "Heatmap", color=TEXT_COLOR, fontsize=12, fontweight="bold", pad=15)


def _create_stacked_bar(ax, data, title=""):
    """Create a styled stacked bar chart."""
    if isinstance(data, pd.DataFrame):
        numeric_cols = data.select_dtypes(include="number").columns
        data[numeric_cols].plot(
            kind="bar", stacked=True, ax=ax,
            color=COLORS[:len(numeric_cols)], width=0.7,
            edgecolor="none", alpha=0.9
        )
        ax.legend(facecolor=CARD_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=8)
    ax.set_title(title or "Stacked Breakdown", color=TEXT_COLOR, fontsize=12, fontweight="bold", pad=15)


# ──────────────────────────────────────────────
# MAIN CHART GENERATOR
# ──────────────────────────────────────────────

def generate_auto_chart(result_data, query_hint="", title=""):
    """
    Takes a pandas result and automatically generates the best chart.
    Returns a matplotlib Figure or None if unplottable.
    """
    if result_data is None:
        return None

    # Scalar values aren't plottable
    if not isinstance(result_data, (pd.DataFrame, pd.Series)):
        return None

    if isinstance(result_data, pd.DataFrame) and result_data.empty:
        return None

    if isinstance(result_data, pd.Series) and len(result_data) < 2:
        return None

    chart_type = _detect_chart_type(result_data, query_hint)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    try:
        chart_generators = {
            "bar": _create_bar_chart,
            "line": _create_line_chart,
            "pie": _create_pie_chart,
            "scatter": _create_scatter_chart,
            "heatmap": _create_heatmap,
            "stacked_bar": _create_stacked_bar,
            "horizontal_bar": _create_bar_chart,
        }

        generator = chart_generators.get(chart_type, _create_bar_chart)
        generator(ax, result_data, title=title)
        _apply_dark_theme(fig, ax)

        plt.tight_layout(pad=2.0)
        return fig

    except Exception as e:
        print(f"Chart generation failed ({chart_type}): {e}")
        plt.close(fig)
        return None


# ──────────────────────────────────────────────
# COMPARISON CHART (Feature 9)
# ──────────────────────────────────────────────

def generate_comparison_chart(comparison_data, metric_name="Value"):
    """
    Generate a side-by-side comparison bar chart from compare_periods() output.
    """
    if not comparison_data or "error" in comparison_data:
        return None

    fig, ax = plt.subplots(figsize=(9, 5.5))

    pa = comparison_data["period_a"]
    pb = comparison_data["period_b"]

    categories = ["Sum", "Average", "Min", "Max", "Count"]
    vals_a = [pa["sum"], pa["mean"], pa["min"], pa["max"], pa["count"]]
    vals_b = [pb["sum"], pb["mean"], pb["min"], pb["max"], pb["count"]]

    x = np.arange(len(categories))
    width = 0.35

    bars_a = ax.bar(x - width/2, vals_a, width, label=pa["period"], color=COLORS[0], alpha=0.9, edgecolor="none")
    bars_b = ax.bar(x + width/2, vals_b, width, label=pb["period"], color=COLORS[1], alpha=0.9, edgecolor="none")

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(facecolor=CARD_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=9)

    change_text = f"{comparison_data['direction'].title()}: {comparison_data['change_pct']:+.1f}%"
    color = "#0ABAB5" if comparison_data["change"] >= 0 else "#E94560"
    ax.text(
        0.98, 0.95, change_text, transform=ax.transAxes,
        ha="right", va="top", fontsize=11, fontweight="bold",
        color=color, bbox=dict(boxstyle="round,pad=0.4", facecolor=CARD_BG, edgecolor=color, alpha=0.9)
    )

    ax.set_title(f"Comparison: {metric_name}", color=TEXT_COLOR, fontsize=13, fontweight="bold", pad=15)
    _apply_dark_theme(fig, ax)
    plt.tight_layout(pad=2.0)
    return fig


# ──────────────────────────────────────────────
# ROOT CAUSE / BREAKDOWN CHART (Features 8, 10)
# ──────────────────────────────────────────────

def generate_driver_chart(drivers, title="Top Drivers of Change"):
    """
    Generate a horizontal bar chart showing top drivers from root cause analysis.
    """
    if not drivers:
        return None

    fig, ax = plt.subplots(figsize=(9, max(4, len(drivers) * 0.5)))

    labels = [f"{d['dimension']}: {d['segment']}" for d in drivers[:10]]
    values = [d["contribution_pct"] for d in drivers[:10]]
    bar_colors = [COLORS[1] if v < 0 else COLORS[0] for v in values]

    labels.reverse()
    values.reverse()
    bar_colors.reverse()

    bars = ax.barh(range(len(labels)), values, color=bar_colors, height=0.6, edgecolor="none", alpha=0.9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Contribution to Change (%)", fontsize=10)

    # Add value labels
    for bar, val in zip(bars, values):
        x_pos = bar.get_width() + 0.5 if val >= 0 else bar.get_width() - 0.5
        ax.text(
            x_pos, bar.get_y() + bar.get_height() / 2,
            f"{val:+.1f}%", ha="left" if val >= 0 else "right",
            va="center", color=TEXT_COLOR, fontsize=8, fontweight="bold"
        )

    ax.axvline(x=0, color=GRID_COLOR, linewidth=0.8)
    ax.set_title(title, color=TEXT_COLOR, fontsize=12, fontweight="bold", pad=15)
    _apply_dark_theme(fig, ax)
    plt.tight_layout(pad=2.0)
    return fig


# ──────────────────────────────────────────────
# ANOMALY CHART (Feature 17)
# ──────────────────────────────────────────────

def generate_anomaly_chart(df, col, anomaly_info):
    """
    Generate a chart highlighting anomalies in a specific column.
    """
    if col not in df.columns:
        return None

    fig, ax = plt.subplots(figsize=(9, 5))

    values = df[col].values
    x = range(len(values))

    # Plot normal points
    ax.scatter(x, values, c=COLORS[0], s=15, alpha=0.5, label="Normal", zorder=2)

    # Highlight anomalies
    anomaly_indices = anomaly_info.get("indices", [])
    if anomaly_indices:
        anomaly_vals = df.loc[[i for i in anomaly_indices if i in df.index], col].values
        ax.scatter(
            [i for i in anomaly_indices if i in df.index], anomaly_vals,
            c=COLORS[1], s=50, alpha=0.9, marker="X",
            label=f"Anomalies ({len(anomaly_indices)})", zorder=3, edgecolors="white", linewidth=0.5
        )

    # Bounds
    ax.axhline(y=anomaly_info["upper_bound"], color=COLORS[1], linestyle="--", alpha=0.5, label="Upper Bound")
    ax.axhline(y=anomaly_info["lower_bound"], color=COLORS[1], linestyle="--", alpha=0.5, label="Lower Bound")

    ax.legend(facecolor=CARD_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=8)
    ax.set_xlabel("Row Index", fontsize=10)
    ax.set_ylabel(col, fontsize=10)
    ax.set_title(f"Anomaly Detection: {col}", color=TEXT_COLOR, fontsize=12, fontweight="bold", pad=15)
    _apply_dark_theme(fig, ax)
    plt.tight_layout(pad=2.0)
    return fig


# ──────────────────────────────────────────────
# TEXT HELPER (preserved from original)
# ──────────────────────────────────────────────

def get_auto_explanation(result_data):
     """ Generates an automatic simple english string describing the table state natively without LLM."""
     if isinstance(result_data, pd.Series):
          return f"Computed a series with {len(result_data)} rows. Index: {result_data.index.name or 'values'}."
     elif isinstance(result_data, pd.DataFrame):
          return f"Computed a Dataframe with {len(result_data)} rows and {len(result_data.columns)} columns."
     else:
          return f"Calculated value: {result_data}"
