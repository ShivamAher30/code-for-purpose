"""
analysis_engine.py — Intelligent Analysis Module for Talk-to-Data
Handles: Root Cause Analysis, Comparisons, Breakdowns, Anomaly Detection,
         Auto Summaries, and Insight Ranking.
All functions are pure data operations (Pandas/NumPy). LLM narration is separate.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# ──────────────────────────────────────────────
# HELPER: Identify column types
# ──────────────────────────────────────────────

def _get_column_types(df):
    """Classify columns into numeric, categorical, and datetime."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    datetime_cols = []
    categorical_cols = []

    for col in df.columns:
        if col in numeric_cols:
            continue
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            datetime_cols.append(col)
        else:
            # Try to parse as date
            try:
                sample = df[col].dropna().head(20)
                if len(sample) > 0:
                    pd.to_datetime(sample)
                    datetime_cols.append(col)
                    continue
            except (ValueError, TypeError):
                pass
            # Check cardinality for categorical
            if df[col].nunique() < min(50, len(df) * 0.5):
                categorical_cols.append(col)

    return numeric_cols, categorical_cols, datetime_cols


# ──────────────────────────────────────────────
# FEATURE 8: Root Cause Analysis
# ──────────────────────────────────────────────

def root_cause_analysis(df, metric_col, period_col=None, current_label=None, previous_label=None):
    """
    Identify top drivers of a metric change by decomposing across all
    categorical dimensions.
    
    Returns a list of dicts: [{"dimension": col, "segment": val, "current": x, "previous": y, "change": delta, "contribution_pct": pct}]
    """
    numeric_cols, categorical_cols, datetime_cols = _get_column_types(df)

    if metric_col not in numeric_cols:
        return {"error": f"'{metric_col}' is not a numeric column."}

    # If period_col provided, split into two periods
    if period_col and period_col in df.columns:
        if period_col in datetime_cols or pd.api.types.is_datetime64_any_dtype(df[period_col]):
            df[period_col] = pd.to_datetime(df[period_col], errors="coerce")
            median_date = df[period_col].median()
            df_prev = df[df[period_col] < median_date]
            df_curr = df[df[period_col] >= median_date]
        else:
            unique_vals = df[period_col].dropna().unique()
            if len(unique_vals) >= 2:
                if current_label and previous_label:
                    df_curr = df[df[period_col] == current_label]
                    df_prev = df[df[period_col] == previous_label]
                else:
                    df_prev = df[df[period_col] == unique_vals[0]]
                    df_curr = df[df[period_col] == unique_vals[-1]]
            else:
                mid = len(df) // 2
                df_prev, df_curr = df.iloc[:mid], df.iloc[mid:]
    else:
        mid = len(df) // 2
        df_prev, df_curr = df.iloc[:mid], df.iloc[mid:]

    total_prev = df_prev[metric_col].sum()
    total_curr = df_curr[metric_col].sum()
    total_change = total_curr - total_prev

    if total_change == 0:
        return {"message": "No change detected in the metric.", "total_previous": total_prev, "total_current": total_curr}

    drivers = []

    for col in categorical_cols:
        if col == period_col or col == metric_col:
            continue
        prev_group = df_prev.groupby(col)[metric_col].sum()
        curr_group = df_curr.groupby(col)[metric_col].sum()
        all_segments = set(prev_group.index) | set(curr_group.index)

        for seg in all_segments:
            prev_val = prev_group.get(seg, 0)
            curr_val = curr_group.get(seg, 0)
            change = curr_val - prev_val
            contribution = (change / total_change * 100) if total_change != 0 else 0

            drivers.append({
                "dimension": col,
                "segment": str(seg),
                "previous": round(float(prev_val), 2),
                "current": round(float(curr_val), 2),
                "change": round(float(change), 2),
                "contribution_pct": round(float(contribution), 1),
            })

    # Sort by absolute contribution
    drivers.sort(key=lambda x: abs(x["contribution_pct"]), reverse=True)

    return {
        "total_previous": round(float(total_prev), 2),
        "total_current": round(float(total_curr), 2),
        "total_change": round(float(total_change), 2),
        "total_change_pct": round(float((total_change / total_prev * 100) if total_prev != 0 else 0), 1),
        "top_drivers": drivers[:15],  # Top 15 drivers
    }


# ──────────────────────────────────────────────
# FEATURE 9: Comparison Engine
# ──────────────────────────────────────────────

def compare_periods(df, metric_col, period_col, period_a, period_b):
    """
    Compare a metric between two specific period values.
    Returns comparison dict with aggregated stats.
    """
    if period_col not in df.columns:
        return {"error": f"Column '{period_col}' not found."}
    if metric_col not in df.columns:
        return {"error": f"Column '{metric_col}' not found."}

    df_a = df[df[period_col] == period_a]
    df_b = df[df[period_col] == period_b]

    if df_a.empty and df_b.empty:
        return {"error": f"No data found for either '{period_a}' or '{period_b}'."}

    stats_a = {
        "period": str(period_a),
        "count": int(len(df_a)),
        "sum": round(float(df_a[metric_col].sum()), 2),
        "mean": round(float(df_a[metric_col].mean()), 2) if not df_a.empty else 0,
        "min": round(float(df_a[metric_col].min()), 2) if not df_a.empty else 0,
        "max": round(float(df_a[metric_col].max()), 2) if not df_a.empty else 0,
    }
    stats_b = {
        "period": str(period_b),
        "count": int(len(df_b)),
        "sum": round(float(df_b[metric_col].sum()), 2),
        "mean": round(float(df_b[metric_col].mean()), 2) if not df_b.empty else 0,
        "min": round(float(df_b[metric_col].min()), 2) if not df_b.empty else 0,
        "max": round(float(df_b[metric_col].max()), 2) if not df_b.empty else 0,
    }

    change = stats_b["sum"] - stats_a["sum"]
    change_pct = (change / stats_a["sum"] * 100) if stats_a["sum"] != 0 else 0

    return {
        "period_a": stats_a,
        "period_b": stats_b,
        "change": round(change, 2),
        "change_pct": round(change_pct, 1),
        "direction": "increased" if change > 0 else "decreased" if change < 0 else "unchanged",
    }


def compare_segments(df, metric_col, group_col):
    """
    Compare a metric across all segments of a categorical column.
    Returns a sorted breakdown.
    """
    if group_col not in df.columns or metric_col not in df.columns:
        return {"error": "Column not found."}

    grouped = df.groupby(group_col)[metric_col].agg(["sum", "mean", "count"]).reset_index()
    grouped.columns = [group_col, "total", "average", "count"]
    grouped = grouped.sort_values("total", ascending=False)
    grand_total = grouped["total"].sum()
    grouped["share_pct"] = (grouped["total"] / grand_total * 100).round(1) if grand_total != 0 else 0

    return {
        "metric": metric_col,
        "grouped_by": group_col,
        "grand_total": round(float(grand_total), 2),
        "segments": grouped.to_dict("records"),
    }


# ──────────────────────────────────────────────
# FEATURE 10: Breakdown / Decomposition
# ──────────────────────────────────────────────

def breakdown_metric(df, metric_col, group_cols=None):
    """
    Decompose a metric across one or more categorical columns.
    If group_cols not specified, auto-detect categorical columns.
    Returns breakdown per dimension with outlier flags.
    """
    numeric_cols, categorical_cols, _ = _get_column_types(df)

    if metric_col not in numeric_cols:
        return {"error": f"'{metric_col}' is not numeric."}

    if group_cols is None:
        group_cols = categorical_cols[:5]  # Limit to top 5

    grand_total = df[metric_col].sum()
    breakdowns = {}

    for col in group_cols:
        if col not in df.columns or col == metric_col:
            continue
        grouped = df.groupby(col)[metric_col].agg(["sum", "mean", "count"]).reset_index()
        grouped.columns = [col, "total", "average", "count"]
        grouped = grouped.sort_values("total", ascending=False)
        grouped["share_pct"] = (grouped["total"] / grand_total * 100).round(1) if grand_total != 0 else 0

        # Flag outliers (segments contributing disproportionately)
        mean_share = 100 / len(grouped) if len(grouped) > 0 else 0
        grouped["is_outlier"] = grouped["share_pct"].apply(
            lambda x: abs(x - mean_share) > mean_share * 1.5
        )

        breakdowns[col] = {
            "segments": grouped.to_dict("records"),
            "top_contributor": grouped.iloc[0].to_dict() if len(grouped) > 0 else None,
            "num_segments": len(grouped),
        }

    return {
        "metric": metric_col,
        "grand_total": round(float(grand_total), 2),
        "breakdowns": breakdowns,
    }


# ──────────────────────────────────────────────
# FEATURE 11: Auto Summary Generator
# ──────────────────────────────────────────────

def generate_data_summary(df):
    """
    Generate a structured summary of the dataset for LLM narration.
    Returns a dict with key statistics, trends, and notable patterns.
    """
    numeric_cols, categorical_cols, datetime_cols = _get_column_types(df)

    summary = {
        "shape": {"rows": len(df), "columns": len(df.columns)},
        "columns": list(df.columns),
        "numeric_summary": {},
        "categorical_summary": {},
        "date_range": {},
        "missing_data": {},
        "notable_patterns": [],
    }

    # Numeric stats
    for col in numeric_cols:
        col_data = df[col].dropna()
        if len(col_data) == 0:
            continue
        summary["numeric_summary"][col] = {
            "mean": round(float(col_data.mean()), 2),
            "median": round(float(col_data.median()), 2),
            "std": round(float(col_data.std()), 2),
            "min": round(float(col_data.min()), 2),
            "max": round(float(col_data.max()), 2),
            "total": round(float(col_data.sum()), 2),
        }
        # Skewness check
        skew = col_data.skew()
        if abs(skew) > 1.5:
            summary["notable_patterns"].append(
                f"'{col}' is highly {'right' if skew > 0 else 'left'}-skewed (skewness={round(skew,2)})"
            )

    # Categorical stats
    for col in categorical_cols[:8]:
        value_counts = df[col].value_counts()
        summary["categorical_summary"][col] = {
            "unique_values": int(df[col].nunique()),
            "top_value": str(value_counts.index[0]) if len(value_counts) > 0 else None,
            "top_count": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
            "distribution": value_counts.head(5).to_dict(),
        }

    # Date range
    for col in datetime_cols:
        try:
            dates = pd.to_datetime(df[col], errors="coerce").dropna()
            if len(dates) > 0:
                summary["date_range"][col] = {
                    "start": str(dates.min().date()),
                    "end": str(dates.max().date()),
                    "span_days": int((dates.max() - dates.min()).days),
                }
        except Exception:
            pass

    # Missing data
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(1)
    for col in df.columns:
        if missing[col] > 0:
            summary["missing_data"][col] = {
                "count": int(missing[col]),
                "percent": float(missing_pct[col]),
            }

    return summary


# ──────────────────────────────────────────────
# FEATURE 17: Anomaly Detection
# ──────────────────────────────────────────────

def detect_anomalies(df, columns=None, method="iqr", threshold=1.5):
    """
    Detect anomalies in numeric columns using IQR or Z-score method.
    Returns dict with anomaly info per column.
    """
    numeric_cols, _, _ = _get_column_types(df)
    if columns:
        numeric_cols = [c for c in columns if c in numeric_cols]

    anomalies = {}

    for col in numeric_cols:
        col_data = df[col].dropna()
        if len(col_data) < 5:
            continue

        if method == "iqr":
            q1 = col_data.quantile(0.25)
            q3 = col_data.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - threshold * iqr
            upper = q3 + threshold * iqr
            mask = (df[col] < lower) | (df[col] > upper)
        else:  # z-score
            mean = col_data.mean()
            std = col_data.std()
            if std == 0:
                continue
            z_scores = ((df[col] - mean) / std).abs()
            mask = z_scores > threshold
            lower = mean - threshold * std
            upper = mean + threshold * std

        anomaly_indices = df[mask].index.tolist()
        if len(anomaly_indices) > 0:
            anomalies[col] = {
                "count": len(anomaly_indices),
                "percent": round(len(anomaly_indices) / len(df) * 100, 1),
                "lower_bound": round(float(lower), 2),
                "upper_bound": round(float(upper), 2),
                "anomaly_values": df.loc[anomaly_indices[:10], col].tolist(),  # Top 10
                "indices": anomaly_indices[:20],
                "direction": {
                    "high": int(df[mask & (df[col] > upper)].shape[0]) if upper is not None else 0,
                    "low": int(df[mask & (df[col] < lower)].shape[0]) if lower is not None else 0,
                },
            }

    return {
        "method": method,
        "threshold": threshold,
        "total_rows": len(df),
        "anomalies": anomalies,
        "has_anomalies": len(anomalies) > 0,
    }


# ──────────────────────────────────────────────
# FEATURE 18: Insight Ranking
# ──────────────────────────────────────────────

def rank_insights(insights_list):
    """
    Rank a list of insight dicts by importance.
    Each insight should have: {"text": str, "magnitude": float, "type": str}
    Returns sorted list with rank field added.
    """
    # Score by magnitude (larger changes = more important)
    scored = []
    for insight in insights_list:
        score = abs(insight.get("magnitude", 0))
        # Boost certain types
        type_boost = {
            "anomaly": 2.0,
            "trend_reversal": 1.8,
            "significant_change": 1.5,
            "breakdown": 1.2,
            "comparison": 1.0,
            "summary": 0.8,
        }
        score *= type_boost.get(insight.get("type", ""), 1.0)
        insight["score"] = round(score, 2)
        scored.append(insight)

    scored.sort(key=lambda x: x["score"], reverse=True)
    for i, insight in enumerate(scored):
        insight["rank"] = i + 1

    return scored
