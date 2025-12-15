import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional


def plot_line(
    dataframe: pd.DataFrame,
    x_column: str,
    y_column: str,
    hue_column: str | None,
    output_path: str,
    metadata=None,
    filter_column: str | None = None,
    filter_value=None,
    monochrome: bool = False,
):
    df = dataframe.copy()
    if filter_column and filter_column in df.columns and filter_value is not None:
        df = df[df[filter_column] == filter_value]

    df = df.sort_values(by=x_column)

    plt.figure(figsize=(10, 6), dpi=300)
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    styles = ["solid", "dashed", "dashdot", "dotted"]
    style_idx = 0
    marker_idx = 0
    if hue_column and hue_column in df.columns:
        for name, group in df.groupby(hue_column):
            group_sorted = group.sort_values(by=x_column)
            kwargs = {}
            if monochrome:
                kwargs["linestyle"] = styles[style_idx % len(styles)]
                kwargs["color"] = "black"
            else:
                kwargs["linestyle"] = styles[style_idx % len(styles)]
            marker = markers[marker_idx % len(markers)]
            plt.plot(
                group_sorted[x_column],
                group_sorted[y_column],
                marker=marker,
                label=f"{name}",
                **kwargs,
            )
            style_idx += 1
            marker_idx += 1
    else:
        marker = markers[marker_idx % len(markers)]
        kwargs = {"linestyle": styles[style_idx % len(styles)]}
        if monochrome:
            kwargs["color"] = "black"
        plt.plot(df[x_column], df[y_column], marker=marker, label=y_column, **kwargs)

    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.title(f"{y_column} vs {x_column}")
    if metadata:
        plt.suptitle(f"Simulation ID: {metadata.get('simulation_id', 'N/A')}", fontsize=8, y=0.98)
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    os.makedirs(output_path, exist_ok=True)
    filename = f"line_{y_column}_by_{x_column}.png"
    plt.savefig(os.path.join(output_path, filename))
    plt.close()


def plot_protocol_lines(
    datasets: List[Tuple[pd.DataFrame, dict, str, str]],
    x_col: str,
    y_col: str,
    freeze_column: str,
    freeze_value: float,
    output_path: str,
    metric_label: str,
    metric_slug: str,
    monochrome: bool = False,
    is_field_experiments: bool = False,
    filename_tag: Optional[str] = None,
    width_factor: float = 1.0,
):
    os.makedirs(output_path, exist_ok=True)
    try:
        factor = float(width_factor)
    except Exception:
        factor = 1.0
    factor = max(0.1, factor)
    plt.figure(figsize=(8 * factor, 5), dpi=300)

    styles = ["solid", "dashed", "dashdot", "dotted"]
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    style_idx = 0
    marker_idx = 0

    plotted = 0
    for df, metadata, protocol_label, env_label in datasets:
        if freeze_column not in df.columns:
            continue
        df_local = df.copy()
        # Filter to the frozen value with tolerance to floating point.
        df_local = df_local[np.isfinite(pd.to_numeric(df_local[freeze_column], errors="coerce"))]
        df_local = df_local[np.isclose(pd.to_numeric(df_local[freeze_column], errors="coerce"), float(freeze_value))]
        if df_local.empty:
            continue
        if x_col not in df_local.columns or y_col not in df_local.columns:
            continue
        df_local = df_local.sort_values(by=x_col)
        kwargs = {"linestyle": styles[style_idx % len(styles)]}
        if monochrome:
            kwargs["color"] = "black"
        marker = markers[marker_idx % len(markers)]
        label = protocol_label
        if not is_field_experiments:
            label = f"{protocol_label} | {env_label}"
        plt.plot(
            df_local[x_col],
            df_local[y_col],
            marker=marker,
            label=label,
            **kwargs,
        )
        style_idx += 1
        marker_idx += 1
        plotted += 1

    if plotted == 0:
        raise ValueError("No data available to plot after filtering for the frozen value.")

    plt.xlabel(x_col)
    y_axis_label = (
        "Bytes per second"
        if (metric_slug or "").lower() in ("normalized-effective-throughput", "normalized-throughput")
        else ""
    )
    plt.ylabel(y_axis_label)
    plt.title(metric_label)
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    fname_freeze = str(freeze_value).replace(".", "_")
    tag = f"{filename_tag}_" if filename_tag else ""
    filename = f"line_{tag}{metric_slug}_ttm-{fname_freeze}.png"
    plt.savefig(os.path.join(output_path, filename))
    plt.close()
