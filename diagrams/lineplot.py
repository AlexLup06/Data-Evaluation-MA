import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from utils.naming import build_output_filename


def _axis_label(name: str) -> str:
    mapping = {
        "timetonextmission": "Mission Interval (in seconds)",
        "numbernodes": "Number of Nodes",
    }
    return mapping.get(str(name).lower(), name)


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
    config_name: Optional[str] = None,
    filename_tag: Optional[str] = None,
    style: Optional[dict] = None,
    ordinal_x: bool = False,
    x_order: Optional[list] = None,
):
    df = dataframe.copy()
    if filter_column and filter_column in df.columns and filter_value is not None:
        df = df[df[filter_column] == filter_value]

    if ordinal_x:
        labels = []
        mapping = {}
        if x_order:
            for val in x_order:
                s = str(val)
                if s not in mapping:
                    mapping[s] = len(labels)
                    labels.append(s)
        x_series = df[x_column].apply(str)
        for val in x_series:
            if val not in mapping:
                mapping[val] = len(labels)
                labels.append(val)
        df = df.assign(_x_pos=x_series.map(mapping))
        df = df.sort_values(by="_x_pos")
    else:
        df = df.sort_values(by=x_column)

    plt.figure(figsize=(10, 6), dpi=300)
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    styles = ["solid", "dashed", "dashdot", "dotted"]
    style_idx = 0
    marker_idx = 0
    if hue_column and hue_column in df.columns:
        for name, group in df.groupby(hue_column):
            group_sorted = group.sort_values(by="_x_pos" if ordinal_x else x_column)
            kwargs = {}
            if monochrome:
                kwargs["linestyle"] = styles[style_idx % len(styles)]
                kwargs["color"] = "black"
            else:
                kwargs["linestyle"] = styles[style_idx % len(styles)]
            marker = markers[marker_idx % len(markers)]
            x_vals = group_sorted["_x_pos"] if ordinal_x else group_sorted[x_column]
            plt.plot(
                x_vals,
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
        x_vals = df["_x_pos"] if ordinal_x else df[x_column]
        plt.plot(x_vals, df[y_column], marker=marker, label=y_column, **kwargs)

    lstyle = (style or {}).get("line", {})
    title_size = lstyle.get("title", 13)
    axis_label_size = lstyle.get("axis_label", 12)
    tick_size = lstyle.get("tick", 10)
    legend_size = lstyle.get("legend", 10)

    plt.xlabel(_axis_label(x_column), fontsize=axis_label_size)
    plt.ylabel(y_column, fontsize=axis_label_size)
    plt.title(f"{y_column} vs {_axis_label(x_column)}", fontsize=title_size)
    if metadata:
        plt.suptitle(f"Simulation ID: {metadata.get('simulation_id', 'N/A')}", fontsize=8, y=0.98)
    plt.legend(fontsize=legend_size)
    plt.grid(True, alpha=0.2)
    if ordinal_x:
        plt.xticks(range(len(labels)), labels, fontsize=tick_size)
    else:
        plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)
    plt.tight_layout()
    os.makedirs(output_path, exist_ok=True)
    filename = build_output_filename("line", filename_tag, y_column, config_name)
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
    config_name: Optional[str] = None,
    style: Optional[dict] = None,
    ordinal_x: bool = False,
    x_order: Optional[list] = None,
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
    okabe_ito = [
        "#E69F00",
        "#56B4E9",
        "#009E73",
        "#F0E442",
        "#0072B2",
        "#D55E00",
        "#CC79A7",
        "#999999",
    ]

    labels = []
    mapping = {}
    if ordinal_x and x_order:
        for val in x_order:
            s = str(val)
            if s not in mapping:
                mapping[s] = len(labels)
                labels.append(s)

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
        if ordinal_x:
            x_series = df_local[x_col].apply(str)
            for val in x_series:
                if val not in mapping:
                    mapping[val] = len(labels)
                    labels.append(val)
            df_local = df_local.assign(_x_pos=x_series.map(mapping))
            df_local = df_local.sort_values(by="_x_pos")
        else:
            df_local = df_local.sort_values(by=x_col)
        kwargs = {"linestyle": styles[style_idx % len(styles)]}
        if monochrome:
            kwargs["color"] = "black"
        else:
            kwargs["color"] = okabe_ito[style_idx % len(okabe_ito)]
        marker = markers[marker_idx % len(markers)]
        label = protocol_label
        if not is_field_experiments:
            label = f"{protocol_label} | {env_label}"
        x_vals = df_local["_x_pos"] if ordinal_x else df_local[x_col]
        plt.plot(
            x_vals,
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

    lstyle = (style or {}).get("line", {})
    title_size = lstyle.get("title", 13)
    axis_label_size = lstyle.get("axis_label", 12)
    tick_size = lstyle.get("tick", 10)
    legend_size = lstyle.get("legend", 10)

    plt.xlabel(_axis_label(x_col), fontsize=axis_label_size)
    y_axis_label = (
        "Bytes per second"
        if (metric_slug or "").lower() in ("normalized-effective-throughput", "normalized-throughput")
        else ""
    )
    plt.ylabel(y_axis_label, fontsize=axis_label_size)
    plt.title(metric_label, fontsize=title_size)
    plt.legend(fontsize=legend_size)
    plt.grid(True, alpha=0.2)
    if ordinal_x:
        plt.xticks(range(len(labels)), labels, fontsize=tick_size)
    else:
        plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)
    plt.tight_layout()
    filename = build_output_filename("line", filename_tag, metric_slug, config_name)
    plt.savefig(os.path.join(output_path, filename))
    plt.close()
