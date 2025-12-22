import os
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Optional
from utils.naming import build_output_filename


def plot_parallel_coordinates(
    dataframe: pd.DataFrame,
    metrics_order: List[str],
    output_path: str,
    monochrome: bool = False,
    filename_tag: Optional[str] = None,
    freeze_nodes: Optional[float] = None,
    freeze_ttm: Optional[float] = None,
    config_name: Optional[str] = None,
    style: Optional[dict] = None,
    width_factor: float = 1.0,
):
    if dataframe.empty:
        raise ValueError("No data provided for parallel coordinates plot.")

    os.makedirs(output_path, exist_ok=True)

    try:
        factor = float(width_factor)
    except Exception:
        factor = 1.0
    factor = max(0.1, factor)

    plt.figure(figsize=(10 * factor, 6), dpi=300)
    ax = plt.gca()
    x_positions = list(range(len(metrics_order)))

    default_colors = plt.rcParams.get("axes.prop_cycle", None)
    color_cycle = default_colors.by_key().get("color", []) if default_colors else []
    if not color_cycle:
        color_cycle = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]
    styles = ["solid", "dashed", "dashdot", "dotted"]
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]

    unique_protocols = list(dataframe["Protocol"].unique())
    for idx, proto in enumerate(unique_protocols):
        row = dataframe[dataframe["Protocol"] == proto].iloc[0]
        y_vals = [row[m] for m in metrics_order]
        color = "black" if monochrome else color_cycle[idx % len(color_cycle)]
        linestyle = styles[idx % len(styles)]
        marker = markers[idx % len(markers)]
        ax.plot(
            x_positions,
            y_vals,
            label=proto,
            color=color,
            linestyle=linestyle,
            linewidth=2,
            marker=marker,
            markersize=5,
            alpha=0.9,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(metrics_order, rotation=0)

    pstyle = (style or {}).get("parallel", {})
    title_size = pstyle.get("title", 13)
    axis_label_size = pstyle.get("axis_label", 12)
    tick_size = pstyle.get("tick", 10)
    legend_size = pstyle.get("legend", 10)

    plt.ylabel("", fontsize=axis_label_size)
    plt.xlabel("", fontsize=axis_label_size)
    plt.title("", fontsize=title_size)
    plt.legend(fontsize=legend_size)
    plt.grid(True, alpha=0.2)
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)
    plt.tight_layout()

    metric_label = "metrics"
    filename = build_output_filename("parallel", filename_tag, metric_label, config_name)
    plt.savefig(os.path.join(output_path, filename))
    plt.close()
