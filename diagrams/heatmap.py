import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import ceil
from typing import List, Tuple, Optional, Sequence


def _metadata_title(metadata: dict) -> str:
    keys = ["protocol", "dimensions", "count", "simulation_id"]
    parts = [f"{k}: {metadata[k]}" for k in keys if k in metadata]
    return " | ".join(parts)


def _compute_edges(values: np.ndarray) -> np.ndarray:
    """Compute cell edge coordinates for pcolormesh from sorted center values."""
    if len(values) == 1:
        step = values[0] * 0.05 if values[0] != 0 else 0.5
        return np.array([values[0] - step, values[0] + step])
    diffs = np.diff(values) / 2.0
    left_edge = values[0] - diffs[0]
    right_edge = values[-1] + diffs[-1]
    return np.concatenate([[left_edge], values[:-1] + diffs, [right_edge]])


def plot_heatmap(
    data,
    x_column,
    y_column,
    value_column,
    output_path,
    metadata=None,
    monochrome=False,
    ordinal_y=False,
    ordinal_x=False,
    x_order: Optional[Sequence] = None,
    y_order: Optional[Sequence] = None,
    colorbar_label: Optional[str] = None,
    y_axis_label: str = "",
    filename_slug: Optional[str] = None,
    filename_tag: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
):
    grid = _prepare_heatmap_arrays(
        data,
        x_column,
        y_column,
        value_column,
        ordinal_x=ordinal_x,
        ordinal_y=ordinal_y,
        x_order=x_order,
        y_order=y_order,
    )
    if vmin is not None or vmax is not None:
        grid["z_vals"] = np.clip(
            grid["z_vals"],
            vmin if vmin is not None else -np.inf,
            vmax if vmax is not None else np.inf,
        )

    plt.figure(figsize=(10, 6), dpi=300)
    cmap = "Greys" if monochrome else "cividis"

    mesh = plt.pcolormesh(
        grid["x_edges"],
        grid["y_edges"],
        grid["z_vals"],
        cmap=cmap,
        shading="auto",
        vmin=vmin,
        vmax=vmax,
    )
    ax = plt.gca()
    fig = plt.gcf()
    _add_top_colorbar(fig, np.array([[ax]]), mesh, colorbar_label or value_column)
    plt.xticks(grid["x_ticks"], grid["x_ticklabels"])
    plt.yticks(grid["y_ticks"], grid["y_ticklabels"])
    if y_axis_label:
        plt.ylabel(y_axis_label)
    else:
        plt.ylabel("")
    plt.title(f"Heatmap of {value_column} by {x_column} and {y_column}")
    if metadata:
        meta_text = _metadata_title(metadata)
        if meta_text:
            plt.suptitle(meta_text, fontsize=8, y=0.98)
    base_label = filename_slug or colorbar_label or value_column
    safe_label = str(base_label).replace(" ", "-").lower()
    tag = f"{filename_tag}-" if filename_tag else ""
    filename = f"heatmap_{tag}{safe_label}_by_{x_column}_and_{y_column}.png"
    plt.savefig(os.path.join(output_path, filename))
    plt.close()


def _prepare_heatmap_arrays(
    data,
    x_column: str,
    y_column: str,
    value_column: str,
    ordinal_x: bool = False,
    ordinal_y: bool = False,
    x_order: Optional[Sequence] = None,
    y_order: Optional[Sequence] = None,
):
    """Compute coordinate arrays and value grid for a dataset."""
    data = data.copy()
    # Snap values to the provided orders so bins align to requested ticks.
    def _snap_to_order(series, order):
        if not order:
            return series
        order_arr = np.array([float(v) for v in order], dtype=float)

        def _snap(val):
            try:
                v = float(val)
            except Exception:
                return np.nan
            idx = np.abs(order_arr - v).argmin()
            return order[idx]

        return series.apply(_snap)

    if x_order:
        data[x_column] = _snap_to_order(data[x_column], x_order)
    if y_order:
        data[y_column] = _snap_to_order(data[y_column], y_order)
    data[value_column] = pd.to_numeric(data[value_column], errors="coerce")
    pivot_table = data.pivot_table(index=y_column, columns=x_column, values=value_column, aggfunc="mean")
    if pivot_table.empty:
        raise ValueError("Pivot table is empty; check that the columns contain data.")

    def _merge_order(order, existing):
        if order:
            # Use exactly the provided ordering (deduped), ignoring values not listed.
            seen = set()
            ordered = []
            for v in order:
                key = float(v)
                if key in seen:
                    continue
                seen.add(key)
                ordered.append(v)
            return ordered
        return sorted(existing, key=float)

    x_labels = _merge_order(x_order, pivot_table.columns.to_numpy())
    y_labels = _merge_order(y_order, pivot_table.index.to_numpy())

    pivot_table = pivot_table.reindex(index=y_labels, columns=x_labels)

    if ordinal_x:
        x_positions = np.arange(len(x_labels), dtype=float)
        x_edges = _compute_edges(x_positions)
        x_ticks = x_positions
        x_ticklabels = [str(v) for v in x_labels]
    else:
        x_numeric = np.array([float(v) for v in x_labels], dtype=float)
        x_positions = x_numeric
        x_edges = _compute_edges(x_numeric)
        x_ticks = x_numeric
        x_ticklabels = [str(v) for v in x_numeric]

    if ordinal_y:
        y_positions = np.arange(len(y_labels), dtype=float)
        y_edges = _compute_edges(y_positions)
        y_ticks = y_positions
        y_ticklabels = [str(v) for v in y_labels]
        z_vals = pivot_table.loc[y_labels, x_labels].to_numpy()
    else:
        y_numeric = np.array([float(v) for v in y_labels], dtype=float)
        y_edges = _compute_edges(y_numeric)
        y_ticks = y_numeric
        y_ticklabels = [str(v) for v in y_numeric]
        z_vals = pivot_table.loc[y_labels, x_labels].to_numpy()

    z_vals = pivot_table.loc[y_labels, x_labels].to_numpy()
    if not np.isfinite(z_vals).any():
        raise ValueError(f"No finite values found for '{value_column}' after pivoting; check data contents.")

    return {
        "x_edges": x_edges,
        "x_ticks": x_ticks,
        "x_ticklabels": x_ticklabels,
        "y_edges": y_edges,
        "y_ticks": y_ticks,
        "y_ticklabels": y_ticklabels,
        "z_vals": z_vals,
    }


def plot_heatmap_grid(
    datasets: List[Tuple],
    x_column: str,
    y_column: str,
    value_column: str,
    output_path: str,
    monochrome: bool = False,
    ordinal_y: bool = False,
    ordinal_x: bool = False,
    facet_columns: int = 4,
    x_order: Optional[Sequence] = None,
    y_order: Optional[Sequence] = None,
    colorbar_label: Optional[str] = None,
    y_axis_label: str = "",
    filename_slug: Optional[str] = None,
    filename_tag: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
):
    """Plot multiple heatmaps in a facet grid to save space.

    datasets is a list of tuples in the shape (dataframe, metadata, label).
    """
    if not datasets:
        raise ValueError("No datasets provided for the facet grid.")

    facet_columns = max(1, facet_columns)
    n_items = len(datasets)
    n_cols = min(facet_columns, n_items)
    n_rows = ceil(n_items / n_cols)
    cmap = "Greys" if monochrome else "cividis"

    # Precompute grids to share color scale.
    grids = []
    z_min, z_max = vmin, vmax
    for entry in datasets:
        df, metadata, label = (entry[:3] if len(entry) >= 3 else entry)
        grid = _prepare_heatmap_arrays(
            df,
            x_column,
            y_column,
            value_column,
            ordinal_x=ordinal_x,
            ordinal_y=ordinal_y,
            x_order=x_order,
            y_order=y_order,
        )
        if vmin is not None or vmax is not None:
            grid["z_vals"] = np.clip(
                grid["z_vals"],
                vmin if vmin is not None else -np.inf,
                vmax if vmax is not None else np.inf,
            )
        grids.append(grid)
        z_vals = grid["z_vals"]
        if (vmin is None or vmax is None) and z_vals.size and np.isfinite(z_vals).any():
            local_min = np.nanmin(z_vals)
            local_max = np.nanmax(z_vals)
            if vmin is None:
                z_min = local_min if z_min is None else min(z_min, local_min)
            if vmax is None:
                z_max = local_max if z_max is None else max(z_max, local_max)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.2 * n_cols, 3.2 * n_rows),
        dpi=300,
        squeeze=False,
        constrained_layout=True,
    )
    all_axes = axes.flatten()
    mesh = None

    for idx, (ax, grid, entry) in enumerate(zip(all_axes, grids, datasets)):
        df, metadata, label = (entry[:3] if len(entry) >= 3 else entry)
        mesh = ax.pcolormesh(
            grid["x_edges"],
            grid["y_edges"],
            grid["z_vals"],
            cmap=cmap,
            shading="auto",
            vmin=z_min,
            vmax=z_max,
        )
        ax.set_xticks(grid["x_ticks"])
        ax.set_yticks(grid["y_ticks"])
        # Show x tick labels only on bottom row.
        is_bottom = idx // n_cols == n_rows - 1
        if is_bottom:
            ax.set_xticklabels(grid["x_ticklabels"], rotation=45, ha="right")
            ax.set_xlabel("Number of Nodes", fontsize=8)
        else:
            ax.set_xticklabels([])
        # Show y tick labels only on leftmost column.
        is_left = idx % n_cols == 0
        if is_left:
            ax.set_yticklabels(grid["y_ticklabels"])
            ax.set_ylabel(y_axis_label or "", fontsize=8)
            if y_axis_label:
                ax.yaxis.set_label_coords(-0.12, 0.5)
        else:
            ax.set_yticklabels([])
            ax.set_ylabel("")
        ax.set_title(label, fontsize=8)
        ax.tick_params(labelsize=7)

    # Hide unused axes if datasets do not fill the grid.
    for ax in all_axes[len(datasets):]:
        ax.axis("off")

    if mesh is not None:
        label = colorbar_label or value_column
        _add_top_colorbar(fig, axes, mesh, label)
    base_label = filename_slug or colorbar_label or value_column
    safe_label = str(base_label).replace(" ", "-").lower()
    tag = f"{filename_tag}-" if filename_tag else ""
    filename = f"heatmap_grid-{tag}{len(datasets)}_{safe_label}.png"
    plt.savefig(os.path.join(output_path, filename))
    plt.close(fig)


def plot_heatmap_grid_matrix(
    datasets: List[Tuple],
    x_column: str,
    y_column: str,
    value_column: str,
    output_path: str,
    row_labels: List[str],
    col_labels: List[str],
    monochrome: bool = False,
    ordinal_x: bool = False,
    ordinal_y: bool = False,
    x_order: Optional[Sequence] = None,
    y_order: Optional[Sequence] = None,
    colorbar_label: Optional[str] = None,
    split_rows: Optional[int] = None,
    y_axis_label: str = "",
    filename_slug: Optional[str] = None,
    filename_tag: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
):
    """Facet grid where rows=protocols (or similar) and cols=environments."""
    if not datasets:
        raise ValueError("No datasets provided for the facet grid.")

    n_rows = len(row_labels)
    n_cols = len(col_labels)
    if n_rows == 0 or n_cols == 0:
        raise ValueError("Row and column labels must not be empty.")

    cmap = "Greys" if monochrome else "cividis"

    grids = []
    z_min, z_max = vmin, vmax
    for df, _, _, _, _ in datasets:
        grid = _prepare_heatmap_arrays(
            df,
            x_column,
            y_column,
            value_column,
            ordinal_x=ordinal_x,
            ordinal_y=ordinal_y,
            x_order=x_order,
            y_order=y_order,
        )
        if vmin is not None or vmax is not None:
            grid["z_vals"] = np.clip(
                grid["z_vals"],
                vmin if vmin is not None else -np.inf,
                vmax if vmax is not None else np.inf,
            )
        grids.append(grid)
        z_vals = grid["z_vals"]
        if (vmin is None or vmax is None) and z_vals.size and np.isfinite(z_vals).any():
            local_min = np.nanmin(z_vals)
            local_max = np.nanmax(z_vals)
            if vmin is None:
                z_min = local_min if z_min is None else min(z_min, local_min)
            if vmax is None:
                z_max = local_max if z_max is None else max(z_max, local_max)

    # Determine row pagination if requested.
    pages = []
    split_rows_val = split_rows if split_rows and split_rows > 0 else None
    if split_rows_val and n_rows > split_rows_val:
        for i in range(0, n_rows, split_rows_val):
            pages.append(row_labels[i : i + split_rows_val])
    else:
        pages.append(row_labels)

    base_label = filename_slug or colorbar_label or value_column
    safe_label = str(base_label).replace(" ", "-").lower()
    tag = f"{filename_tag}-" if filename_tag else ""
    filename_base = f"heatmap_grid-{tag}{len(datasets)}_{safe_label}"

    for page_idx, page_rows in enumerate(pages, start=1):
        page_n_rows = len(page_rows)
        fig, axes = plt.subplots(
            page_n_rows,
            n_cols,
            figsize=(3.2 * n_cols, 3.2 * page_n_rows),
            dpi=300,
            squeeze=False,
            constrained_layout=True,
        )
        mesh = None

        row_index = {label: idx for idx, label in enumerate(page_rows)}
        col_index = {label: idx for idx, label in enumerate(col_labels)}

        for grid, entry in zip(grids, datasets):
            if len(entry) >= 5:
                df, metadata, label, row_label, col_label = entry[:5]
            elif len(entry) == 3:
                df, metadata, label = entry
                row_label = col_label = None
            else:
                raise ValueError("Dataset entries must have at least 3 fields (df, metadata, label).")
            r = row_index.get(row_label)
            c = col_index.get(col_label)
            if r is None or c is None:
                continue
            ax = axes[r][c]
            mesh = ax.pcolormesh(
                grid["x_edges"],
                grid["y_edges"],
                grid["z_vals"],
                cmap=cmap,
                shading="auto",
                vmin=z_min,
                vmax=z_max,
            )
            ax.set_xticks(grid["x_ticks"])
            ax.set_yticks(grid["y_ticks"])
            # Only bottom row gets x tick labels.
            if r == page_n_rows - 1:
                ax.set_xticklabels(grid["x_ticklabels"], rotation=45, ha="right")
                ax.set_xlabel("Number of Nodes", fontsize=8)
            else:
                ax.set_xticklabels([])
            # Only left column gets y tick labels.
            if c == 0:
                ax.set_yticklabels(grid["y_ticklabels"])
                ax.set_ylabel(y_axis_label or "", fontsize=8)
                if y_axis_label:
                    ax.yaxis.set_label_coords(-0.12, 0.5)
            else:
                ax.set_yticklabels([])
                ax.set_ylabel("")
            ax.set_title(label, fontsize=8)
            ax.tick_params(labelsize=7)

        # Hide unused axes.
        for r in range(page_n_rows):
            for c in range(n_cols):
                if axes[r][c].has_data():
                    continue
                axes[r][c].axis("off")

        if mesh is not None:
            label = colorbar_label or value_column
            _add_top_colorbar(fig, axes, mesh, label)

        filename = filename_base
        if len(pages) > 1:
            filename = f"{filename}_part{page_idx}"
        filename = f"{filename}.png"
        plt.savefig(os.path.join(output_path, filename))
        plt.close(fig)


def _add_top_colorbar(fig, axes, mesh, label: str):
    # Simple top colorbar using Matplotlib positioning to avoid overlap.
    try:
        cbar = fig.colorbar(
            mesh,
            ax=axes,
            label=label,
            fraction=0.03,
            pad=0.02,
            orientation="horizontal",
            location="top",
            shrink=0.8,
            aspect=40,
        )
    except TypeError:
        cbar = fig.colorbar(
            mesh,
            ax=axes,
            label=label,
            fraction=0.03,
            pad=0.02,
            orientation="horizontal",
            shrink=0.8,
            aspect=40,
        )
        cbar.ax.xaxis.set_ticks_position("top")
        cbar.ax.xaxis.set_label_position("top")
