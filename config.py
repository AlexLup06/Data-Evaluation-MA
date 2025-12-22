import json
from copy import deepcopy
from typing import Any, Dict

# Default schema paths for the supported diagram types.
DEFAULT_SCHEMAS: Dict[str, str] = {
    "heatmap": "schema/heatmap.schema.json",
    "line": "schema/line.schema.json",
    "parallel": "schema/parallel-coordinates.json",
}

DEFAULT_STYLE: Dict[str, Dict[str, Any]] = {
    # Heatmap fonts
    "heatmap": {
        "title": 13,
        "axis_label": 12,
        "tick": 10,
        "grid_title": 12,
        "grid_axis_label": 11,
        "grid_tick": 9,
        "colorbar_label": 11,
        "meta": 8,
    },
    # Line plot fonts
    "line": {
        "title": 13,
        "axis_label": 12,
        "tick": 10,
        "legend": 10,
    },
    # Parallel coordinates fonts
    "parallel": {
        "title": 13,
        "axis_label": 12,
        "tick": 10,
        "legend": 10,
    },
}


def load_config(path: str) -> Dict[str, Any]:
    """Load a diagram config file.

    The config must include at least:
      - diagram: one of the supported diagram types
      - data: path to the data JSON file, or a list of paths for facet grids
    Optional fields:
      - outdir: output directory (defaults to 'outputs')
      - schema: path to a JSON schema (defaults to DEFAULT_SCHEMAS[diagram])
      - params: dict of diagram-specific parameters (e.g., columns or bins)
    """
    with open(path, "r") as f:
        cfg = json.load(f)

    if "diagram" not in cfg or "data" not in cfg:
        raise ValueError("Config must include 'diagram' and 'data' fields.")

    return cfg


def merge_style(user_style: Dict[str, Any] | None) -> Dict[str, Any]:
    """Deep-merge user style overrides into defaults."""
    merged = deepcopy(DEFAULT_STYLE)
    if not user_style:
        return merged
    for section, values in user_style.items():
        if section not in merged or not isinstance(values, dict):
            merged[section] = deepcopy(values)
            continue
        for key, val in values.items():
            merged[section][key] = val
    return merged
