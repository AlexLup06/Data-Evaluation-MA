import json
from typing import Any, Dict

# Default schema paths for the supported diagram types.
DEFAULT_SCHEMAS: Dict[str, str] = {
    "heatmap": "schema/heatmap.schema.json",
    "line": "schema/line.schema.json",
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
