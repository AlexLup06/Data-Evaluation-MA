import os
import re
from typing import Optional


def _slugify(value: Optional[str]) -> str:
    """Convert a string to a safe, lowercase filename segment."""
    if value is None:
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9_-]+", "-", text)
    text = re.sub(r"-{2,}", "-", text)
    text = re.sub(r"_{2,}", "_", text)
    return text.strip("-_")


def build_output_filename(
    plot_type: str,
    tag: Optional[str],
    metric: Optional[str],
    config_name: Optional[str],
    part: Optional[int] = None,
) -> str:
    """Compose filenames like '<plot>-<tag>-<metric>-<config>-<part>.png'."""
    segments = []
    plot_part = _slugify(plot_type) or "plot"
    segments.append(plot_part)

    tag_part = _slugify(tag)
    if tag_part:
        segments.append(tag_part)

    metric_part = _slugify(metric.replace("-", "_") if metric else metric)
    if metric_part:
        segments.append(metric_part)

    config_part = _slugify(config_name)
    if config_part:
        segments.append(config_part)

    if part is not None:
        segments.append(str(part))

    filename = "-".join(segments)
    return f"{filename}.png"


def derive_config_name(config: dict, config_path: str) -> str:
    """Prefer explicit name from config, else fall back to the config filename."""
    if "name" in config and config["name"]:
        return str(config["name"])
    if "title" in config and config["title"]:
        return str(config["title"])
    base = os.path.splitext(os.path.basename(config_path))[0]
    return base
