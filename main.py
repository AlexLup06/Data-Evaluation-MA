import argparse
import os
import pandas as pd
import numpy as np
from config import DEFAULT_SCHEMAS, load_config, merge_style
from utils.loader import load_data
from utils.validate import validate_json
from utils.naming import derive_config_name
from diagrams import heatmap, lineplot, parallel_coordinates


def get_param(params: dict, metadata: dict, key: str, default=None, required: bool = False):
    value = params.get(key, metadata.get(key, default))
    if required and value is None:
        raise ValueError(
            f"Missing required parameter '{key}' (config params or data metadata)."
        )
    return value


def _format_metric(name: str) -> str:
    label = name.replace("-", " ").replace("_", " ").strip()
    label = " ".join(word.capitalize() for word in label.split())
    label = label.replace(" Per ", " per ")
    return label


def _normalize_token(value) -> str:
    return "".join(ch.lower() for ch in str(value) if ch.isalnum())


def _topology_matches(env_label: str, field_topology: str) -> bool:
    """Allow matching when the configured topology is a prefix (e.g., FULLY vs FULLY_MESHED)."""
    env_norm = _normalize_token(env_label)
    topo_norm = _normalize_token(field_topology)
    if not env_norm or not topo_norm:
        return False
    return env_norm == topo_norm or env_norm.startswith(topo_norm) or topo_norm.startswith(env_norm)


def run_single_config(config_path: str):
    cfg = load_config(config_path)
    style_cfg = merge_style(cfg.get("style"))
    diagram = cfg["diagram"]
    data_cfg = cfg["data"]
    outdir = cfg.get("outdir", "outputs")
    schema_path = cfg.get("schema", DEFAULT_SCHEMAS.get(diagram))
    params = cfg.get("params", {})
    config_name = params.get("name") or cfg.get("name") or derive_config_name(cfg, config_path)

    if diagram != "parallel" and not schema_path:
        raise ValueError(f"No schema found for diagram type '{diagram}'.")

    os.makedirs(outdir, exist_ok=True)

    # If a metric_name is provided and the data path is a directory with metric subfolders,
    # descend into the matching metric directory for heatmaps/lines.
    if diagram in ("heatmap", "line") and isinstance(cfg["data"], str) and os.path.isdir(cfg["data"]):
        metric_param = (
            params.get("metric_name")
            or params.get("metric_label")
            or params.get("metric")
        )
        if metric_param:
            metric_slug = str(metric_param)
            candidate = os.path.join(cfg["data"], metric_slug)
            if os.path.isdir(candidate):
                data_cfg = candidate

    # Expand folder inputs for diagrams that accept directories; validation handles list inputs.
    if diagram == "parallel":
        data_paths = [data_cfg]
    elif diagram in ("heatmap", "line") and isinstance(data_cfg, str) and os.path.isdir(data_cfg):
        data_paths = [
            os.path.join(data_cfg, fname)
            for fname in sorted(os.listdir(data_cfg))
            if fname.lower().endswith(".json")
        ]
    else:
        data_paths = data_cfg if isinstance(data_cfg, list) else [data_cfg]

    if not data_paths:
        raise ValueError("No data files found for the provided configuration.")

    if diagram != "parallel":
        is_valid, message = validate_json(
            data_paths if len(data_paths) > 1 else data_paths[0], schema_path
        )
        if not is_valid:
            raise SystemExit(f"Validation failed: {message}")

    if diagram == "line":
        protocol_display = {
            "aloha": "ALOHA",
            "csma": "CSMA",
            "meshrouter": "MeshRouter",
            "rs-mitra": "RS-MiTra",
            "rsmitra": "RS-MiTra",
            "rsmitranr": "RS-MiTra-NR",
            "rs-mitra-nr": "RS-MiTra-NR",
            "rsmitranav": "RS-MiTra-NAV",
            "rs-mitra-nav": "RS-MiTra-NAV",
            "irsmitra": "I-RS-MiTra",
            "i-rs-mitra": "I-RS-MiTra",
            "mirs": "Mi-RS",
            "mi-rs": "Mi-RS",
        }
        protocol_filters = params.get("mac_protocols") or params.get("protocols")
        if isinstance(protocol_filters, str):
            protocol_filters = [protocol_filters]
        if protocol_filters:
            protocol_filters = {
                protocol_display.get(str(p).lower(), str(p)) for p in protocol_filters
            }
        env_filters = params.get("environments") or params.get("envs")
        if isinstance(env_filters, str):
            env_filters = [env_filters]
        if env_filters:
            env_filters = {str(env) for env in env_filters}
        raw_field_flag = params.get("isFieldExperiments") or params.get("is_field_experiments")
        if isinstance(raw_field_flag, str):
            is_field_experiments = raw_field_flag.lower() == "true"
        else:
            is_field_experiments = bool(raw_field_flag)
        field_topology = params.get("field_topology")

        datasets = []
        metric_label = None
        metric_slug = None
        available_protocols = set()
        available_envs = set()

        freeze_column = params.get("freeze_column", "timeToNextMission")
        freeze_value = params.get("freeze_value")

        for path in data_paths:
            df, metadata = load_data(path)
            base = os.path.splitext(os.path.basename(path))[0]
            parts = base.split("_")
            file_protocol = parts[0] if parts else None
            environment = parts[1] if len(parts) > 1 else None
            metric_raw = parts[2] if len(parts) > 2 else None

            protocol_raw = metadata.get("protocol") or file_protocol or "unknown"
            protocol_label = protocol_display.get(
                str(protocol_raw).lower(), protocol_raw
            )
            environment_label = environment or metadata.get("dimensions") or "env"

            available_protocols.add(protocol_label)
            available_envs.add(environment_label)

            if protocol_filters and protocol_label not in protocol_filters:
                continue
            if not is_field_experiments and env_filters and str(environment_label) not in env_filters:
                continue
            if is_field_experiments and field_topology:
                if not _topology_matches(environment_label, field_topology):
                    continue

            if metric_label is None:
                metric_label_param = (
                    params.get("metric_name")
                    or params.get("metric_label")
                    or params.get("metric")
                )
                if metric_label_param:
                    metric_slug = str(metric_label_param)
                    metric_label = _format_metric(metric_label_param)
                elif metric_raw:
                    metric_slug = metric_raw
                    metric_label = _format_metric(metric_raw)
                elif metadata.get("metric"):
                    metric_slug = str(metadata["metric"])
                    metric_label = _format_metric(metric_slug)

            if freeze_value is None:
                freeze_value = metadata.get(freeze_column)

            datasets.append((df, metadata, protocol_label, environment_label))

        if freeze_value is None:
            raise ValueError("Line plot requires 'freeze_value' in params or metadata.")
        freeze_value = float(freeze_value)

        if not datasets:
            raise ValueError(
                "No datasets matched the requested filters; "
                f"available protocols: {sorted(available_protocols) or 'none found'}; "
                f"available environments: {sorted(available_envs) or 'none found'}"
            )

        # Use metadata from the first selected file to resolve defaults.
        _, first_metadata, _, _ = datasets[0]
        monochrome = bool(get_param(params, first_metadata, "monochrome", default=False))
        x_col = get_param(params, first_metadata, "x_column", required=True)
        y_col = get_param(params, first_metadata, "value_column", required=True)
        x_order = params.get("x_order")
        ordinal_x = bool(get_param(params, first_metadata, "ordinal_x", default=bool(x_order)))

        lineplot.plot_protocol_lines(
            datasets,
            x_col=x_col,
            y_col=y_col,
            freeze_column=freeze_column,
            freeze_value=freeze_value,
            output_path=outdir,
            metric_label=metric_label or y_col,
            metric_slug=metric_slug or y_col,
            monochrome=monochrome,
            is_field_experiments=is_field_experiments,
            filename_tag="field" if is_field_experiments else "simu",
            config_name=config_name,
            width_factor=float(params.get("width_factor", 1.0)),
            style=style_cfg,
            ordinal_x=ordinal_x,
            x_order=x_order,
        )
    elif diagram == "parallel":
        protocol_display = {
            "aloha": "ALOHA",
            "csma": "CSMA",
            "meshrouter": "MeshRouter",
            "rs-mitra": "RS-MiTra",
            "rsmitra": "RS-MiTra",
            "rsmitranr": "RS-MiTra-NR",
            "rs-mitra-nr": "RS-MiTra-NR",
            "rsmitranav": "RS-MiTra-NAV",
            "rs-mitra-nav": "RS-MiTra-NAV",
            "irsmitra": "I-RS-MiTra",
            "i-rs-mitra": "I-RS-MiTra",
            "mirs": "Mi-RS",
            "mi-rs": "Mi-RS",
        }
        protocol_filters = params.get("mac_protocols") or params.get("protocols")
        if isinstance(protocol_filters, str):
            protocol_filters = [protocol_filters]
        if protocol_filters:
            protocol_filters = {
                protocol_display.get(str(p).lower(), str(p)) for p in protocol_filters
            }
        env_filters = params.get("environments") or params.get("envs")
        if isinstance(env_filters, str):
            env_filters = [env_filters]
        if env_filters:
            env_filters = {str(env) for env in env_filters}
        raw_field_flag = params.get("isFieldExperiments") or params.get("is_field_experiments")
        if isinstance(raw_field_flag, str):
            is_field_experiments = raw_field_flag.lower() == "true"
        else:
            is_field_experiments = bool(raw_field_flag)
        field_topology = params.get("field_topology")

        freeze_nodes = params.get("freeze_nodes") or params.get("number_nodes")
        freeze_ttm = params.get("freeze_time") or params.get("time_to_next_mission")
        if freeze_nodes is None or freeze_ttm is None:
            raise ValueError("Parallel coordinates require 'freeze_nodes' and 'freeze_time' parameters.")
        freeze_nodes = float(freeze_nodes)
        freeze_ttm = float(freeze_ttm)

        metrics_order = [
            ("Reachability", ["node-reachability", "node-reachibility"]),
            ("RSR", ["reception-success-ratio"]),
            ("NEDTPN", ["normalized-data-throughput"]),
            ("Airtime Fairness", ["time-on-air"]),
            ("Collision per Node", ["collision-per-node"]),
        ]
        value_column = params.get("value_column", "mean")
        base_dir = data_cfg
        if not os.path.isdir(base_dir):
            raise ValueError(f"Data path for parallel coordinates must be a directory: {base_dir}")

        protocol_metrics = {}
        available_protocols = set()
        available_envs = set()

        for display_name, dir_candidates in metrics_order:
            metric_dir = None
            for candidate in dir_candidates:
                candidate_path = os.path.join(base_dir, candidate)
                if os.path.isdir(candidate_path):
                    metric_dir = candidate_path
                    break
            if not metric_dir:
                raise ValueError(f"Metric directory not found for {display_name}: tried {dir_candidates}")

            for fname in sorted(os.listdir(metric_dir)):
                if not fname.lower().endswith(".json"):
                    continue
                path = os.path.join(metric_dir, fname)
                df, metadata = load_data(path)
                base = os.path.splitext(os.path.basename(path))[0]
                parts = base.split("_")
                file_protocol = parts[0] if parts else None
                environment = parts[1] if len(parts) > 1 else None

                protocol_raw = metadata.get("protocol") or file_protocol or "unknown"
                protocol_label = protocol_display.get(str(protocol_raw).lower(), protocol_raw)
                environment_label = environment or metadata.get("dimensions") or "env"

                available_protocols.add(protocol_label)
                available_envs.add(environment_label)

                if protocol_filters and protocol_label not in protocol_filters:
                    continue
                if not is_field_experiments and env_filters and str(environment_label) not in env_filters:
                    continue
                if is_field_experiments and field_topology:
                    if not _topology_matches(environment_label, field_topology):
                        continue

                if "numberNodes" not in df.columns or "timeToNextMission" not in df.columns:
                    continue
                df["numberNodes_num"] = pd.to_numeric(df["numberNodes"], errors="coerce")
                df["timeToNextMission_num"] = pd.to_numeric(df["timeToNextMission"], errors="coerce")
                df = df[df["numberNodes_num"].notna() & df["timeToNextMission_num"].notna()]
                df = df[
                    np.isclose(df["numberNodes_num"], freeze_nodes)
                    & np.isclose(df["timeToNextMission_num"], freeze_ttm)
                ]
                if df.empty:
                    continue
                value_series = None
                if value_column in df.columns:
                    value_series = pd.to_numeric(df[value_column], errors="coerce")
                    if value_series.isna().all():
                        # Handle nested dict column with 'mean' values.
                        value_series = df[value_column].apply(
                            lambda v: v.get("mean") if isinstance(v, dict) else None
                        )
                        value_series = pd.to_numeric(value_series, errors="coerce")
                # Fallback: find a column that stores dicts with 'mean'.
                if value_series is None or value_series.isna().all():
                    for col in df.columns:
                        if col in ("numberNodes", "timeToNextMission", "numberNodes_num", "timeToNextMission_num"):
                            continue
                        series = df[col]
                        if series.apply(lambda v: isinstance(v, dict) and "mean" in v).any():
                            value_series = pd.to_numeric(
                                series.apply(lambda v: v.get("mean") if isinstance(v, dict) else None),
                                errors="coerce",
                            )
                            break
                if value_series is None or value_series.dropna().empty:
                    continue
                value_series = value_series.dropna()
                if value_series.empty:
                    continue
                val = float(value_series.mean())
                protocol_entry = protocol_metrics.setdefault(protocol_label, {})
                protocol_entry[display_name] = val

        if not protocol_metrics:
            raise ValueError(
                "No datasets matched the requested filters; "
                f"available protocols: {sorted(available_protocols) or 'none found'}; "
                f"available environments: {sorted(available_envs) or 'none found'}"
            )

        # Collision score transform: inverse then max-normalize.
        collision_scores = []
        for proto, metrics in protocol_metrics.items():
            if "Collision per Node" in metrics:
                raw = float(metrics["Collision per Node"])
                score = float("inf") if raw == 0 else 1.0 / raw
                metrics["Collision per Node"] = score
                collision_scores.append(score)
        if collision_scores:
            finite_scores = [s for s in collision_scores if pd.notna(s) and s != float("inf")]
            max_score = max(finite_scores) if finite_scores else None
            for proto, metrics in protocol_metrics.items():
                score = metrics.get("Collision per Node")
                if score is None:
                    continue
                if max_score and pd.notna(score) and score != float("inf"):
                    metrics["Collision per Node"] = score / max_score
                else:
                    metrics["Collision per Node"] = 0.0

        # NDTPN normalization by max across protocols.
        ndtpn_values = [metrics.get("NEDTPN") for metrics in protocol_metrics.values() if "NEDTPN" in metrics]
        ndtpn_values = [float(v) for v in ndtpn_values if pd.notna(v)]
        if ndtpn_values:
            ndtpn_max = max(ndtpn_values)
            if ndtpn_max > 0:
                for metrics in protocol_metrics.values():
                    if "NEDTPN" in metrics and pd.notna(metrics["NEDTPN"]):
                        metrics["NEDTPN"] = float(metrics["NEDTPN"]) / ndtpn_max

        metrics_names_order = [name for name, _ in metrics_order]
        rows = []
        for proto, metrics in protocol_metrics.items():
            if all(m in metrics for m in metrics_names_order):
                row = {"Protocol": proto}
                row.update({m: metrics[m] for m in metrics_names_order})
                rows.append(row)

        if not rows:
            raise ValueError("No protocols had complete data for all metrics at the frozen values.")

        df_plot = pd.DataFrame(rows)
        parallel_coordinates.plot_parallel_coordinates(
            df_plot,
            metrics_order=metrics_names_order,
            output_path=outdir,
            monochrome=bool(params.get("monochrome", False)),
            filename_tag="field" if is_field_experiments else "simu",
            freeze_nodes=freeze_nodes,
            freeze_ttm=freeze_ttm,
            config_name=config_name,
            style=style_cfg,
            width_factor=float(params.get("width_factor", 1.0)),
        )
    elif diagram == "heatmap":
        datasets = []
        protocol_display = {
            "aloha": "ALOHA",
            "csma": "CSMA",
            "meshrouter": "MeshRouter",
            "rs-mitra": "RS-MiTra",
            "rsmitra": "RS-MiTra",
            "rsmitranr": "RS-MiTra-NR",
            "rs-mitra-nr": "RS-MiTra-NR",
            "rsmitranav": "RS-MiTra-NAV",
            "rs-mitra-nav": "RS-MiTra-NAV",
            "irsmitra": "I-RS-MiTra",
            "i-rs-mitra": "I-RS-MiTra",
            "mirs": "Mi-RS",
            "mi-rs": "Mi-RS",
        }
        protocol_filters = params.get("mac_protocols") or params.get("protocols")
        if isinstance(protocol_filters, str):
            protocol_filters = [protocol_filters]
        if protocol_filters:
            protocol_filters = {
                protocol_display.get(str(p).lower(), str(p)) for p in protocol_filters
            }
        env_filters = params.get("environments") or params.get("envs")
        if isinstance(env_filters, str):
            env_filters = [env_filters]
        if env_filters:
            env_filters = {str(env) for env in env_filters}
        raw_field_flag = params.get("isFieldExperiments") or params.get("is_field_experiments")
        if isinstance(raw_field_flag, str):
            is_field_experiments = raw_field_flag.lower() == "true"
        else:
            is_field_experiments = bool(raw_field_flag)
        field_topology = params.get("field_topology")
        default_protocol_order_keys = [
            "aloha",
            "csma",
            "meshrouter",
            "rsmitra",
            "rsmitranr",
            "rsmitranav",
            "irsmitra",
            "mirs",
        ]
        default_env_order = ["300m", "1000m", "5000m", "10000m"]
        protocols = set()
        available_protocols = set()
        environments = set()
        available_envs = set()
        metric_label = None
        metric_slug = None

        for path in data_paths:
            df, metadata = load_data(path)
            base = os.path.splitext(os.path.basename(path))[0]
            parts = base.split("_")
            file_protocol = parts[0] if parts else None
            environment = parts[1] if len(parts) > 1 else None
            metric_raw = parts[2] if len(parts) > 2 else None

            protocol_raw = metadata.get("protocol") or file_protocol or "unknown"
            protocol_label = protocol_display.get(
                str(protocol_raw).lower(), protocol_raw
            )
            environment_label = environment or metadata.get("dimensions") or "env"
            label = protocol_label if is_field_experiments else f"{protocol_label} | {environment_label}"

            available_protocols.add(protocol_label)
            available_envs.add(environment_label)
            if protocol_filters and protocol_label not in protocol_filters:
                continue
            if env_filters and str(environment_label) not in env_filters:
                continue
            if is_field_experiments and field_topology:
                if not _topology_matches(environment_label, field_topology):
                    continue
            protocols.add(protocol_label)
            environments.add(environment_label)
            if metric_raw and not metric_label:
                metric_slug = metric_raw
                metric_label = _format_metric(metric_raw)
            datasets.append((df, metadata, label, protocol_label, environment_label))

        if not datasets:
            raise ValueError(
                "No datasets matched the requested filters; "
                f"available protocols: {sorted(available_protocols) or 'none found'}; "
                f"available environments: {sorted(available_envs) or 'none found'}"
            )

        # Use metadata from the first file to resolve defaults when available.
        _, first_metadata, _, _, _ = datasets[0]
        monochrome = bool(
            get_param(params, first_metadata, "monochrome", default=False)
        )
        x_col = get_param(params, first_metadata, "x_column", required=True)
        y_col = get_param(params, first_metadata, "y_column", required=True)
        val_col = get_param(params, first_metadata, "value_column", required=True)
        x_order = params.get("x_order")
        y_order = params.get("y_order")
        ordinal_x = bool(
            get_param(params, first_metadata, "ordinal_x", default=bool(x_order))
        )
        ordinal_y = bool(
            get_param(params, first_metadata, "ordinal_y", default=bool(y_order))
        )
        split_rows = params.get("split_rows")
        color_min = params.get("color_min")
        color_max = params.get("color_max")
        if color_min is not None:
            color_min = float(color_min)
        if color_max is not None:
            color_max = float(color_max)
        metric_label_param = (
            params.get("metric_name")
            or params.get("metric_label")
            or params.get("metric")
        )
        if metric_label_param:
            metric_slug = str(metric_label_param)
            metric_label = _format_metric(metric_label_param)
        elif metric_label is None and first_metadata.get("metric"):
            metric_slug = str(first_metadata["metric"])
            metric_label = _format_metric(metric_slug)
        elif metric_label and not metric_slug:
            metric_slug = metric_label.lower().replace(" ", "-")

        metric_label_display = metric_label
        metric_slug_lower = (metric_slug or "").lower()
        if metric_label and metric_slug_lower in {"effective-throughput", "throughput"}:
            metric_label_display = f"{metric_label} (Bytes per second)"

        y_axis_label = (
            "Bytes per second"
            if (metric_slug or "").lower() in ("normalized-effective-throughput", "normalized-throughput")
            else ""
        )

        def _ordered_subset(labels, preferred):
            if not labels:
                return []
            ordered = []
            seen = set()
            for item in preferred or []:
                if item in labels and item not in seen:
                    ordered.append(item)
                    seen.add(item)
            for label in labels:
                if label not in seen:
                    ordered.append(label)
                    seen.add(label)
            return ordered

        if len(datasets) == 1:
            df, metadata, _, _, _ = datasets[0]
            heatmap.plot_heatmap(
                df,
                x_col,
                y_col,
                val_col,
                outdir,
                metadata=metadata,
                monochrome=monochrome,
                ordinal_y=ordinal_y,
                ordinal_x=ordinal_x,
                x_order=x_order,
                y_order=y_order,
                colorbar_label=metric_label_display,
                y_axis_label=y_axis_label,
                metric_slug=metric_slug,
                filename_tag="field" if is_field_experiments else "simu",
                config_name=config_name,
                vmin=color_min,
                vmax=color_max,
                style=style_cfg,
            )
        elif is_field_experiments:
            protocol_order = [protocol_display.get(k, k) for k in default_protocol_order_keys]
            ordered_protocols = _ordered_subset(protocols, protocol_order)
            protocol_to_dataset = {entry[3]: entry for entry in datasets}
            ordered_datasets = [
                protocol_to_dataset[p] for p in ordered_protocols if p in protocol_to_dataset
            ]
            heatmap.plot_heatmap_grid(
                ordered_datasets,
                x_col,
                y_col,
                val_col,
                outdir,
                monochrome=monochrome,
                ordinal_y=ordinal_y,
                ordinal_x=ordinal_x,
                facet_columns=4,
                x_order=x_order,
                y_order=y_order,
                colorbar_label=metric_label_display,
                y_axis_label=y_axis_label,
                metric_slug=metric_slug,
                filename_tag="field" if is_field_experiments else "simu",
                config_name=config_name,
                vmin=color_min,
                vmax=color_max,
                style=style_cfg,
            )
        else:
            protocol_order = [protocol_display.get(k, k) for k in default_protocol_order_keys]
            env_order = default_env_order
            unique_protocols = _ordered_subset(protocols, protocol_order)
            unique_envs = _ordered_subset(environments, env_order)
            # Use matrix layout when we have both protocol and environment labels.
            if len(unique_protocols) > 1 or len(unique_envs) > 1:
                heatmap.plot_heatmap_grid_matrix(
                    datasets,
                    x_col,
                    y_col,
                    val_col,
                    outdir,
                    row_labels=unique_protocols,
                    col_labels=unique_envs,
                    monochrome=monochrome,
                    ordinal_x=ordinal_x,
                    ordinal_y=ordinal_y,
                    x_order=x_order,
                    y_order=y_order,
                    colorbar_label=metric_label_display,
                    split_rows=split_rows,
                    y_axis_label=y_axis_label,
                    metric_slug=metric_slug,
                    filename_tag="field" if is_field_experiments else "simu",
                    config_name=config_name,
                    vmin=color_min,
                    vmax=color_max,
                    style=style_cfg,
                )
            else:
                facet_cols = int(
                    get_param(params, first_metadata, "facet_columns", default=4)
                )
                heatmap.plot_heatmap_grid(
                    datasets,
                    x_col,
                    y_col,
                    val_col,
                    outdir,
                    monochrome=monochrome,
                    ordinal_y=ordinal_y,
                    ordinal_x=ordinal_x,
                    facet_columns=facet_cols,
                    x_order=x_order,
                    y_order=y_order,
                    colorbar_label=metric_label_display,
                    y_axis_label=y_axis_label,
                    metric_slug=metric_slug,
                    filename_tag="field" if is_field_experiments else "simu",
                    config_name=config_name,
                    vmin=color_min,
                    vmax=color_max,
                    style=style_cfg,
                )
    else:
        raise ValueError(f"Unsupported diagram type '{diagram}'.")


def main():
    parser = argparse.ArgumentParser(description="Diagram Generator")
    parser.add_argument(
        "--config", required=True, help="Path to diagram config file or directory."
    )
    args = parser.parse_args()

    config_input = args.config
    if os.path.isdir(config_input):
        config_paths = []
        for root, _, files in os.walk(config_input):
            for fname in files:
                if fname.lower().endswith((".config", ".json")):
                    config_paths.append(os.path.join(root, fname))
        config_paths.sort()
        if not config_paths:
            raise SystemExit(f"No config files found under directory: {config_input}")
        for path in config_paths:
            run_single_config(path)
    else:
        run_single_config(config_input)


if __name__ == "__main__":
    main()
