import argparse
import json
import os
from config import DEFAULT_SCHEMAS, load_config
from utils.loader import load_data
from utils.validate import validate_json
from diagrams import heatmap, lineplot


def get_param(
    params: dict, metadata: dict, key: str, default=None, required: bool = False
):
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


def main():
    parser = argparse.ArgumentParser(description="Diagram Generator")
    parser.add_argument(
        "--config", required=True, help="Path to diagram config file (JSON)."
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    diagram = cfg["diagram"]
    data_cfg = cfg["data"]
    outdir = cfg.get("outdir", "outputs")
    schema_path = cfg.get("schema", DEFAULT_SCHEMAS.get(diagram))
    params = cfg.get("params", {})

    if not schema_path:
        raise ValueError(f"No schema found for diagram type '{diagram}'.")

    os.makedirs(outdir, exist_ok=True)

    # Expand folder inputs for diagrams that accept directories; validation handles list inputs.
    if diagram in ("heatmap", "contour", "line") and isinstance(data_cfg, str) and os.path.isdir(data_cfg):
        data_paths = [
            os.path.join(data_cfg, fname)
            for fname in sorted(os.listdir(data_cfg))
            if fname.lower().endswith(".json")
        ]
    else:
        data_paths = data_cfg if isinstance(data_cfg, list) else [data_cfg]

    if not data_paths:
        raise ValueError("No data files found for the provided configuration.")

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
                colorbar_label=metric_label,
                y_axis_label=y_axis_label,
                filename_slug=metric_slug,
                filename_tag="field" if is_field_experiments else "simu",
                vmin=color_min,
                vmax=color_max,
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
                colorbar_label=metric_label,
                y_axis_label=y_axis_label,
                filename_slug=metric_slug,
                filename_tag="field" if is_field_experiments else "simu",
                vmin=color_min,
                vmax=color_max,
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
                    colorbar_label=metric_label,
                    split_rows=split_rows,
                    y_axis_label=y_axis_label,
                    filename_slug=metric_slug,
                    filename_tag="field" if is_field_experiments else "simu",
                    vmin=color_min,
                    vmax=color_max,
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
                    colorbar_label=metric_label,
                    y_axis_label=y_axis_label,
                    filename_slug=metric_slug,
                    filename_tag="field" if is_field_experiments else "simu",
                    vmin=color_min,
                    vmax=color_max,
                )
    else:
        raise ValueError(f"Unsupported diagram type '{diagram}'.")


if __name__ == "__main__":
    main()
