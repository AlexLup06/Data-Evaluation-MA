# Data Evaluation Toolkit

Python scripts to generate publication-ready figures for MAC protocol experiments. Supports heatmaps, protocol line plots, and parallel coordinates across multiple metrics.

## Features
- **Heatmaps:** Parameter sweeps (nodes vs. mission send interval) per protocol/env; grid layouts with shared scales; optional protocol/env filters; clamped color scales (`color_min`/`color_max`). Palette: cividis (Greys for monochrome).
- **Line plots:** Fix one parameter (e.g., mission interval) and plot metric vs. nodes; one line per protocol, optional filters; width scaling via `width_factor`. Palette: Okabe–Ito (black when monochrome).
- **Parallel coordinates:** Compare multiple metrics per protocol at a fixed node/time point; metrics order: Reachability → RSR → NDTPN (max-normalized) → MAC Efficiency → Airtime Fairness → Collision Score (inverted then max-normalized). Separate configs for simu/field; filenames tagged with `simu`/`field`.

## Data format
JSON files with:
```json
{
  "data": [
    {
      "metadata": { "numberNodes": 10, "timeToNextMission": 2, "protocol": "ALOHA", ... },
      "data": { "mean": 0.5, "count": 10, "std": 0.1, "ci95": [0.4, 0.6] }
    }
  ],
  "metadata": { ... }
}
```
Nested dict metrics (e.g., `{ "ratio": { "mean": ... } }`) are supported.

## Configs
Each config JSON has:
- `diagram`: `"heatmap"`, `"line"`, or `"parallel"`.
- `name` (optional): label used in output filenames (format `type-metric-name-part.png`; metric omitted for parallel; part only when a heatmap grid is split).
- `data`: path to a file, list of files, or a base directory (for heatmap/line, `metric_name` may select a metric subfolder).
- `outdir`: output directory (default `outputs`).
- `schema`: optional; defaults from `config.py`.
- `style` (optional): font sizes per diagram; matches keys in `config.DEFAULT_STYLE` (e.g., `{"heatmap": {"title": 14, "axis_label": 12}, "line": {"tick": 11}}`).
- `params`: diagram-specific options, e.g.:
  - Common: `mac_protocols`, `environments`, `isFieldExperiments`.
  - Heatmap: `x_column`, `y_column`, `value_column`, `x_order`, `y_order`, `ordinal_x/y`, `split_rows`, `color_min`, `color_max`, `metric_name`.
  - Line: `x_column`, `value_column`, `freeze_column` (e.g., `timeToNextMission`), `freeze_value`, `width_factor`, `metric_name`.
  - Parallel: `freeze_nodes`, `freeze_time`, `value_column`, `mac_protocols`, `environments` (ignored for field), `metric_name` (implied by metric directories).

Filenames are tagged with the metric slug and `simu`/`field`.

## Usage
1. Install deps:
   ```bash
   python -m venv venv && ./venv/bin/pip install -r requirements.txt
   ```
2. Run:
   ```bash
   python main.py --config configs/heatmap-simu-throughput-config.json
   python main.py --config configs/line-simu-normalized-effective-throughput-config.json
   python main.py --config configs/parallel-field-config.json
   ```
3. Outputs: PNGs in `outputs/`.

## Notes
- Heatmap y-axis is unlabeled except throughput metrics (“Bytes per second”).
- Line plots add a title with the metric name.
- Parallel coordinates require all metrics to be present for the chosen freeze values; collision is inverted and normalized; NDTPN is max-normalized across protocols.
