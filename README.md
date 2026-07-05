# apoptosis

CLI for classifying cell viability from per-cell microscopy ROI time-lapses. Manual labels drive a ResNet classifier trained with PyTorch Lightning; inference compares morphology-based death timing with Toto-3 fluorescence.

## Install

```bash
uv sync
```

The `apoptosis` entry point is available in the project virtual environment.

```bash
uv run apoptosis --help
uv run apoptosis <command> --help
```

## Input layout

Commands expect an experiment root with per-position ROI stacks. Each ROI TIFF interleaves brightfield (channel 0) and Toto-3 (channel 1) frames:

```text
data_dir/
  roi/
    Pos0/
      Roi0.tif
      Roi1.tif
      ...
    Pos28/
      Roi0.tif
      ...
```

Manual labels are stored as JSON (default: `<project>/labels.json`):

```json
[
  {
    "position": "Pos0",
    "roi_id": 0,
    "death_frame": 106,
    "labeled_at": "2026-06-28T20:10:53.364121+00:00"
  }
]
```

A `death_frame` equal to the ROI timepoint count means the cell stayed healthy through the acquisition.

## Workflow

Typical end-to-end pipeline:

```text
label -> dataset-build -> train -> eval -> predict
```

```bash
# Annotate ROIs in the browser (writes labels.json)
uv run apoptosis label --data-dir /path/to/data

# Build per-frame train/val manifest from labels
uv run apoptosis dataset-build --data-dir /path/to/data --labels-path labels.json

# Train ResNet viability classifier
uv run apoptosis train --manifest datasets/viability/manifest.json

# Report accuracy/F1 on train and val splits
uv run apoptosis eval --checkpoint runs/viability/lightning_logs/.../best-*.ckpt

# Infer all cells and write validation figure
uv run apoptosis predict --data-dir /path/to/data
```

**Outputs**

| Step | Output |
|---|---|
| `label` | `labels.json` (or `--labels-path`) |
| `dataset-build` | `datasets/viability/manifest.json` |
| `train` | `runs/viability/lightning_logs/version_*/checkpoints/best-*.ckpt` |
| `eval` | Metrics printed to stdout |
| `predict` | `runs/viability/inference.json`, `runs/viability/validation_plot.png` |

## Labeling web app

`apoptosis label` starts a FastAPI server (`apoptosis.api`) backed by `routes/labeling.py`. The browser UI (`static/label.html`) lists ROIs, shows brightfield/Toto-3 frames, and saves death-frame annotations via REST (`/api/rois`, `/api/labels`). Open the URL printed by the command (default `http://127.0.0.1:8000`).

## Command reference

| Command | Role |
|---|---|
| `label` | Launch the ROI viability labeling webapp |
| `dataset-build` | Build a per-frame viability dataset manifest from manual labels |
| `train` | Train a ResNet viability classifier with PyTorch Lightning |
| `eval` | Evaluate a trained model on train and val splits |
| `predict` | Run viability inference on all cells; plot Toto-3 vs morphology timing |
| `hello` | Greet someone (smoke test) |
| `version` | Show the installed version |

## Paper figures

One-off figure scripts that are not CLI commands live in `scripts/`. For example, `scripts/plot_fig6.py` reads `runs/viability/inference.json` and writes paper figures to an external repo path configured in the script.