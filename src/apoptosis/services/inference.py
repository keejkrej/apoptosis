from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import tifffile
import torch
from torch.utils.data import DataLoader, TensorDataset

from apoptosis.core.roi import discover_rois, roi_path, timepoint_count
from apoptosis.core.toto import (
    death_time_from_probability,
    death_time_from_toto,
    position_median_traces,
    toto_alive_signal,
    toto_trace,
)
from apoptosis.ml.preprocess import stack_brightfield_tensor
from apoptosis.ml.viability_module import ViabilityModule

POSITION_META = {
    "Pos0": {"label": "100 nM STS", "color": "#1f77b4"},
    "Pos28": {"label": "500 nM STS", "color": "#d62728"},
    "Pos70": {"label": "Control", "color": "#2ca02c"},
}


@dataclass(frozen=True)
class CellInference:
    position: str
    roi_id: int
    timepoints: int
    death_time_toto: int
    death_time_viability: int
    toto_raw: list[float]
    toto_alive: list[float]
    death_probability: list[float]


def _default_checkpoint(project_root: Path) -> Path:
    root = project_root / "runs" / "viability" / "lightning_logs"
    candidates = sorted(
        root.rglob("*.ckpt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        if "best-" in path.name or "best-epoch" in path.name:
            return path
    if candidates:
        return candidates[0]
    msg = f"No checkpoints found under {root}"
    raise FileNotFoundError(msg)


def _predict_death_probability(
    model: ViabilityModule,
    stack: np.ndarray,
    timepoints: int,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    tensors = [
        stack_brightfield_tensor(stack, time_index) for time_index in range(timepoints)
    ]
    loader = DataLoader(
        TensorDataset(torch.stack(tensors)),
        batch_size=batch_size,
        shuffle=False,
    )
    probs: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for (batch,) in loader:
            logits = model(batch.to(device))
            dead_prob = torch.softmax(logits, dim=1)[:, 0].cpu().numpy()
            probs.append(dead_prob)
    return np.concatenate(probs).astype(np.float32)


def infer_all_cells(
    data_dir: Path,
    checkpoint_path: Path,
    batch_size: int = 64,
    accelerator: str = "auto",
) -> list[CellInference]:
    if accelerator == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif accelerator == "gpu":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = ViabilityModule.load_from_checkpoint(str(checkpoint_path))
    model.to(device)

    all_rois = discover_rois(data_dir)
    medians = position_median_traces(data_dir, all_rois)

    results: list[CellInference] = []
    for roi in all_rois:
        stack = tifffile.imread(roi_path(data_dir, roi))
        timepoints = timepoint_count(data_dir, roi)
        trace = toto_trace(data_dir, roi)
        # Use position-median residual for jump detection: suppresses common artifacts
        # (e.g. global illumination/focus shifts) while highlighting cell-specific
        # intensity increases (dye uptake on death). This applies to all positions,
        # including controls (Pos70 cells can still die).
        pos_median = medians.get(roi.position)
        contrast = (trace - pos_median) if pos_median is not None else trace
        death_prob = _predict_death_probability(
            model=model,
            stack=stack,
            timepoints=timepoints,
            device=device,
            batch_size=batch_size,
        )
        results.append(
            CellInference(
                position=roi.position,
                roi_id=roi.roi_id,
                timepoints=timepoints,
                death_time_toto=death_time_from_toto(contrast, timepoints, raw_trace=trace),
                death_time_viability=death_time_from_probability(
                    death_prob, timepoints
                ),
                toto_raw=trace.tolist(),
                toto_alive=toto_alive_signal(trace).tolist(),
                death_probability=death_prob.tolist(),
            )
        )
    return results


def save_inference(results: list[CellInference], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(item) for item in results]
    output_path.write_text(json.dumps(payload, indent=2) + "\n")


def load_inference(path: Path) -> list[CellInference]:
    payload = json.loads(path.read_text())
    return [CellInference(**item) for item in payload]
