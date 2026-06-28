from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from apoptosis.ml.frame_dataset import ViabilityFrameDataset, load_manifest_samples
from apoptosis.ml.viability_module import ViabilityModule


@dataclass(frozen=True)
class SplitMetrics:
    split: str
    samples: int
    accuracy: float
    f1: float
    precision: float
    recall: float
    true_viable: int
    true_dead: int
    pred_viable: int
    pred_dead: int
    confusion: list[list[int]]


def _binary_metrics(
    preds: list[int],
    labels: list[int],
) -> tuple[float, float, float, float, list[list[int]]]:
    pairs = list(zip(preds, labels, strict=True))
    tp = sum(1 for pred, label in pairs if pred == 1 and label == 1)
    tn = sum(1 for pred, label in pairs if pred == 0 and label == 0)
    fp = sum(1 for pred, label in pairs if pred == 1 and label == 0)
    fn = sum(1 for pred, label in pairs if pred == 0 and label == 1)
    total = len(labels)
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    denom = precision + recall
    f1 = (2 * precision * recall / denom) if denom else 0.0
    return accuracy, f1, precision, recall, [[tn, fp], [fn, tp]]


def _evaluate_split(
    model: ViabilityModule,
    dataset: ViabilityFrameDataset,
    split: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> SplitMetrics:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    preds: list[int] = []
    labels: list[int] = []

    model.eval()
    with torch.inference_mode():
        for images, batch_labels in loader:
            images = images.to(device)
            batch_preds = torch.argmax(model(images), dim=1).cpu().tolist()
            preds.extend(int(value) for value in batch_preds)
            labels.extend(int(value) for value in batch_labels.tolist())

    accuracy, f1, precision, recall, confusion = _binary_metrics(preds, labels)
    true_dead = sum(1 for label in labels if label == 0)
    true_viable = len(labels) - true_dead
    pred_dead = sum(1 for pred in preds if pred == 0)
    pred_viable = len(preds) - pred_dead

    return SplitMetrics(
        split=split,
        samples=len(labels),
        accuracy=accuracy,
        f1=f1,
        precision=precision,
        recall=recall,
        true_viable=true_viable,
        true_dead=true_dead,
        pred_viable=pred_viable,
        pred_dead=pred_dead,
        confusion=confusion,
    )


def evaluate_checkpoint(
    checkpoint_path: Path,
    manifest_path: Path,
    batch_size: int = 64,
    num_workers: int = 4,
    accelerator: str = "auto",
) -> list[SplitMetrics]:
    if accelerator == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif accelerator == "gpu":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = ViabilityModule.load_from_checkpoint(str(checkpoint_path))
    model.to(device)

    data_dir = Path(json.loads(manifest_path.read_text())["data_dir"])
    results: list[SplitMetrics] = []
    for split in ("train", "val"):
        samples = load_manifest_samples(manifest_path, split)
        dataset = ViabilityFrameDataset(data_dir, samples)
        results.append(
            _evaluate_split(
                model=model,
                dataset=dataset,
                split=split,
                device=device,
                batch_size=batch_size,
                num_workers=num_workers,
            )
        )
    return results
