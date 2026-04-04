from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np
import tifffile
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18


DEFAULT_DATASET_ROOT = Path(r"C:\Users\ctyja\data\20260327\dataset")
DEFAULT_IMAGE_SIZE = 224
DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_SEED = 42
DEFAULT_THRESHOLD = 0.5
DEFAULT_NUM_WORKERS = 0
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15

BF_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_ROOT = BF_ROOT / "artifacts"
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


@dataclass(frozen=True)
class ExampleRecord:
    split_folder: str
    image_relpath: str
    image_path: Path
    position: str
    roi: int
    time_index: int
    dead_probability: float
    source_tif: str
    live_anchor_t: int
    dead_anchor_t: int | None
    annotation_mode: str

    @property
    def roi_group(self) -> str:
        return f"{self.position}_roi{self.roi:03d}"


@dataclass(frozen=True)
class TrainingConfig:
    dataset_root: Path = DEFAULT_DATASET_ROOT
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT
    run_name: str | None = None
    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    image_size: int = DEFAULT_IMAGE_SIZE
    lr: float = DEFAULT_LR
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    seed: int = DEFAULT_SEED
    num_workers: int = DEFAULT_NUM_WORKERS
    threshold: float = DEFAULT_THRESHOLD
    pretrained: bool = True
    device: str = "auto"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["dataset_root"] = str(self.dataset_root)
        payload["artifact_root"] = str(self.artifact_root)
        return payload


@dataclass(frozen=True)
class TrainingArtifacts:
    run_dir: Path
    best_checkpoint_path: Path
    last_checkpoint_path: Path
    config_path: Path
    metrics_csv_path: Path
    test_metrics_path: Path
    train_split_path: Path
    val_split_path: Path
    test_split_path: Path


@dataclass(frozen=True)
class PredictionResult:
    image_path: Path
    dead_probability: float
    hard_label: str
    checkpoint_path: Path
    threshold: float


@dataclass(frozen=True)
class TimelapsePredictionRow:
    time_index: int
    dead_probability: float
    hard_label: str


@dataclass(frozen=True)
class TimelapsePredictionResult:
    input_path: Path
    checkpoint_path: Path
    output_csv_path: Path
    channel: int
    frame_count: int
    threshold: float
    rows: list[TimelapsePredictionRow]


class ApoptosisFrameDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, records: list[ExampleRecord], image_size: int) -> None:
        if not records:
            raise ValueError("Dataset split is empty")
        self.records = records
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        image = preprocess_tiff_image(record.image_path, image_size=self.image_size)
        target = torch.tensor(record.dead_probability, dtype=torch.float32)
        return image, target


def default_run_name() -> str:
    return datetime.now().strftime("resnet18_%Y%m%d_%H%M%S")


def choose_device(requested_device: str) -> torch.device:
    lowered = requested_device.lower()
    if lowered == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(lowered)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def windows_relpath_to_path(relative_path: str) -> Path:
    return Path(*PureWindowsPath(relative_path).parts)


def parse_optional_int(value: str) -> int | None:
    stripped = value.strip()
    return int(stripped) if stripped else None


def default_scores_csv_path(image_path: Path, channel: int) -> Path:
    return image_path.with_name(f"{image_path.stem}_ch{channel}_scores.csv")


def default_scores_plot_path(scores_csv_path: Path) -> Path:
    return scores_csv_path.with_suffix(".png")


def load_manifest(dataset_root: Path) -> list[ExampleRecord]:
    dataset_root = dataset_root.resolve()
    labels_csv = dataset_root / "labels.csv"
    if not labels_csv.exists():
        raise FileNotFoundError(f"labels.csv not found at {labels_csv}")

    records: list[ExampleRecord] = []
    with labels_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_path = (dataset_root / windows_relpath_to_path(row["image_relpath"])).resolve()
            if not image_path.exists():
                raise FileNotFoundError(f"Image path from labels.csv does not exist: {image_path}")
            records.append(
                ExampleRecord(
                    split_folder=row["split_folder"],
                    image_relpath=row["image_relpath"],
                    image_path=image_path,
                    position=row["position"],
                    roi=int(row["roi"]),
                    time_index=int(row["time_index"]),
                    dead_probability=float(row["dead_probability"]),
                    source_tif=row["source_tif"],
                    live_anchor_t=int(row["live_anchor_t"]),
                    dead_anchor_t=parse_optional_int(row["dead_anchor_t"]),
                    annotation_mode=row["annotation_mode"],
                )
            )
    if not records:
        raise ValueError(f"No rows found in {labels_csv}")
    return records


def split_group_ids(group_ids: list[str], seed: int) -> dict[str, set[str]]:
    if len(group_ids) < 3:
        raise ValueError("At least 3 ROI groups are required for train/val/test splitting")

    shuffled = list(group_ids)
    random.Random(seed).shuffle(shuffled)

    train_count = int(round(len(shuffled) * TRAIN_FRACTION))
    train_count = min(max(train_count, 1), len(shuffled) - 2)
    val_count = int(round(len(shuffled) * VAL_FRACTION))
    val_count = min(max(val_count, 1), len(shuffled) - train_count - 1)
    test_count = len(shuffled) - train_count - val_count
    if test_count <= 0:
        test_count = 1
        if train_count >= val_count:
            train_count -= 1
        else:
            val_count -= 1

    train_ids = set(shuffled[:train_count])
    val_ids = set(shuffled[train_count : train_count + val_count])
    test_ids = set(shuffled[train_count + val_count : train_count + val_count + test_count])
    return {"train": train_ids, "val": val_ids, "test": test_ids}


def split_records_by_roi(records: list[ExampleRecord], seed: int) -> dict[str, list[ExampleRecord]]:
    split_ids = split_group_ids(sorted({record.roi_group for record in records}), seed=seed)
    split_records: dict[str, list[ExampleRecord]] = {"train": [], "val": [], "test": []}
    for record in records:
        if record.roi_group in split_ids["train"]:
            split_records["train"].append(record)
        elif record.roi_group in split_ids["val"]:
            split_records["val"].append(record)
        elif record.roi_group in split_ids["test"]:
            split_records["test"].append(record)
        else:
            raise AssertionError(f"Record {record.image_path} was not assigned to a split")
    return split_records


def preprocess_image_array(image_array: np.ndarray, image_size: int) -> torch.Tensor:
    image_array = np.asarray(image_array)
    image_array = np.squeeze(image_array)
    if image_array.ndim != 2:
        raise ValueError(f"Expected a 2D image, got shape {image_array.shape}")

    image_tensor = torch.from_numpy(image_array.astype(np.float32, copy=False))
    min_value = float(image_tensor.min())
    max_value = float(image_tensor.max())
    if max_value > min_value:
        image_tensor = (image_tensor - min_value) / (max_value - min_value)
    else:
        image_tensor = torch.zeros_like(image_tensor)

    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
    image_tensor = F.interpolate(
        image_tensor,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )
    image_tensor = image_tensor.squeeze(0).repeat(3, 1, 1)
    image_tensor = (image_tensor - IMAGENET_MEAN) / IMAGENET_STD
    return image_tensor.to(dtype=torch.float32)


def preprocess_tiff_image(image_path: Path, image_size: int) -> torch.Tensor:
    return preprocess_image_array(np.asarray(tifffile.imread(image_path)), image_size=image_size)


def build_model(pretrained: bool) -> nn.Module:
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


def load_roi_shape_from_index(tif_path: Path) -> tuple[int, int, int, int, int] | None:
    index_path = tif_path.parent / "index.json"
    if not index_path.exists():
        return None

    payload = json.loads(index_path.read_text(encoding="utf-8"))
    for roi_entry in payload.get("rois", []):
        if str(roi_entry.get("fileName")) == tif_path.name:
            shape = tuple(int(size) for size in roi_entry["shape"])
            if len(shape) != 5:
                raise ValueError(f"ROI shape from {index_path} must have 5 dimensions, got {shape}")
            return shape
    return None


def select_frames_from_interleaved_pages(
    raw_stack: np.ndarray,
    *,
    channel: int,
    channel_count: int,
) -> np.ndarray:
    if raw_stack.ndim != 3:
        raise ValueError(f"Expected flattened pages with shape (N, Y, X), got {raw_stack.shape}")
    if channel_count <= 0:
        raise ValueError(f"channel_count must be positive, got {channel_count}")
    if not 0 <= channel < channel_count:
        raise ValueError(f"channel must be between 0 and {channel_count - 1}, got {channel}")
    if raw_stack.shape[0] % channel_count != 0:
        raise ValueError(
            f"Page count {raw_stack.shape[0]} is not divisible by channel_count={channel_count}"
        )
    time_count = raw_stack.shape[0] // channel_count
    reshaped = raw_stack.reshape(time_count, channel_count, raw_stack.shape[1], raw_stack.shape[2])
    return np.asarray(reshaped[:, channel, :, :])


def extract_timelapse_frames(
    tif_path: Path,
    *,
    channel: int,
    channel_count: int | None = None,
) -> np.ndarray:
    resolved_path = tif_path.resolve()
    with tifffile.TiffFile(resolved_path) as tif:
        series = tif.series[0]
        axes = series.axes
        raw_stack = np.asarray(series.asarray())

    roi_shape = load_roi_shape_from_index(resolved_path)
    if roi_shape is not None:
        time_count, indexed_channel_count, z_count, height, width = roi_shape
        if not 0 <= channel < indexed_channel_count:
            raise ValueError(
                f"channel must be between 0 and {indexed_channel_count - 1}, got {channel}"
            )
        flattened_pages = time_count * indexed_channel_count * z_count
        if raw_stack.shape == roi_shape:
            reshaped = raw_stack
        elif raw_stack.ndim == 3 and raw_stack.shape == (flattened_pages, height, width):
            reshaped = raw_stack.reshape(roi_shape)
        elif raw_stack.ndim == 4 and raw_stack.shape == (time_count, indexed_channel_count, height, width):
            reshaped = raw_stack.reshape(time_count, indexed_channel_count, z_count, height, width)
        else:
            raise ValueError(
                f"{resolved_path} must reshape to {roi_shape}, got raw TIFF shape {raw_stack.shape}"
            )
        return np.asarray(reshaped[:, channel, 0, :, :])

    if raw_stack.ndim == 2:
        if channel != 0:
            raise ValueError(f"{resolved_path} is a single-channel frame; channel must be 0")
        return raw_stack[np.newaxis, :, :]

    if axes == "TYX":
        if channel != 0:
            raise ValueError(f"{resolved_path} has no explicit channel axis; channel must be 0")
        return np.asarray(raw_stack)

    if axes == "CYX":
        if not 0 <= channel < raw_stack.shape[0]:
            raise ValueError(f"channel must be between 0 and {raw_stack.shape[0] - 1}, got {channel}")
        return np.asarray(raw_stack[channel : channel + 1, :, :])

    if axes == "TCYX":
        if not 0 <= channel < raw_stack.shape[1]:
            raise ValueError(f"channel must be between 0 and {raw_stack.shape[1] - 1}, got {channel}")
        return np.asarray(raw_stack[:, channel, :, :])

    if axes == "TCZYX":
        if not 0 <= channel < raw_stack.shape[1]:
            raise ValueError(f"channel must be between 0 and {raw_stack.shape[1] - 1}, got {channel}")
        return np.asarray(raw_stack[:, channel, 0, :, :])

    if axes == "IYX":
        inferred_channel_count = channel_count if channel_count is not None else 1
        return select_frames_from_interleaved_pages(
            np.asarray(raw_stack),
            channel=channel,
            channel_count=inferred_channel_count,
        )

    raise ValueError(f"Unsupported TIFF axes {axes!r} for {resolved_path}")


def build_dataloader(
    records: list[ExampleRecord],
    *,
    image_size: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    dataset = ApoptosisFrameDataset(records=records, image_size=image_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def binary_accuracy(probabilities: list[float], targets: list[float], threshold: float) -> float:
    hard_targets = [1.0 if target >= threshold else 0.0 for target in targets]
    hard_predictions = [1.0 if probability >= threshold else 0.0 for probability in probabilities]
    correct = sum(int(prediction == target) for prediction, target in zip(hard_predictions, hard_targets))
    return correct / len(probabilities)


def binary_auroc(probabilities: list[float], targets: list[float], threshold: float) -> float:
    hard_targets = [1 if target >= threshold else 0 for target in targets]
    positives = sum(hard_targets)
    negatives = len(hard_targets) - positives
    if positives == 0 or negatives == 0:
        return float("nan")

    paired = sorted(zip(probabilities, hard_targets), key=lambda item: item[0], reverse=True)
    true_positives = 0
    false_positives = 0
    points: list[tuple[float, float]] = [(0.0, 0.0)]
    previous_score: float | None = None

    for score, label in paired:
        if previous_score is not None and score != previous_score:
            points.append((false_positives / negatives, true_positives / positives))
        if label == 1:
            true_positives += 1
        else:
            false_positives += 1
        previous_score = score
    points.append((false_positives / negatives, true_positives / positives))

    auc = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        auc += (x1 - x0) * (y0 + y1) * 0.5
    return auc


def summarize_epoch(
    *,
    probabilities: list[float],
    targets: list[float],
    average_loss: float,
    threshold: float,
) -> dict[str, float]:
    mae = sum(abs(probability - target) for probability, target in zip(probabilities, targets)) / len(probabilities)
    return {
        "loss": average_loss,
        "mae": mae,
        "accuracy": binary_accuracy(probabilities, targets, threshold=threshold),
        "auroc": binary_auroc(probabilities, targets, threshold=threshold),
    }


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    *,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    threshold: float,
) -> dict[str, float]:
    loss_fn = nn.BCEWithLogitsLoss()
    training = optimizer is not None
    if training:
        model.train()
    else:
        model.eval()

    total_examples = 0
    total_loss = 0.0
    probabilities: list[float] = []
    targets: list[float] = []

    for images, batch_targets in dataloader:
        images = images.to(device)
        batch_targets = batch_targets.to(device)

        with torch.set_grad_enabled(training):
            logits = model(images).squeeze(1)
            loss = loss_fn(logits, batch_targets)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        batch_size = images.shape[0]
        total_examples += batch_size
        total_loss += float(loss.detach().cpu()) * batch_size
        probabilities.extend(float(value) for value in torch.sigmoid(logits).detach().cpu().tolist())
        targets.extend(float(value) for value in batch_targets.detach().cpu().tolist())

    if total_examples == 0:
        raise ValueError("Dataloader produced zero examples")

    return summarize_epoch(
        probabilities=probabilities,
        targets=targets,
        average_loss=total_loss / total_examples,
        threshold=threshold,
    )


def make_run_dir(artifact_root: Path, run_name: str | None) -> Path:
    resolved_root = artifact_root.resolve()
    resolved_root.mkdir(parents=True, exist_ok=True)
    run_dir = resolved_root / (run_name or default_run_name())
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_split_manifest(path: Path, records: list[ExampleRecord]) -> None:
    fieldnames = [
        "split_folder",
        "image_relpath",
        "position",
        "roi",
        "time_index",
        "source_tif",
        "live_anchor_t",
        "dead_anchor_t",
        "dead_probability",
        "annotation_mode",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "split_folder": record.split_folder,
                    "image_relpath": record.image_relpath,
                    "position": record.position,
                    "roi": record.roi,
                    "time_index": record.time_index,
                    "source_tif": record.source_tif,
                    "live_anchor_t": record.live_anchor_t,
                    "dead_anchor_t": "" if record.dead_anchor_t is None else record.dead_anchor_t,
                    "dead_probability": f"{record.dead_probability:.6f}",
                    "annotation_mode": record.annotation_mode,
                }
            )


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def format_metric(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.6f}"


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    config: TrainingConfig,
    epoch: int,
    metrics: dict[str, float],
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config.to_dict(),
            "epoch": epoch,
            "metrics": metrics,
        },
        path,
    )


def summarize_split(records: list[ExampleRecord]) -> dict[str, int]:
    return {
        "frames": len(records),
        "rois": len({record.roi_group for record in records}),
        "live_frames": sum(1 for record in records if record.dead_probability <= 0.0),
        "mixed_frames": sum(1 for record in records if 0.0 < record.dead_probability < 1.0),
        "dead_frames": sum(1 for record in records if record.dead_probability >= 1.0),
    }


def train_model(config: TrainingConfig) -> TrainingArtifacts:
    set_seed(config.seed)
    dataset_root = config.dataset_root.resolve()
    artifact_root = config.artifact_root.resolve()
    device = choose_device(config.device)

    records = load_manifest(dataset_root)
    split_map = split_records_by_roi(records, seed=config.seed)
    run_dir = make_run_dir(artifact_root, config.run_name)

    train_split_path = run_dir / "train_split.csv"
    val_split_path = run_dir / "val_split.csv"
    test_split_path = run_dir / "test_split.csv"
    write_split_manifest(train_split_path, split_map["train"])
    write_split_manifest(val_split_path, split_map["val"])
    write_split_manifest(test_split_path, split_map["test"])

    config_path = run_dir / "config.json"
    metrics_csv_path = run_dir / "metrics.csv"
    best_checkpoint_path = run_dir / "best.pt"
    last_checkpoint_path = run_dir / "last.pt"
    test_metrics_path = run_dir / "test_metrics.json"

    save_json(
        config_path,
        {
            **config.to_dict(),
            "device_resolved": str(device),
            "split_summary": {
                split_name: summarize_split(split_records)
                for split_name, split_records in split_map.items()
            },
        },
    )

    train_loader = build_dataloader(
        split_map["train"],
        image_size=config.image_size,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = build_dataloader(
        split_map["val"],
        image_size=config.image_size,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    test_loader = build_dataloader(
        split_map["test"],
        image_size=config.image_size,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    model = build_model(pretrained=config.pretrained).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    best_val_loss = float("inf")
    history_rows: list[dict[str, str | int]] = []

    for epoch in range(1, config.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            device=device,
            threshold=config.threshold,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            optimizer=None,
            device=device,
            threshold=config.threshold,
        )
        combined_metrics = {
            "train_loss": train_metrics["loss"],
            "train_mae": train_metrics["mae"],
            "train_accuracy": train_metrics["accuracy"],
            "train_auroc": train_metrics["auroc"],
            "val_loss": val_metrics["loss"],
            "val_mae": val_metrics["mae"],
            "val_accuracy": val_metrics["accuracy"],
            "val_auroc": val_metrics["auroc"],
        }
        save_checkpoint(
            last_checkpoint_path,
            model=model,
            config=config,
            epoch=epoch,
            metrics=combined_metrics,
        )
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(
                best_checkpoint_path,
                model=model,
                config=config,
                epoch=epoch,
                metrics=combined_metrics,
            )

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": format_metric(train_metrics["loss"]),
                "train_mae": format_metric(train_metrics["mae"]),
                "train_accuracy": format_metric(train_metrics["accuracy"]),
                "train_auroc": format_metric(train_metrics["auroc"]),
                "val_loss": format_metric(val_metrics["loss"]),
                "val_mae": format_metric(val_metrics["mae"]),
                "val_accuracy": format_metric(val_metrics["accuracy"]),
                "val_auroc": format_metric(val_metrics["auroc"]),
            }
        )

    with metrics_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_mae",
                "train_accuracy",
                "train_auroc",
                "val_loss",
                "val_mae",
                "val_accuracy",
                "val_auroc",
            ],
        )
        writer.writeheader()
        writer.writerows(history_rows)

    best_checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    test_metrics = run_epoch(
        model,
        test_loader,
        optimizer=None,
        device=device,
        threshold=config.threshold,
    )
    save_json(
        test_metrics_path,
        {
            "loss": format_metric(test_metrics["loss"]),
            "mae": format_metric(test_metrics["mae"]),
            "accuracy": format_metric(test_metrics["accuracy"]),
            "auroc": format_metric(test_metrics["auroc"]),
        },
    )

    return TrainingArtifacts(
        run_dir=run_dir,
        best_checkpoint_path=best_checkpoint_path,
        last_checkpoint_path=last_checkpoint_path,
        config_path=config_path,
        metrics_csv_path=metrics_csv_path,
        test_metrics_path=test_metrics_path,
        train_split_path=train_split_path,
        val_split_path=val_split_path,
        test_split_path=test_split_path,
    )


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = build_model(pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, config


@torch.inference_mode()
def predict_timelapse(
    checkpoint_path: Path,
    tif_path: Path,
    *,
    channel: int = 0,
    channel_count: int | None = None,
    output_csv_path: Path | None = None,
    device: str = "auto",
    threshold: float | None = None,
    batch_size: int = 64,
) -> TimelapsePredictionResult:
    resolved_tif_path = tif_path.resolve()
    resolved_checkpoint_path = checkpoint_path.resolve()
    resolved_output_csv_path = (
        output_csv_path.resolve()
        if output_csv_path is not None
        else default_scores_csv_path(resolved_tif_path, channel)
    )
    resolved_device = choose_device(device)
    model, config = load_checkpoint(resolved_checkpoint_path, device=resolved_device)
    image_size = int(config["image_size"])
    decision_threshold = float(threshold if threshold is not None else config.get("threshold", DEFAULT_THRESHOLD))
    frames = extract_timelapse_frames(
        resolved_tif_path,
        channel=channel,
        channel_count=channel_count,
    )

    rows: list[TimelapsePredictionRow] = []
    for batch_start in range(0, frames.shape[0], batch_size):
        batch_frames = frames[batch_start : batch_start + batch_size]
        batch_tensor = torch.stack(
            [preprocess_image_array(frame, image_size=image_size) for frame in batch_frames],
            dim=0,
        ).to(resolved_device)
        batch_probabilities = torch.sigmoid(model(batch_tensor).squeeze(1)).cpu().tolist()
        for frame_offset, probability in enumerate(batch_probabilities):
            time_index = batch_start + frame_offset
            hard_label = "dead" if float(probability) >= decision_threshold else "live"
            rows.append(
                TimelapsePredictionRow(
                    time_index=time_index,
                    dead_probability=float(probability),
                    hard_label=hard_label,
                )
            )

    resolved_output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_output_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["time_index", "dead_probability", "predicted_label", "input_tif", "channel"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "time_index": row.time_index,
                    "dead_probability": f"{row.dead_probability:.6f}",
                    "predicted_label": row.hard_label,
                    "input_tif": str(resolved_tif_path),
                    "channel": channel,
                }
            )

    return TimelapsePredictionResult(
        input_path=resolved_tif_path,
        checkpoint_path=resolved_checkpoint_path,
        output_csv_path=resolved_output_csv_path,
        channel=channel,
        frame_count=len(rows),
        threshold=decision_threshold,
        rows=rows,
    )


@torch.inference_mode()
def predict_single_image(
    checkpoint_path: Path,
    image_path: Path,
    *,
    device: str = "auto",
    threshold: float | None = None,
) -> PredictionResult:
    timelapse_prediction = predict_timelapse(
        checkpoint_path=checkpoint_path,
        tif_path=image_path,
        channel=0,
        channel_count=1,
        device=device,
        threshold=threshold,
    )
    if timelapse_prediction.frame_count != 1:
        raise ValueError(
            f"{image_path} produced {timelapse_prediction.frame_count} frames; "
            "use predict_timelapse for multi-frame TIFFs."
        )
    row = timelapse_prediction.rows[0]
    return PredictionResult(
        image_path=timelapse_prediction.input_path,
        dead_probability=row.dead_probability,
        hard_label=row.hard_label,
        checkpoint_path=timelapse_prediction.checkpoint_path,
        threshold=timelapse_prediction.threshold,
    )


def build_train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a soft-target binary ResNet18 on bright-field TIFF frames using "
            "dead_probability from labels.csv."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use ImageNet-pretrained ResNet18 weights. Disable with --no-pretrained.",
    )
    return parser


def build_infer_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run timelapse inference with a trained bright-field ResNet checkpoint. "
            "Single-frame TIFFs are treated as a one-frame timelapse."
        )
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("tif", type=Path)
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--channel-count", type=int, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser


def build_plot_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot a per-frame dead-probability score series from inference CSV output."
    )
    parser.add_argument("scores_csv", type=Path)
    parser.add_argument("--output-png", type=Path, default=None)
    parser.add_argument("--title", default=None)
    return parser


def plot_score_series(
    scores_csv_path: Path,
    *,
    output_png_path: Path | None = None,
    title: str | None = None,
) -> Path:
    resolved_scores_csv = scores_csv_path.resolve()
    resolved_output_png = (
        output_png_path.resolve()
        if output_png_path is not None
        else default_scores_plot_path(resolved_scores_csv)
    )

    time_indices: list[int] = []
    dead_probabilities: list[float] = []
    with resolved_scores_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            time_indices.append(int(row["time_index"]))
            dead_probabilities.append(float(row["dead_probability"]))
    if not time_indices:
        raise ValueError(f"No rows found in {resolved_scores_csv}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(10, 4))
    axis.plot(time_indices, dead_probabilities, color="#c42121", linewidth=2)
    axis.axhline(DEFAULT_THRESHOLD, color="#555555", linestyle="--", linewidth=1)
    axis.set_xlabel("Time Index")
    axis.set_ylabel("Dead Probability")
    axis.set_ylim(-0.02, 1.02)
    axis.set_title(title or resolved_scores_csv.stem)
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    resolved_output_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(resolved_output_png, dpi=160)
    plt.close(figure)
    return resolved_output_png


def train_main(argv: list[str] | None = None) -> None:
    args = build_train_parser().parse_args(argv)
    config = TrainingConfig(
        dataset_root=args.dataset_root.resolve(),
        artifact_root=args.artifact_root.resolve(),
        run_name=args.run_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        num_workers=args.num_workers,
        threshold=args.threshold,
        pretrained=args.pretrained,
        device=args.device,
    )
    artifacts = train_model(config)
    print(f"Run directory: {artifacts.run_dir}")
    print(f"Best checkpoint: {artifacts.best_checkpoint_path}")
    print(f"Last checkpoint: {artifacts.last_checkpoint_path}")
    print(f"Metrics CSV: {artifacts.metrics_csv_path}")
    print(f"Test metrics JSON: {artifacts.test_metrics_path}")


def infer_main(argv: list[str] | None = None) -> None:
    args = build_infer_parser().parse_args(argv)
    prediction = predict_timelapse(
        checkpoint_path=args.checkpoint,
        tif_path=args.tif,
        channel=args.channel,
        channel_count=args.channel_count,
        output_csv_path=args.output_csv,
        device=args.device,
        threshold=args.threshold,
        batch_size=args.batch_size,
    )
    probabilities = [row.dead_probability for row in prediction.rows]
    print(f"Input TIFF: {prediction.input_path}")
    print(f"Checkpoint: {prediction.checkpoint_path}")
    print(f"Output CSV: {prediction.output_csv_path}")
    print(f"Channel: {prediction.channel}")
    print(f"Frames scored: {prediction.frame_count}")
    print(f"Threshold: {prediction.threshold:.3f}")
    print(f"Min dead probability: {min(probabilities):.6f}")
    print(f"Max dead probability: {max(probabilities):.6f}")
    print(f"Mean dead probability: {sum(probabilities) / len(probabilities):.6f}")


def plot_main(argv: list[str] | None = None) -> None:
    args = build_plot_parser().parse_args(argv)
    output_path = plot_score_series(
        args.scores_csv,
        output_png_path=args.output_png,
        title=args.title,
    )
    print(f"Wrote plot: {output_path}")
