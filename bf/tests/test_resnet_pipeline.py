from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import tifffile
import torch

from apoptosis_bf.resnet_pipeline import (
    TrainingConfig,
    load_manifest,
    predict_single_image,
    preprocess_tiff_image,
    split_records_by_roi,
    train_model,
)


def build_image(dead_probability: float) -> np.ndarray:
    base = np.full((16, 16), int(dead_probability * 40000), dtype=np.uint16)
    gradient = np.arange(16, dtype=np.uint16).reshape(16, 1) * 200
    return base + gradient


def write_synthetic_dataset(dataset_root: Path, roi_count: int = 10) -> list[Path]:
    images_root = dataset_root / "images"
    for folder in ("live", "mixed", "dead"):
        (images_root / folder).mkdir(parents=True, exist_ok=True)

    labels_path = dataset_root / "labels.csv"
    sample_images: list[Path] = []
    with labels_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
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
            ],
        )
        writer.writeheader()
        for roi in range(roi_count):
            for time_index, dead_probability in enumerate((0.0, 0.5, 1.0)):
                if dead_probability <= 0.0:
                    split_folder = "live"
                elif dead_probability >= 1.0:
                    split_folder = "dead"
                else:
                    split_folder = "mixed"
                file_name = f"Pos0_Roi{roi}_T{time_index:03d}.tif"
                image_path = images_root / split_folder / file_name
                tifffile.imwrite(image_path, build_image(dead_probability))
                if roi == 0 and time_index == 0:
                    sample_images.append(image_path)
                writer.writerow(
                    {
                        "split_folder": split_folder,
                        "image_relpath": str(Path("images") / split_folder / file_name).replace("/", "\\"),
                        "position": "Pos0",
                        "roi": roi,
                        "time_index": time_index,
                        "source_tif": f"synthetic_roi_{roi}.tif",
                        "live_anchor_t": 0,
                        "dead_anchor_t": 2,
                        "dead_probability": f"{dead_probability:.6f}",
                        "annotation_mode": "live_to_dead",
                    }
                )
    return sample_images


def test_split_records_by_roi_keeps_groups_isolated(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    write_synthetic_dataset(dataset_root, roi_count=10)
    records = load_manifest(dataset_root)

    split_map = split_records_by_roi(records, seed=7)
    train_groups = {record.roi_group for record in split_map["train"]}
    val_groups = {record.roi_group for record in split_map["val"]}
    test_groups = {record.roi_group for record in split_map["test"]}

    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)
    assert len(train_groups | val_groups | test_groups) == 10


def test_preprocess_tiff_image_outputs_resnet_tensor(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.tif"
    tifffile.imwrite(image_path, build_image(0.5))

    tensor = preprocess_tiff_image(image_path, image_size=64)

    assert tensor.shape == (3, 64, 64)
    assert tensor.dtype == torch.float32
    assert bool(tensor.isfinite().all())


def test_train_model_and_single_image_inference_smoke(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    sample_images = write_synthetic_dataset(dataset_root, roi_count=10)
    artifact_root = tmp_path / "artifacts"

    artifacts = train_model(
        TrainingConfig(
            dataset_root=dataset_root,
            artifact_root=artifact_root,
            run_name="testrun",
            epochs=1,
            batch_size=4,
            image_size=64,
            lr=1e-3,
            weight_decay=0.0,
            seed=123,
            num_workers=0,
            pretrained=False,
            device="cpu",
        )
    )

    assert artifacts.run_dir.exists()
    assert artifacts.best_checkpoint_path.exists()
    assert artifacts.last_checkpoint_path.exists()
    assert artifacts.metrics_csv_path.exists()
    assert artifacts.test_metrics_path.exists()
    assert artifacts.train_split_path.exists()
    assert artifacts.val_split_path.exists()
    assert artifacts.test_split_path.exists()

    prediction = predict_single_image(
        checkpoint_path=artifacts.best_checkpoint_path,
        image_path=sample_images[0],
        device="cpu",
    )
    assert 0.0 <= prediction.dead_probability <= 1.0
    assert prediction.hard_label in {"live", "dead"}
