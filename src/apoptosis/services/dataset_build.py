from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from apoptosis.core.labels import LabelStore
from apoptosis.core.roi import RoiRef, timepoint_count
from apoptosis.core.session import DEFAULT_DATA_DIR, DEFAULT_LABELS_PATH
from apoptosis.core.viability import frame_label


@dataclass(frozen=True)
class FrameSample:
    position: str
    roi_id: int
    time_index: int
    label: int
    split: str

    @property
    def key(self) -> str:
        return f"{self.position}/Roi{self.roi_id}"


@dataclass(frozen=True)
class DatasetManifest:
    data_dir: str
    labels_path: str
    train_cells: int
    val_cells: int
    train_samples: int
    val_samples: int
    viable_train: int
    dead_train: int
    samples: list[FrameSample]


def build_dataset_manifest(
    data_dir: Path,
    labels_path: Path,
    output_dir: Path,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> DatasetManifest:
    store = LabelStore(labels_path)
    labels = store.all_labels()
    if not labels:
        msg = f"No labels found in {labels_path}"
        raise ValueError(msg)

    cell_keys = sorted({f"{label.position}/Roi{label.roi_id}" for label in labels})
    rng = random.Random(seed)
    shuffled = cell_keys.copy()
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_fraction)))
    val_cells = set(shuffled[:val_count])

    samples: list[FrameSample] = []
    for label in labels:
        roi = RoiRef(position=label.position, roi_id=label.roi_id)
        timepoints = timepoint_count(data_dir, roi)
        split = "val" if roi.key in val_cells else "train"
        for time_index in range(timepoints):
            samples.append(
                FrameSample(
                    position=label.position,
                    roi_id=label.roi_id,
                    time_index=time_index,
                    label=frame_label(label.death_frame, time_index, timepoints),
                    split=split,
                )
            )

    train_samples = [sample for sample in samples if sample.split == "train"]
    val_samples = [sample for sample in samples if sample.split == "val"]
    viable_train = sum(sample.label == 1 for sample in train_samples)

    manifest = DatasetManifest(
        data_dir=str(data_dir),
        labels_path=str(labels_path),
        train_cells=len(cell_keys) - len(val_cells),
        val_cells=len(val_cells),
        train_samples=len(train_samples),
        val_samples=len(val_samples),
        viable_train=viable_train,
        dead_train=len(train_samples) - viable_train,
        samples=samples,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    payload = {
        "data_dir": manifest.data_dir,
        "labels_path": manifest.labels_path,
        "train_cells": manifest.train_cells,
        "val_cells": manifest.val_cells,
        "train_samples": manifest.train_samples,
        "val_samples": manifest.val_samples,
        "viable_train": manifest.viable_train,
        "dead_train": manifest.dead_train,
        "samples": [asdict(sample) for sample in manifest.samples],
    }
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n")
    return manifest


def default_dataset_dir(project_root: Path | None = None) -> Path:
    root = project_root or Path(__file__).resolve().parents[3]
    return root / "datasets" / "viability"


def default_data_and_labels() -> tuple[Path, Path]:
    return DEFAULT_DATA_DIR, DEFAULT_LABELS_PATH
