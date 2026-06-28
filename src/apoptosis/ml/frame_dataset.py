from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import override

import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset

from apoptosis.core.roi import CHANNEL_BRIGHTFIELD, RoiRef, frame_index, roi_path
from apoptosis.ml.preprocess import IMAGE_SIZE, frame_to_tensor, normalize_frame

__all__ = [
    "IMAGE_SIZE",
    "ManifestSample",
    "ViabilityFrameDataset",
    "load_manifest_samples",
    "normalize_frame",
]


@dataclass(frozen=True)
class ManifestSample:
    position: str
    roi_id: int
    time_index: int
    label: int
    split: str


def load_manifest_samples(manifest_path: Path, split: str) -> list[ManifestSample]:
    payload = json.loads(manifest_path.read_text())
    return [
        ManifestSample(**item) for item in payload["samples"] if item["split"] == split
    ]


class ViabilityFrameDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, data_dir: Path, samples: list[ManifestSample]) -> None:
        self.data_dir = data_dir
        self.samples = samples
        self._stack_cache: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def _load_stack(self, roi: RoiRef) -> np.ndarray:
        if roi.key not in self._stack_cache:
            self._stack_cache[roi.key] = tifffile.imread(roi_path(self.data_dir, roi))
        return self._stack_cache[roi.key]

    @override
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        roi = RoiRef(position=sample.position, roi_id=sample.roi_id)
        stack = self._load_stack(roi)
        frame = stack[frame_index(sample.time_index, CHANNEL_BRIGHTFIELD)].copy()
        tensor = frame_to_tensor(frame)
        label = torch.tensor(sample.label, dtype=torch.long)
        return tensor, label
