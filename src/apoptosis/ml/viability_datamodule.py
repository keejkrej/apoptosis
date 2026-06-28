from __future__ import annotations

import json
from pathlib import Path
from typing import override

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from apoptosis.ml.frame_dataset import ViabilityFrameDataset, load_manifest_samples


class ViabilityDataModule(LightningDataModule):
    def __init__(
        self,
        manifest_path: Path,
        batch_size: int = 32,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.manifest_path = manifest_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = Path(json.loads(manifest_path.read_text())["data_dir"])

    @override
    def setup(self, stage: str | None = None) -> None:
        train_samples = load_manifest_samples(self.manifest_path, "train")
        val_samples = load_manifest_samples(self.manifest_path, "val")
        self.train_dataset = ViabilityFrameDataset(self.data_dir, train_samples)
        self.val_dataset = ViabilityFrameDataset(self.data_dir, val_samples)

    @override
    def train_dataloader(self) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    @override
    def val_dataloader(self) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def class_weights(self) -> list[float]:
        payload = json.loads(self.manifest_path.read_text())
        viable = payload["viable_train"]
        dead = payload["dead_train"]
        total = viable + dead
        if total == 0 or viable == 0 or dead == 0:
            return [1.0, 1.0]
        weight_viable = total / (2 * viable)
        weight_dead = total / (2 * dead)
        return [weight_viable, weight_dead]
