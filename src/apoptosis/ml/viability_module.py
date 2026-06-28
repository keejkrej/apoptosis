from __future__ import annotations

from typing import override

import torch
import torch.nn as nn
from lightning import LightningModule
from torchmetrics.classification import Accuracy, F1Score
from torchvision.models import resnet18


class ViabilityModule(LightningModule):
    def __init__(
        self,
        lr: float = 1e-4,
        class_weights: list[float] | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        backbone = resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        backbone.fc = nn.Linear(backbone.fc.in_features, 2)
        self.model = backbone

        weight_tensor = None
        if class_weights is not None:
            weight_tensor = torch.tensor(class_weights, dtype=torch.float32)
        self.loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)

        self.train_acc = Accuracy(task="binary")
        self.val_acc = Accuracy(task="binary")
        self.val_f1 = F1Score(task="binary")

    @override
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)

    def _shared_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        stage: str,
    ) -> torch.Tensor:
        images, labels = batch
        logits = self(images)
        loss = self.loss_fn(logits, labels)
        preds = torch.argmax(logits, dim=1)

        if stage == "train":
            self.train_acc(preds, labels)
            self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log("train/acc", self.train_acc, prog_bar=True, on_epoch=True)
        else:
            self.val_acc(preds, labels)
            self.val_f1(preds, labels)
            self.log("val/loss", loss, prog_bar=True, on_epoch=True)
            self.log("val/acc", self.val_acc, prog_bar=True, on_epoch=True)
            self.log("val/f1", self.val_f1, prog_bar=True, on_epoch=True)
        return loss

    @override
    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        return self._shared_step(batch, "train")

    @override
    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        return self._shared_step(batch, "val")

    @override
    def configure_optimizers(self) -> torch.optim.Adam:
        return torch.optim.Adam(self.parameters(), lr=self.lr)
