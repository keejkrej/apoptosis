from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from apoptosis.core.roi import CHANNEL_BRIGHTFIELD, frame_index

IMAGE_SIZE = 128


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    low, high = np.percentile(frame, (1, 99))
    if high <= low:
        high = low + 1
    scaled = np.clip((frame.astype(np.float32) - low) / (high - low), 0, 1)
    return scaled.astype(np.float32)


def frame_to_tensor(frame: np.ndarray) -> torch.Tensor:
    normalized = normalize_frame(frame.copy())
    rgb = np.repeat(normalized[np.newaxis, ...], 3, axis=0)
    tensor = torch.tensor(rgb, dtype=torch.float32)
    if tensor.shape[-2:] != (IMAGE_SIZE, IMAGE_SIZE):
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    return tensor


def stack_brightfield_tensor(stack: np.ndarray, time_index: int) -> torch.Tensor:
    frame = stack[frame_index(time_index, CHANNEL_BRIGHTFIELD)].copy()
    return frame_to_tensor(frame)
