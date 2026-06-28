from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

CHANNEL_BRIGHTFIELD = 0
CHANNEL_TOTO = 1
FRAMES_PER_TIMEPOINT = 2


@dataclass(frozen=True)
class RoiRef:
    position: str
    roi_id: int

    @property
    def key(self) -> str:
        return f"{self.position}/Roi{self.roi_id}"

    @property
    def filename(self) -> str:
        return f"Roi{self.roi_id}.tif"


def _roi_id_from_name(name: str) -> int:
    match = re.search(r"Roi(\d+)", name)
    if match is None:
        msg = f"Could not parse ROI id from {name}"
        raise ValueError(msg)
    return int(match.group(1))


def discover_rois(data_dir: Path) -> list[RoiRef]:
    roi_root = data_dir / "roi"
    rois: list[RoiRef] = []
    for position_dir in sorted(roi_root.iterdir()):
        if not position_dir.is_dir() or not position_dir.name.startswith("Pos"):
            continue
        for path in sorted(
            position_dir.glob("Roi*.tif"),
            key=lambda p: _roi_id_from_name(p.name),
        ):
            roi_id = _roi_id_from_name(path.name)
            rois.append(RoiRef(position=position_dir.name, roi_id=roi_id))
    return rois


def roi_path(data_dir: Path, roi: RoiRef) -> Path:
    return data_dir / "roi" / roi.position / roi.filename


def timepoint_count(data_dir: Path, roi: RoiRef) -> int:
    with tifffile.TiffFile(roi_path(data_dir, roi)) as tif:
        frame_count = len(tif.pages)
    if frame_count % FRAMES_PER_TIMEPOINT != 0:
        msg = f"Unexpected frame count {frame_count} for {roi.key}"
        raise ValueError(msg)
    return frame_count // FRAMES_PER_TIMEPOINT


def frame_index(time_index: int, channel: int) -> int:
    return time_index * FRAMES_PER_TIMEPOINT + channel


def load_frame(
    data_dir: Path,
    roi: RoiRef,
    time_index: int,
    channel: int = CHANNEL_BRIGHTFIELD,
) -> np.ndarray:
    index = frame_index(time_index, channel)
    with tifffile.TiffFile(roi_path(data_dir, roi)) as tif:
        if index >= len(tif.pages):
            msg = f"Frame {index} out of range for {roi.key} ({len(tif.pages)} frames)"
            raise IndexError(msg)
        return tif.pages[index].asarray()


def frame_to_png(
    data_dir: Path,
    roi: RoiRef,
    time_index: int,
    channel: int = CHANNEL_BRIGHTFIELD,
) -> bytes:
    frame = load_frame(data_dir, roi, time_index, channel)
    low, high = np.percentile(frame, (1, 99))
    if high <= low:
        high = low + 1
    scaled = np.clip((frame.astype(np.float32) - low) / (high - low), 0, 1)
    image = Image.fromarray((scaled * 255).astype(np.uint8), mode="L")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def healthy_label_value(data_dir: Path, roi: RoiRef) -> int:
    """Stored death_frame value meaning the cell stayed healthy."""
    return timepoint_count(data_dir, roi)


def is_healthy_label(death_frame: int, timepoints: int) -> bool:
    return death_frame >= timepoints
