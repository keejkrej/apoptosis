from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from apoptosis.core.roi import RoiRef, healthy_sentinel_time


@dataclass
class CellLabel:
    position: str
    roi_id: int
    death_frame: int
    labeled_at: str

    @property
    def key(self) -> str:
        return f"{self.position}/Roi{self.roi_id}"


class LabelStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._labels: dict[str, CellLabel] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        raw = json.loads(self.path.read_text())
        for item in raw:
            label = CellLabel(**item)
            self._labels[label.key] = label

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            asdict(label)
            for label in sorted(self._labels.values(), key=lambda item: item.key)
        ]
        self.path.write_text(json.dumps(payload, indent=2) + "\n")

    def get(self, roi: RoiRef) -> CellLabel | None:
        return self._labels.get(roi.key)

    def set_death_frame(
        self,
        data_dir: Path,
        roi: RoiRef,
        death_frame: int,
    ) -> CellLabel:
        sentinel = healthy_sentinel_time(data_dir, roi)
        if death_frame < 0 or death_frame > sentinel:
            msg = f"death_frame must be between 0 and {sentinel}"
            raise ValueError(msg)
        label = CellLabel(
            position=roi.position,
            roi_id=roi.roi_id,
            death_frame=death_frame,
            labeled_at=datetime.now(UTC).isoformat(),
        )
        self._labels[roi.key] = label
        self.save()
        return label

    def all_labels(self) -> list[CellLabel]:
        return sorted(self._labels.values(), key=lambda label: label.key)

    def labeled_count(self) -> int:
        return len(self._labels)
