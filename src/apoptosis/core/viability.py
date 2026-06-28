from __future__ import annotations

from apoptosis.core.roi import is_healthy_label


def frame_is_viable(death_frame: int, time_index: int, timepoints: int) -> bool:
    if is_healthy_label(death_frame, timepoints):
        return True
    return time_index < death_frame


def frame_label(death_frame: int, time_index: int, timepoints: int) -> int:
    return int(frame_is_viable(death_frame, time_index, timepoints))
