from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import tifffile

from apoptosis.core.roi import CHANNEL_TOTO, RoiRef, frame_index, roi_path

BASELINE_FRAMES = 10
SUSTAINED_FRAMES = 3


def toto_trace(data_dir: Path, roi: RoiRef) -> np.ndarray:
    stack = tifffile.imread(roi_path(data_dir, roi))
    frames = [
        stack[frame_index(t, CHANNEL_TOTO)].astype(np.float32)
        for t in range(len(stack) // 2)
    ]
    return np.array([frame.mean() for frame in frames], dtype=np.float32)


def toto_alive_signal(
    trace: np.ndarray, baseline_frames: int = BASELINE_FRAMES
) -> np.ndarray:
    baseline = float(np.median(trace[:baseline_frames]))
    peak = float(np.max(trace))
    denom = max(peak - baseline, 1.0)
    stain = np.clip((trace - baseline) / denom, 0.0, 1.0)
    return (1.0 - stain).astype(np.float32)


def _first_sustained_above(
    values: np.ndarray,
    threshold: float,
    sustained: int,
) -> int | None:
    run = 0
    for index, value in enumerate(values):
        if value >= threshold:
            run += 1
            if run >= sustained:
                return index - sustained + 1
        else:
            run = 0
    return None


def _has_intensity_jump(trace: np.ndarray, rel_thresh: float = 1.25, abs_thresh: float = 300.0) -> bool:
    """Return True if trace (or residual) min/max shows meaningful dynamic range that could be a jump."""
    if len(trace) == 0:
        return False
    mi = float(np.min(trace))
    ma = float(np.max(trace))
    rng = ma - mi
    if rng < abs_thresh:
        return False
    # For raw intensities (positive), also consider relative; for residuals (can be neg/pos) rng is enough
    if mi > 10:  # looks like raw intensities
        if ma / max(mi, 1.0) < rel_thresh:
            return False
    return True


def _best_upward_step_local(trace: np.ndarray, window: int = 3) -> tuple[int | None, float]:
    """Find t and delta for largest upward step using local windowed means (pre vs post window)."""
    x = np.asarray(trace, dtype=np.float64)
    N = len(x)
    if N < 2 * window + 1:
        return None, 0.0
    best_t: int | None = None
    best_delta = -np.inf
    for t in range(window, N - window):
        delta = float(np.mean(x[t : t + window]) - np.mean(x[t - window : t]))
        if delta > best_delta:
            best_delta = delta
            best_t = t
    return best_t, float(best_delta)


def _best_sustained_up_step(
    trace: np.ndarray, pre_w: int = 3, post_w: int = 8, min_t: int = 8
) -> tuple[int | None, float]:
    """Step fit preferring a sustained higher level after t (longer post window).
    Returns (t, delta_mean_post_minus_pre) or (None, 0).
    """
    x = np.asarray(trace, dtype=np.float64)
    N = len(x)
    if N < min_t + post_w + 1:
        return None, 0.0
    best_t: int | None = None
    best_delta = -np.inf
    for t in range(min_t, N - post_w):
        m_pre = float(np.mean(x[t - pre_w : t]))
        m_post = float(np.mean(x[t : t + post_w]))
        delta = m_post - m_pre
        if delta > best_delta:
            best_delta = delta
            best_t = t
    return best_t, float(best_delta)


def _best_mean_step_up(values: np.ndarray, min_seg: int = 3) -> tuple[int | None, float]:
    """Find changepoint t and (mean_after - mean_before) that best fits a low-to-high step."""
    x = np.asarray(values, dtype=np.float64)
    N = len(x)
    if N < 2 * min_seg:
        return None, 0.0
    prefix = np.cumsum(x)
    total = prefix[-1]
    best_t: int | None = None
    best_delta = -np.inf
    for t in range(min_seg, N - min_seg):
        mL = prefix[t - 1] / t
        mR = (total - prefix[t - 1]) / (N - t)
        delta = mR - mL
        if delta > best_delta:
            best_delta = delta
            best_t = t
    return best_t, float(best_delta)


def death_time_from_toto(
    trace: np.ndarray,
    timepoints: int,
    fold_threshold: float = 2.0,
    baseline_frames: int = BASELINE_FRAMES,
    sustained: int = SUSTAINED_FRAMES,
) -> int:
    """Detect death as step increase in toto-3 intensity.

    First checks min/max overall for any range (quick out), then requires a sufficiently
    strong upward local step (user suggested min/max gate + step fit). If no clear jump
    detected, cell is treated as always alive (return timepoints).
    """
    trace = np.asarray(trace, dtype=np.float32)
    if len(trace) == 0:
        return timepoints
    # Quick min/max gate per user request: if very little dynamic range, always alive
    if not _has_intensity_jump(trace, rel_thresh=1.15, abs_thresh=300.0):
        return timepoints
    t, delta = _best_sustained_up_step(trace, pre_w=sustained, post_w=8, min_t=8)
    if t is None or delta < 300.0:  # sustained step amp gate on (residual) signal
        return timepoints
    # return the step location (start of higher period); clamp reasonable
    if t >= timepoints - 3:
        return timepoints
    return int(t)


def death_time_from_probability(
    death_probability: np.ndarray,
    timepoints: int,
    threshold: float = 0.5,
    sustained: int = SUSTAINED_FRAMES,
) -> int:
    """Detect death as step increase in viability model death probability.

    First checks min/max of prob for a real jump to high values (else always alive).
    Then fits an optimal mean step-up changepoint.
    """
    probs = np.asarray(death_probability, dtype=np.float32)
    if len(probs) == 0:
        return timepoints
    pmin = float(np.min(probs))
    pmax = float(np.max(probs))
    if (pmax - pmin) < 0.25 or pmax < 0.35:
        return timepoints
    t, delta = _best_mean_step_up(probs, min_seg=sustained)
    if t is None or delta < 0.25:
        return timepoints
    if t >= timepoints - 2:
        return timepoints
    return int(t)


def position_median_traces(
    data_dir: Path, rois: list[RoiRef]
) -> dict[str, np.ndarray]:
    """Compute the per-position median toto trace (captures common field artifacts, focus shifts, etc.)."""
    by_pos: dict[str, list[np.ndarray]] = defaultdict(list)
    for roi in rois:
        tr = toto_trace(data_dir, roi)
        by_pos[roi.position].append(tr)
    medians: dict[str, np.ndarray] = {}
    for pos, traces in by_pos.items():
        if traces:
            medians[pos] = np.median(np.stack(traces, axis=0), axis=0).astype(np.float32)
    return medians
