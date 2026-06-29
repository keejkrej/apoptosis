from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import tifffile

from apoptosis.core.roi import CHANNEL_BRIGHTFIELD, CHANNEL_TOTO, RoiRef, frame_index, roi_path


def _box_mean_2d(image: np.ndarray, *, radius: int) -> np.ndarray:
    if radius < 0:
        raise ValueError(f"Variation radius must be >= 0, got {radius}")
    if radius == 0:
        return image.astype(np.float64, copy=False)
    window = radius * 2 + 1
    padded = np.pad(image.astype(np.float64, copy=False), ((radius, radius), (radius, radius)), mode="edge")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    summed = (
        integral[window:, window:]
        - integral[:-window, window:]
        - integral[window:, :-window]
        + integral[:-window, :-window]
    )
    return summed / float(window * window)


def variation_filter_2d(image: np.ndarray, *, radius: int) -> np.ndarray:
    values = image.astype(np.float64, copy=False)
    mean = _box_mean_2d(values, radius=radius)
    mean_square = _box_mean_2d(values * values, radius=radius)
    variance = np.maximum(mean_square - mean * mean, 0.0)
    return np.sqrt(variance)


def _gaussian_kernel_1d(sigma: float) -> np.ndarray:
    if sigma <= 0:
        return np.array([1.0], dtype=np.float64)
    radius = max(1, int(np.ceil(sigma * 3.0)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(x * x) / (2.0 * sigma * sigma))
    return kernel / kernel.sum()


def _convolve_axis_reflect(image: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    pad = len(kernel) // 2
    if pad == 0:
        return image
    pad_width = [(0, 0)] * image.ndim
    pad_width[axis] = (pad, pad)
    padded = np.pad(image, pad_width, mode="edge")
    return np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="valid"), axis, padded)


def gaussian_filter_2d(image: np.ndarray, *, sigma: float) -> np.ndarray:
    if sigma < 0:
        raise ValueError(f"Gaussian sigma must be >= 0, got {sigma}")
    kernel = _gaussian_kernel_1d(sigma)
    smoothed = _convolve_axis_reflect(image.astype(np.float64, copy=False), kernel, axis=0)
    return _convolve_axis_reflect(smoothed, kernel, axis=1)


def otsu_threshold(image: np.ndarray, *, bins: int = 256) -> float:
    values = image[np.isfinite(image)].astype(np.float64, copy=False)
    if values.size == 0:
        raise ValueError("Cannot compute Otsu threshold for an empty image")
    min_value = float(values.min())
    max_value = float(values.max())
    if min_value == max_value:
        return min_value
    hist, edges = np.histogram(values, bins=bins, range=(min_value, max_value))
    centers = (edges[:-1] + edges[1:]) * 0.5
    weight_foreground = np.cumsum(hist).astype(np.float64)
    weight_background = float(values.size) - weight_foreground
    intensity_sum = np.cumsum(hist * centers)
    total_intensity_sum = intensity_sum[-1]
    valid = (weight_foreground > 0) & (weight_background > 0)
    if not np.any(valid):
        return min_value
    mean_foreground = np.zeros_like(centers)
    mean_background = np.zeros_like(centers)
    mean_foreground[valid] = intensity_sum[valid] / weight_foreground[valid]
    mean_background[valid] = (total_intensity_sum - intensity_sum[valid]) / weight_background[valid]
    variance = np.zeros_like(centers)
    variance[valid] = (
        weight_foreground[valid]
        * weight_background[valid]
        * np.square(mean_foreground[valid] - mean_background[valid])
    )
    return float(centers[int(np.argmax(variance))])


def fill_binary_holes_2d(mask: np.ndarray) -> np.ndarray:
    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape={mask_bool.shape}")
    background = ~mask_bool
    exterior = np.zeros(mask_bool.shape, dtype=bool)
    stack: list[tuple[int, int]] = []
    height, width = mask_bool.shape
    for x in range(width):
        if background[0, x]:
            stack.append((0, x))
        if height > 1 and background[height - 1, x]:
            stack.append((height - 1, x))
    for y in range(height):
        if background[y, 0]:
            stack.append((y, 0))
        if width > 1 and background[y, width - 1]:
            stack.append((y, width - 1))
    while stack:
        y, x = stack.pop()
        if exterior[y, x] or not background[y, x]:
            continue
        exterior[y, x] = True
        if y > 0:
            stack.append((y - 1, x))
        if y + 1 < height:
            stack.append((y + 1, x))
        if x > 0:
            stack.append((y, x - 1))
        if x + 1 < width:
            stack.append((y, x + 1))
    holes = background & ~exterior
    return mask_bool | holes


def segment_frame(
    frame: np.ndarray,
    *,
    variation_radius: int = 2,
    gaussian_sigma: float = 1.0,
) -> np.ndarray:
    varied = variation_filter_2d(frame, radius=variation_radius)
    smoothed = gaussian_filter_2d(varied, sigma=gaussian_sigma)
    threshold = otsu_threshold(smoothed)
    return fill_binary_holes_2d(smoothed > threshold)

BASELINE_FRAMES = 10
SUSTAINED_FRAMES = 3
TOTO_POST_WINDOW = 3
TOTO_MIN_CONTRAST_DELTA = 400_000.0
TOTO_MIN_RAW_STEP = 500_000.0


def toto_trace(data_dir: Path, roi: RoiRef) -> np.ndarray:
    """Compute bg-corrected Toto-3 signal using cell mask from brightfield.

    For each timepoint:
      - Segment cell mask from BF channel (variation + gaussian + otsu)
      - On Toto channel: sum(foreground) - median(background) * area
    This follows the mask + signal channel pattern for more precise cell-specific measurement.
    """
    stack = tifffile.imread(roi_path(data_dir, roi))
    n_times = len(stack) // 2
    signals = []
    for t in range(n_times):
        bf = stack[frame_index(t, CHANNEL_BRIGHTFIELD)].astype(np.float64)
        toto = stack[frame_index(t, CHANNEL_TOTO)].astype(np.float64)

        mask = segment_frame(bf, variation_radius=2, gaussian_sigma=1.0)
        area = int(mask.sum())
        if area == 0:
            signals.append(0.0)
            continue

        fg = toto[mask]
        bg_pixels = toto[~mask]
        background = float(np.median(bg_pixels)) if bg_pixels.size > 0 else 0.0
        intensity = float(fg.sum())
        corrected = intensity - area * background
        signals.append(corrected)
    return np.array(signals, dtype=np.float32)


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


def _detrend(x: np.ndarray) -> np.ndarray:
    """Remove linear trend (bleaching/decay) from the signal."""
    t = np.arange(len(x))
    p = np.polyfit(t, x, 1)
    return x - np.polyval(p, t)

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
    contrast: np.ndarray,
    timepoints: int,
    raw_trace: np.ndarray | None = None,
    min_raw_step: float = TOTO_MIN_RAW_STEP,
    fold_threshold: float = 2.0,
    baseline_frames: int = BASELINE_FRAMES,
    sustained: int = SUSTAINED_FRAMES,
    min_contrast_delta: float = TOTO_MIN_CONTRAST_DELTA,
    post_window: int = TOTO_POST_WINDOW,
) -> int:
    """Detect death as the first sustained step increase in toto-3 intensity.

    Uses position-contrast (raw - pos median) for robust timing, with detrending.
    Scans forward for the earliest frame where contrast rises and the raw
    bg-corrected trace shows a matching amplitude jump (not the largest step
    later in the trace).
    """
    contrast = np.asarray(contrast, dtype=np.float32)
    if len(contrast) == 0:
        return timepoints

    dcontrast = _detrend(contrast).astype(np.float32)

    if not _has_intensity_jump(dcontrast, rel_thresh=1.1, abs_thresh=150.0):
        return timepoints

    raw = np.asarray(raw_trace, dtype=np.float32) if raw_trace is not None else None
    pre_w = sustained
    min_t = baseline_frames

    for t in range(min_t, len(dcontrast) - post_window):
        delta = float(
            np.mean(dcontrast[t : t + post_window])
            - np.mean(dcontrast[t - pre_w : t])
        )
        if delta < min_contrast_delta:
            continue
        if raw is not None:
            pre = max(0, t - pre_w)
            post = min(len(raw), t + post_window + 1)
            raw_step = float(np.mean(raw[t:post]) - np.mean(raw[pre:t]))
            if raw_step < min_raw_step:
                continue
        if t >= timepoints - 3:
            return timepoints
        return int(t)

    return timepoints


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
