from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import nd2
import numpy as np
import pandas as pd
import typer


DEFAULT_QUARTILES = "0.10,0.25,0.50,0.75,0.90"
DEFAULT_CORRECTED_QUANTILE = 0.25


@dataclass(frozen=True)
class RoiBox:
    roi: int
    x: int
    y: int
    w: int
    h: int


@dataclass(frozen=True)
class FrameLookup:
    sequence_axes: tuple[str, ...]
    index_by_coords: dict[tuple[int, ...], int]


app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help=(
        "Read an ND2 file directly, sum stain-channel intensity within ROI "
        "bounding boxes, apply quantile-based background correction, and "
        "write a long-form CSV."
    ),
)


def read_bbox_csv(csv_path: Path) -> list[RoiBox]:
    rows: list[RoiBox] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        expected = {"roi", "x", "y", "w", "h"}
        if reader.fieldnames is None or set(reader.fieldnames) != expected:
            raise ValueError(
                f"{csv_path} must contain exactly the columns roi,x,y,w,h; got {reader.fieldnames}"
            )
        for row in reader:
            rows.append(
                RoiBox(
                    roi=int(row["roi"]),
                    x=int(row["x"]),
                    y=int(row["y"]),
                    w=int(row["w"]),
                    h=int(row["h"]),
                )
            )
    if not rows:
        raise ValueError(f"No ROI rows found in {csv_path}")
    return rows


def quantile_column_name(quartile: float) -> str:
    quartile_pct = quartile * 100.0
    if abs(quartile_pct - round(quartile_pct)) > 1e-9:
        raise ValueError(
            f"Quartiles must map to integer percentage column names, got {quartile}"
        )
    return f"q{int(round(quartile_pct))}"


def parse_quartiles(quartiles: str) -> list[float]:
    values: list[float] = []
    for raw_value in quartiles.split(","):
        value = float(raw_value.strip())
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Quartiles must be between 0 and 1, got {value}")
        quantile_column_name(value)
        values.append(value)
    if not values:
        raise ValueError("At least one quartile is required")
    unique_values = sorted(set(values))
    if len(unique_values) != len(values):
        raise ValueError(f"Quartiles must be unique, got {quartiles}")
    return unique_values


def validate_corrected_quantile(quartiles: list[float], corrected_quantile: float) -> None:
    if corrected_quantile not in quartiles:
        raise ValueError(
            f"Quartiles must include {corrected_quantile:.2f} so the corrected column can be computed"
        )


def build_frame_lookup(handle: Any) -> FrameLookup:
    loop_indices = tuple(handle.loop_indices)
    if not loop_indices:
        return FrameLookup(sequence_axes=(), index_by_coords={(): 0})

    sequence_axes = tuple(
        axis
        for axis in ("P", "T", "C", "Z")
        if any(axis in frame_indices for frame_indices in loop_indices)
    )
    index_by_coords = {
        tuple(frame_indices.get(axis, 0) for axis in sequence_axes): seq_index
        for seq_index, frame_indices in enumerate(loop_indices)
    }
    return FrameLookup(sequence_axes=sequence_axes, index_by_coords=index_by_coords)


def read_frame_2d(handle: Any, lookup: FrameLookup, p: int, t: int, c: int, z: int = 0) -> np.ndarray:
    coords = {"P": p, "T": t, "C": c, "Z": z}
    seq_key = tuple(coords[axis] for axis in lookup.sequence_axes)
    if seq_key not in lookup.index_by_coords:
        raise ValueError(f"No ND2 frame found for coordinates P={p}, T={t}, C={c}, Z={z}")

    seq_index = lookup.index_by_coords[seq_key]
    frame = np.asarray(handle.read_frame(seq_index))

    if "C" not in lookup.sequence_axes and handle.sizes.get("C", 1) > 1:
        if frame.ndim >= 3 and frame.shape[0] == handle.sizes["C"]:
            frame = frame[c]
        elif frame.ndim >= 3 and frame.shape[-1] == handle.sizes["C"]:
            frame = frame[..., c]
        else:
            raise ValueError(
                "Unable to locate the channel axis in ND2 frame data for in-pixel channels"
            )

    if frame.ndim == 3 and frame.shape[0] == 1:
        frame = frame[0]
    elif frame.ndim == 3 and frame.shape[-1] == 1:
        frame = frame[..., 0]

    if frame.ndim != 2:
        raise ValueError(f"Expected a 2D frame, got shape={frame.shape}")

    return np.asarray(frame)


def clip_roi(roi: RoiBox, height: int, width: int) -> tuple[slice, slice, int, int]:
    x0 = min(max(roi.x, 0), width)
    y0 = min(max(roi.y, 0), height)
    x1 = min(max(roi.x + roi.w, 0), width)
    y1 = min(max(roi.y + roi.h, 0), height)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(
            f"ROI {roi.roi} does not overlap the frame after clipping: "
            f"(x={roi.x}, y={roi.y}, w={roi.w}, h={roi.h}), frame={width}x{height}"
        )
    return slice(y0, y1), slice(x0, x1), x1 - x0, y1 - y0


def validate_indices(handle: Any, pos: int, channel: int) -> None:
    n_pos = handle.sizes.get("P", 1)
    n_chan = handle.sizes.get("C", 1)
    if pos < 0 or pos >= n_pos:
        raise ValueError(f"--pos must be between 0 and {n_pos - 1}, got {pos}")
    if channel < 0 or channel >= n_chan:
        raise ValueError(f"--channel must be between 0 and {n_chan - 1}, got {channel}")


def channel_name(handle: Any, channel: int) -> str | None:
    metadata = getattr(handle, "metadata", None)
    channels = getattr(metadata, "channels", None)
    if channels is None or channel >= len(channels):
        return None

    candidate = channels[channel]
    nested = getattr(candidate, "channel", None)
    name = getattr(nested, "name", None)
    if name is not None:
        return str(name)
    fallback_name = getattr(candidate, "name", None)
    return str(fallback_name) if fallback_name is not None else None


def relative_time_ms(handle: Any, lookup: FrameLookup, pos: int, t: int, channel: int) -> float:
    coords = {"P": pos, "T": t, "C": channel, "Z": 0}
    seq_key = tuple(coords[axis] for axis in lookup.sequence_axes)
    seq_index = lookup.index_by_coords[seq_key]
    metadata = handle.frame_metadata(seq_index)
    channels = getattr(metadata, "channels", None)
    if channels:
        selected_channel = channels[min(channel, len(channels) - 1)]
        time_info = getattr(selected_channel, "time", None)
        relative_time = getattr(time_info, "relativeTimeMs", None)
        if relative_time is not None:
            return float(relative_time)
    raise ValueError(f"ND2 frame metadata does not expose relativeTimeMs for P={pos}, T={t}")


def default_output_csv_path(bbox_csv: Path, pos: int, channel: int, output_csv: Path | None) -> Path:
    suffix = f"_pos{pos:03d}_ch{channel:03d}_timeseries"
    csv_path = output_csv or bbox_csv.with_name(f"{bbox_csv.stem}{suffix}.csv")
    return csv_path.resolve()


def compute_metrics(
    handle: Any,
    lookup: FrameLookup,
    rois: list[RoiBox],
    *,
    pos: int,
    channel: int,
    quartiles: list[float],
    corrected_quantile: float = DEFAULT_CORRECTED_QUANTILE,
) -> pd.DataFrame:
    validate_corrected_quantile(quartiles, corrected_quantile)

    n_time = handle.sizes.get("T", 1)
    sample = read_frame_2d(handle, lookup, pos, 0, channel, 0)
    height, width = sample.shape

    roi_slices = {roi.roi: clip_roi(roi, height=height, width=width) for roi in rois}
    corrected_column = quantile_column_name(corrected_quantile)
    current_channel_name = channel_name(handle, channel)

    rows: list[dict[str, int | float | str | None]] = []
    for timepoint in range(n_time):
        frame = read_frame_2d(handle, lookup, pos, timepoint, channel, 0)
        t_ms = relative_time_ms(handle, lookup, pos, timepoint, channel)
        for roi in rois:
            y_slice, x_slice, clipped_w, clipped_h = roi_slices[roi.roi]
            patch = np.asarray(frame[y_slice, x_slice], dtype=np.uint64)
            quantile_values = np.quantile(patch, quartiles, method="linear")
            metrics = {
                quantile_column_name(quartile): float(quantile_value)
                for quartile, quantile_value in zip(quartiles, np.atleast_1d(quantile_values))
            }
            sum_value = int(patch.sum(dtype=np.uint64))
            rows.append(
                {
                    "pos": pos,
                    "channel": channel,
                    "channel_name": current_channel_name,
                    "t": timepoint,
                    "t_ms": t_ms,
                    "t_min": t_ms / 60000.0,
                    "roi": roi.roi,
                    "x": roi.x,
                    "y": roi.y,
                    "w": roi.w,
                    "h": roi.h,
                    "clipped_w": clipped_w,
                    "clipped_h": clipped_h,
                    "area": int(patch.size),
                    "sum": sum_value,
                    **metrics,
                    "corrected": float(sum_value - patch.size * metrics[corrected_column]),
                }
            )

    if not rows:
        raise ValueError("No rows produced")
    return pd.DataFrame(rows).sort_values(["roi", "t"]).reset_index(drop=True)


def write_metrics_csv(df: pd.DataFrame, output_csv: Path) -> None:
    if df.empty:
        raise ValueError("No rows to write")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)


@app.command()
def cli(
    input_nd2: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        help="Path to the ND2 file.",
    ),
    bbox_csv: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        help="CSV with columns roi,x,y,w,h for one position.",
    ),
    pos: int = typer.Option(
        0,
        "--pos",
        min=0,
        help="Position index in the ND2 file.",
    ),
    channel: int = typer.Option(
        ...,
        "--channel",
        min=0,
        help="Channel index in the ND2 file.",
    ),
    output_csv: Path | None = typer.Option(
        None,
        "--output-csv",
        help="Output CSV path. Default: <bbox_stem>_posPPP_chCCC_timeseries.csv",
    ),
    quartiles: str = typer.Option(
        DEFAULT_QUARTILES,
        "--quartiles",
        help="Comma-separated quantiles to write as qXX columns.",
    ),
) -> None:
    input_nd2 = input_nd2.resolve()
    bbox_csv = bbox_csv.resolve()
    resolved_quartiles = parse_quartiles(quartiles)
    rois = read_bbox_csv(bbox_csv)
    resolved_output_csv = default_output_csv_path(
        bbox_csv=bbox_csv,
        pos=pos,
        channel=channel,
        output_csv=output_csv,
    )

    with nd2.ND2File(str(input_nd2)) as handle:
        validate_indices(handle, pos, channel)
        lookup = build_frame_lookup(handle)
        df = compute_metrics(
            handle,
            lookup,
            rois,
            pos=pos,
            channel=channel,
            quartiles=resolved_quartiles,
        )

    write_metrics_csv(df, resolved_output_csv)
    print(f"Wrote metrics CSV: {resolved_output_csv}")


def main() -> None:
    app(prog_name="apoptosis-roi-timeseries")


class _FakeHandle:
    def __init__(
        self,
        *,
        sizes: dict[str, int],
        loop_indices: tuple[dict[str, int], ...],
        frames: list[np.ndarray],
        metadata_channels: list[str] | None = None,
        relative_times_ms: list[float] | None = None,
    ) -> None:
        self.sizes = sizes
        self.loop_indices = loop_indices
        self._frames = frames
        names = metadata_channels or [f"channel_{i}" for i in range(sizes.get("C", 1))]
        self.metadata = SimpleNamespace(
            channels=[SimpleNamespace(channel=SimpleNamespace(name=name)) for name in names]
        )
        self._relative_times_ms = relative_times_ms or [
            float(i) for i in range(sizes.get("T", 1))
        ]

    def read_frame(self, index: int) -> np.ndarray:
        return self._frames[index]

    def frame_metadata(self, index: int) -> Any:
        if "T" in {axis for frame in self.loop_indices for axis in frame}:
            timepoint = self.loop_indices[index].get("T", 0)
        else:
            timepoint = 0
        t_ms = self._relative_times_ms[timepoint]
        return SimpleNamespace(
            channels=[
                SimpleNamespace(time=SimpleNamespace(relativeTimeMs=t_ms))
                for _ in range(self.sizes.get("C", 1))
            ]
        )
