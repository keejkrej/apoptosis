#!/usr/bin/env python3
"""Regenerate Figure 6 for the LISCA paper.

Standalone script: reads viability inference JSON and produces a three-panel
figure (single-cell time course + two scatter panels). Not exposed as a CLI
command.

Run with:
    uv run python scripts/plot_fig6.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
import numpy as np

from apoptosis.core.toto import BASELINE_FRAMES
from apoptosis.services.inference import POSITION_META, CellInference, load_inference

TIME_INTERVAL_MIN = 10  # minutes per timepoint
# Distinct colors for panel A signals (no longer rely on linestyle alone)
COLOR_PROB_DEAD = "#1f77b4"   # blue – viability P(dead)
COLOR_TOTO = "#ff7f0e"        # orange – Toto-3 fluorescence
COLOR_DEATH_MORPH = "#d62728" # red – death time from morphology
COLOR_DEATH_TOTO = "#9467bd"  # purple – death time from Toto-3
PANEL_LABEL_FONTSIZE = 20
AXIS_LABEL_FONTSIZE = 16
TICK_LABEL_FONTSIZE = 14
LEGEND_FONTSIZE_A = 12
LEGEND_FONTSIZE_B = 12


def _style_axes(ax: plt.Axes, *, xlabel: str | None = None, ylabel: str | None = None) -> None:
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_FONTSIZE)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("black")


def _toto_transition_interval(
    toto_raw: np.ndarray,
    death_toto: int,
    timepoints: int,
    *,
    pre_frames: int = 15,
    post_window: int = 12,
) -> tuple[int, int] | None:
    """Frame range where Toto-3 rises from pre-death baseline to post-transition level."""
    if death_toto >= timepoints:
        return None
    trace = np.asarray(toto_raw, dtype=np.float64)
    pre_start = max(BASELINE_FRAMES, death_toto - pre_frames)
    if pre_start >= death_toto:
        pre_start = max(0, death_toto - 5)
    baseline = float(np.median(trace[pre_start:death_toto]))
    window_end = min(len(trace), death_toto + post_window)
    local_peak = float(np.max(trace[death_toto:window_end]))
    if local_peak <= baseline + 1.0:
        return None
    low_thresh = baseline + 0.15 * (local_peak - baseline)
    high_thresh = baseline + 0.85 * (local_peak - baseline)
    start = death_toto
    for frame in range(death_toto - 1, pre_start - 1, -1):
        if trace[frame] < low_thresh:
            start = frame + 1
            break
    end = death_toto
    for frame in range(death_toto, window_end):
        if trace[frame] >= high_thresh:
            end = frame
            break
    else:
        end = min(death_toto + 4, len(trace) - 1)
    if end < start:
        end = start
    return start, end


def _panel_a_legend_handles() -> list[Line2D]:
    """Legend handles for the single-sample panel A using distinct colors.

    Morphology and Toto-3 death markers are drawn on the panel but omitted
    from the legend to keep it readable.
    """
    return [
        Line2D(
            [0],
            [0],
            color=COLOR_PROB_DEAD,
            linewidth=1.8,
            label="P(dead)",
        ),
        Line2D(
            [0],
            [0],
            color=COLOR_TOTO,
            linewidth=1.8,
            label="Toto-3",
        ),
    ]


def plot_fig6(
    results: list[CellInference],
    output_path: Path,
) -> None:
    """Build the revised fig 6: single-sample panel A + two-sample scatter B-i/B-ii."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

    # ---- Panel A: single sample (Pos0 = 100 nM STS) time course ----
    ax_a = axes[0]
    ax_a.text(
        -0.12,
        1.05,
        "A",
        transform=ax_a.transAxes,
        fontsize=PANEL_LABEL_FONTSIZE,
        fontweight="bold",
    )

    pos0_cells = [cell for cell in results if cell.position == "Pos0"]
    example = pos0_cells[0]
    time_axis = np.arange(len(example.death_probability)) * TIME_INTERVAL_MIN / 60  # hours

    ax_a.plot(
        time_axis,
        example.death_probability,
        color=COLOR_PROB_DEAD,
        linewidth=1.8,
        alpha=0.9,
    )
    _style_axes(ax_a, xlabel="Time (h)", ylabel="Viability P(dead)")
    ax_a.set_xlim(0, time_axis[-1])
    ax_a.set_ylim(-0.05, 1.05)

    ax_a2 = ax_a.twinx()
    for side in ("top", "right", "bottom", "left"):
        ax_a2.spines[side].set_visible(True)
        ax_a2.spines[side].set_color("black")
    ax_a2.tick_params(axis="y", labelsize=TICK_LABEL_FONTSIZE)
    ax_a2.plot(
        time_axis,
        example.toto_raw,
        color=COLOR_TOTO,
        linewidth=1.8,
    )
    _style_axes(ax_a2, ylabel="Toto-3")

    # Death markers span the full panel height but stay out of the legend.
    panel_x = transforms.blended_transform_factory(ax_a.transData, ax_a.transAxes)
    death_h = example.death_time_viability * TIME_INTERVAL_MIN / 60
    if example.death_time_viability < example.timepoints:
        ax_a.vlines(
            death_h,
            0,
            1,
            transform=panel_x,
            colors=COLOR_DEATH_MORPH,
            linewidth=1.5,
            zorder=2,
            clip_on=False,
        )
        ax_a.text(
            death_h,
            0.96,
            r"$T_D$ morphology",
            transform=panel_x,
            ha="center",
            va="top",
            fontsize=LEGEND_FONTSIZE_A,
            color=COLOR_DEATH_MORPH,
        )
    toto_interval = _toto_transition_interval(
        example.toto_raw,
        example.death_time_toto,
        example.timepoints,
    )
    if toto_interval is not None:
        start_h, end_h = (frame * TIME_INTERVAL_MIN / 60 for frame in toto_interval)
        ax_a.axvspan(
            start_h,
            end_h,
            ymin=0,
            ymax=1,
            transform=panel_x,
            color=COLOR_DEATH_TOTO,
            alpha=0.25,
            zorder=0,
            clip_on=False,
        )
        ax_a.text(
            0.5 * (start_h + end_h),
            0.06,
            r"$T_D$ Toto-3",
            transform=panel_x,
            ha="center",
            va="bottom",
            fontsize=LEGEND_FONTSIZE_A,
            color=COLOR_DEATH_TOTO,
        )

    legend_handles = _panel_a_legend_handles()
    ax_a.legend(
        legend_handles,
        [handle.get_label() for handle in legend_handles],
        fontsize=LEGEND_FONTSIZE_A,
        loc="upper left",
        frameon=True,
        fancybox=False,
        edgecolor="0.8",
        framealpha=0.9,
    )

    # ---- Panel B-i: Pos0 (100 nM STS) scatter ----
    ax_bi = axes[1]
    ax_bi.text(
        -0.12,
        1.05,
        "B",
        transform=ax_bi.transAxes,
        fontsize=PANEL_LABEL_FONTSIZE,
        fontweight="bold",
    )
    ax_bi.text(
        0.04,
        1.05,
        "i",
        transform=ax_bi.transAxes,
        fontsize=PANEL_LABEL_FONTSIZE,
        fontweight="bold",
    )
    _scatter_panel(ax_bi, pos0_cells, POSITION_META["Pos0"]["color"], label=POSITION_META["Pos0"]["label"])

    # ---- Panel B-ii: Pos28 (500 nM STS) scatter ----
    ax_bii = axes[2]
    ax_bii.text(
        0.04,
        1.05,
        "ii",
        transform=ax_bii.transAxes,
        fontsize=PANEL_LABEL_FONTSIZE,
        fontweight="bold",
    )
    pos28_cells = [cell for cell in results if cell.position == "Pos28"]
    color28 = POSITION_META["Pos28"]["color"]
    _scatter_panel(ax_bii, pos28_cells, color28, label=POSITION_META["Pos28"]["label"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".svg":
        fig.savefig(output_path, format="svg", bbox_inches="tight")
    else:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _confidence_ellipse(ax: plt.Axes, x: np.ndarray, y: np.ndarray, *, n_std: float = 2.0, **kwargs) -> None:
    """Draw a covariance-based confidence ellipse around (x, y), tilted along their correlation.

    Standard matplotlib recipe: build a unit circle scaled by sqrt(1 +/- pearson_r)
    along the +-45deg diagonal, then scale/rotate/translate it by the data's
    per-axis std and mean. The resulting tilt makes it easy to see whether the
    point cloud sits above or below the y=x line.
    """
    if x.size < 2:
        return
    cov = np.cov(x, y)
    if not np.all(np.isfinite(cov)) or cov[0, 0] <= 0 or cov[1, 1] <= 0:
        return
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    pearson = float(np.clip(pearson, -0.999, 0.999))
    radius_x = np.sqrt(1 + pearson)
    radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=radius_x * 2, height=radius_y * 2, **kwargs)
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(float(np.mean(x)), float(np.mean(y)))
    )
    ellipse.set_transform(transf + ax.transData)
    ax.add_patch(ellipse)


def _scatter_panel(
    ax: plt.Axes,
    cells: list[CellInference],
    color: str,
    *,
    label: str,
) -> None:
    """Plot death_time_toto vs death_time_viability for one sample."""
    filtered = [c for c in cells if c.death_time_viability <= 200 and c.death_time_toto <= 200]
    if filtered:
        x = np.array([c.death_time_viability for c in filtered], dtype=np.float32) * TIME_INTERVAL_MIN / 60
        y = np.array([c.death_time_toto for c in filtered], dtype=np.float32) * TIME_INTERVAL_MIN / 60
        ax.scatter(
            x,
            y,
            s=18,
            alpha=0.65,
            color=color,
            edgecolors="none",
            label=label,
        )
        _confidence_ellipse(
            ax, x, y, n_std=1.0,
            facecolor="none", edgecolor="black", linestyle="--", linewidth=1.5, zorder=3,
        )
    lim = 200 * TIME_INTERVAL_MIN / 60 + 0.5
    ax.plot([0, lim], [0, lim], color="black", linewidth=1.0, zorder=0)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    _style_axes(
        ax,
        xlabel=r"$T_D$ morphology (h)",
        ylabel=r"$T_D$ Toto-3 (h)",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=LEGEND_FONTSIZE_B, loc="upper left", frameon=True, fancybox=False, edgecolor="0.8", framealpha=0.9)


def main() -> None:
    inference_path = Path("/home/jack/workspace/lisca-killing-assay/runs/viability/inference.json")
    results = load_inference(inference_path)

    out_dir = Path("/home/jack/workspace/lisca-paper/figs")
    svg_path = out_dir / "fig6.svg"

    plot_fig6(results, svg_path)

    pos0_total = sum(1 for c in results if c.position == "Pos0")
    pos28_total = sum(1 for c in results if c.position == "Pos28")
    pos0_scatter = sum(
        1 for c in results if c.position == "Pos0" and c.death_time_viability <= 200 and c.death_time_toto <= 200
    )
    pos28_scatter = sum(
        1 for c in results if c.position == "Pos28" and c.death_time_viability <= 200 and c.death_time_toto <= 200
    )
    print(f"Pos0 total: {pos0_total}, scatter cells: {pos0_scatter}")
    print(f"Pos28 total: {pos28_total}, scatter cells: {pos28_scatter}")
    print(f"SVG: {svg_path} ({svg_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()