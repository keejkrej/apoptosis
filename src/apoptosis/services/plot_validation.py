from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from apoptosis.services.inference import POSITION_META, CellInference

TIME_INTERVAL_MIN = 10  # minutes per timepoint
SIGNAL_LEGEND_COLOR = "0.35"
PANEL_LABEL_FONTSIZE = 18
AXIS_LABEL_FONTSIZE = 13
TICK_LABEL_FONTSIZE = 11
LEGEND_FONTSIZE_A = 10
LEGEND_FONTSIZE_B = 11


def _style_axes(ax: plt.Axes, *, xlabel: str | None = None, ylabel: str | None = None) -> None:
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_FONTSIZE)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)


def _panel_a_legend_handles(results: list[CellInference]) -> list[Line2D]:
    signal_handles = [
        Line2D(
            [0],
            [0],
            color=SIGNAL_LEGEND_COLOR,
            linestyle="--",
            linewidth=1.2,
            label="Viability P(dead)",
        ),
        Line2D(
            [0],
            [0],
            color=SIGNAL_LEGEND_COLOR,
            linestyle=":",
            linewidth=1.8,
            label="Toto-3",
        ),
        Line2D(
            [0],
            [0],
            color=SIGNAL_LEGEND_COLOR,
            linestyle="-",
            linewidth=2.5,
            alpha=0.7,
            label="Death time from morphology",
        ),
        Line2D(
            [0],
            [0],
            color=SIGNAL_LEGEND_COLOR,
            linestyle="-",
            linewidth=2.0,
            alpha=0.5,
            label="Death time from Toto-3 fluorescence",
        ),
    ]
    sample_handles = [
        Line2D(
            [0],
            [0],
            color=meta["color"],
            linestyle="-",
            linewidth=2.0,
            label=meta["label"],
        )
        for position, meta in POSITION_META.items()
        if any(cell.position == position for cell in results)
    ]
    return signal_handles + sample_handles


def plot_validation(
    results: list[CellInference],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    ax_a = axes[0]
    ax_a.text(
        -0.12,
        1.05,
        "A",
        transform=ax_a.transAxes,
        fontsize=PANEL_LABEL_FONTSIZE,
        fontweight="bold",
    )
    time_axis = np.arange(len(results[0].death_probability)) * TIME_INTERVAL_MIN / 60  # hours
    for position, meta in POSITION_META.items():
        cells = [cell for cell in results if cell.position == position]
        if not cells:
            continue
        # Pick one example cell per position (instead of median across all)
        example = cells[0]
        color = meta["color"]
        ax_a.plot(
            time_axis,
            example.death_probability,
            color=color,
            linewidth=1.2,
            linestyle="--",
            alpha=0.9,
        )

    _style_axes(ax_a, xlabel="Time (h)", ylabel="Viability P(dead)")
    ax_a.set_xlim(0, time_axis[-1])
    ax_a.set_ylim(-0.05, 1.05)

    ax_a2 = ax_a.twinx()
    for position, meta in POSITION_META.items():
        cells = [cell for cell in results if cell.position == position]
        if not cells:
            continue
        example = cells[0]
        color = meta["color"]
        ax_a2.plot(
            time_axis,
            example.toto_raw,
            color=color,
            linewidth=1.8,
            linestyle=":",
        )

    _style_axes(ax_a2, ylabel="Toto-3")

    # Inferred death times as solid vertical lines on the primary axis
    for position, meta in POSITION_META.items():
        cells = [cell for cell in results if cell.position == position]
        if not cells:
            continue
        example = cells[0]
        color = meta["color"]
        death_h = example.death_time_viability * TIME_INTERVAL_MIN / 60
        if example.death_time_viability < example.timepoints:
            ax_a.axvline(death_h, color=color, linestyle="-", alpha=0.7, linewidth=2.5)
        toto_h = example.death_time_toto * TIME_INTERVAL_MIN / 60
        if example.death_time_toto < example.timepoints:
            ax_a.axvline(toto_h, color=color, linestyle="-", alpha=0.5, linewidth=2.0)

    legend_handles = _panel_a_legend_handles(results)
    ax_a.legend(
        legend_handles,
        [handle.get_label() for handle in legend_handles],
        fontsize=LEGEND_FONTSIZE_A,
        loc="upper left",
        frameon=False,
    )

    ax_b = axes[1]
    ax_b.text(
        -0.12,
        1.05,
        "B",
        transform=ax_b.transAxes,
        fontsize=PANEL_LABEL_FONTSIZE,
        fontweight="bold",
    )

    max_time = max(cell.timepoints for cell in results)
    for position, meta in POSITION_META.items():
        cells = [cell for cell in results if cell.position == position]
        if not cells:
            continue
        # Filter out cells that die very late (> 200) as requested
        cells = [c for c in cells if c.death_time_viability <= 200 and c.death_time_toto <= 200]
        if not cells:
            continue
        x = np.array([cell.death_time_viability for cell in cells], dtype=np.float32) * TIME_INTERVAL_MIN / 60
        y = np.array([cell.death_time_toto for cell in cells], dtype=np.float32) * TIME_INTERVAL_MIN / 60
        ax_b.scatter(
            x,
            y,
            s=18,
            alpha=0.65,
            color=meta["color"],
            label=meta["label"],
            edgecolors="none",
        )

    lim = (max_time * TIME_INTERVAL_MIN / 60) + 0.5
    ax_b.plot([0, lim], [0, lim], color="black", linewidth=1.0, zorder=0)
    ax_b.set_xlim(0, lim)
    ax_b.set_ylim(0, lim)
    _style_axes(
        ax_b,
        xlabel="Death time from morphology (h)",
        ylabel="Death time from Toto-3 fluorescence (h)",
    )
    ax_b.set_aspect("equal", adjustable="box")
    ax_b.legend(fontsize=LEGEND_FONTSIZE_B, loc="upper left", frameon=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
