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
from matplotlib.lines import Line2D
import numpy as np

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


def _panel_a_legend_handles() -> list[Line2D]:
    """Legend handles for the single-sample panel A using distinct colors."""
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
        Line2D(
            [0],
            [0],
            color=COLOR_DEATH_MORPH,
            linewidth=3.5,
            alpha=0.8,
            label=r"$T_D$ morphology",
        ),
        Line2D(
            [0],
            [0],
            color=COLOR_DEATH_TOTO,
            linewidth=3.5,
            alpha=0.8,
            label=r"$T_D$ Toto-3",
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
    ax_a2.plot(
        time_axis,
        example.toto_raw,
        color=COLOR_TOTO,
        linewidth=1.8,
    )
    _style_axes(ax_a2, ylabel="Toto-3")

    death_h = example.death_time_viability * TIME_INTERVAL_MIN / 60
    if example.death_time_viability < example.timepoints:
        ax_a.axvline(death_h, color=COLOR_DEATH_MORPH, alpha=0.8, linewidth=3.5)
    toto_h = example.death_time_toto * TIME_INTERVAL_MIN / 60
    if example.death_time_toto < example.timepoints:
        ax_a.axvline(toto_h, color=COLOR_DEATH_TOTO, alpha=0.8, linewidth=3.5)

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
    inference_path = Path("/home/jack/workspace/apoptosis/runs/viability/inference.json")
    results = load_inference(inference_path)

    out_dir = Path("/home/jack/workspace/lisca-paper/figs")
    png_path = out_dir / "fig6.png"
    svg_path = out_dir / "fig6.svg"

    plot_fig6(results, png_path)
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
    print(f"PNG: {png_path} ({png_path.stat().st_size} bytes)")
    print(f"SVG: {svg_path} ({svg_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()