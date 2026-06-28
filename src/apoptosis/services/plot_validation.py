from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from apoptosis.services.inference import POSITION_META, CellInference


def _median_curve(cells: list[CellInference], field: str) -> np.ndarray:
    arrays = [np.asarray(getattr(cell, field), dtype=np.float32) for cell in cells]
    return np.median(np.stack(arrays), axis=0)


def plot_validation(
    results: list[CellInference],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    ax_a = axes[0]
    ax_a.text(
        -0.12, 1.05, "A", transform=ax_a.transAxes, fontsize=14, fontweight="bold"
    )
    time_axis = np.arange(len(results[0].toto_alive))
    for position, meta in POSITION_META.items():
        cells = [cell for cell in results if cell.position == position]
        if not cells:
            continue
        toto_median = _median_curve(cells, "toto_alive")
        death_median = _median_curve(cells, "death_probability")
        color = meta["color"]
        ax_a.plot(
            time_axis,
            toto_median,
            color=color,
            linewidth=1.8,
            label=f"{meta['label']} toto-3",
        )
        ax_a.plot(
            time_axis,
            death_median,
            color=color,
            linewidth=1.2,
            linestyle="--",
            alpha=0.9,
            label=f"{meta['label']} viability P(dead)",
        )

    ax_a.set_xlim(0, time_axis[-1])
    ax_a.set_ylim(-0.05, 1.05)
    ax_a.set_xlabel("Time index")
    ax_a.set_ylabel("Normalized signal / probability")
    ax_a.legend(fontsize=7, loc="upper right", frameon=False)

    ax_b = axes[1]
    ax_b.text(
        -0.12, 1.05, "B", transform=ax_b.transAxes, fontsize=14, fontweight="bold"
    )
    ax_b.set_title("TOTO-3 vs Bright field apoptosis timing", fontsize=11)

    max_time = max(cell.timepoints for cell in results)
    for position, meta in POSITION_META.items():
        cells = [cell for cell in results if cell.position == position]
        if not cells:
            continue
        x = np.array([cell.death_time_viability for cell in cells], dtype=np.float32)
        y = np.array([cell.death_time_toto for cell in cells], dtype=np.float32)
        ax_b.scatter(
            x,
            y,
            s=18,
            alpha=0.65,
            color=meta["color"],
            label=meta["label"],
            edgecolors="none",
        )

    lim = max_time + 5
    ax_b.plot([0, lim], [0, lim], color="black", linewidth=1.0, zorder=0)
    ax_b.set_xlim(0, lim)
    ax_b.set_ylim(0, lim)
    ax_b.set_xlabel("Bright field death time")
    ax_b.set_ylabel("TOTO-3 death time")
    ax_b.set_aspect("equal", adjustable="box")
    ax_b.legend(fontsize=8, loc="upper left", frameon=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
