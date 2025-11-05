from __future__ import annotations

import os
from typing import List, Dict


def save_winrate_vs_mc(aggregates: List[Dict], out_dir: str, title: str | None = None) -> bool:
    """
    Save a simple line plot of MC win rate vs mc_n for each (layout, baseline_profile) group.
    Returns True if a file was saved, or False if matplotlib is unavailable.
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        # Matplotlib not available in environment; signal gracefully
        return False

    # Group by (layout, baseline_profile)
    groups: Dict[tuple, list] = {}
    for row in aggregates:
        key = (row.get("layout"), row.get("baseline_profile"))
        groups.setdefault(key, []).append(row)

    fig, ax = plt.subplots(figsize=(6, 4))
    for (layout, profile), rows in groups.items():
        rows_sorted = sorted(rows, key=lambda r: r.get("mc_n", 0))
        xs = [r.get("mc_n", 0) for r in rows_sorted]
        ys = [r.get("win_rate_mc_team", 0.0) for r in rows_sorted]
        label = f"{layout}-{profile}"
        ax.plot(xs, ys, marker="o", label=label)

    ax.set_xlabel("mc_n")
    ax.set_ylabel("MC team win rate")
    if title:
        ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "winrate_vs_mc.png")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True
