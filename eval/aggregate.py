from __future__ import annotations

from typing import List, Dict, Tuple


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return center - half, center + half


def aggregate_win_rates(matches: List[Dict]) -> List[Dict]:
    """
    Group by (layout, baseline_profile, mc_n) and compute counts, wins, and win rate for MC team.
    Returns list of rows with keys: layout, baseline_profile, mc_n, matches, mc_team_wins, win_rate_mc_team,
    ci_low, ci_high
    """
    groups: Dict[Tuple[str, str, int], List[Dict]] = {}
    for m in matches:
        key = (m["layout"], m["baseline_profile"], m["mc_n"])
        groups.setdefault(key, []).append(m)

    rows: List[Dict] = []
    for (layout, profile, mc_n), items in groups.items():
        n = len(items)
        k = sum(1 for it in items if bool(it.get("mc_team_win")))
        wr = k / n if n > 0 else float("nan")
        lo, hi = wilson_ci(k, n) if n > 0 else (float("nan"), float("nan"))
        rows.append({
            "layout": layout,
            "baseline_profile": profile,
            "mc_n": mc_n,
            "matches": n,
            "mc_team_wins": k,
            "win_rate_mc_team": wr,
            "ci_low": lo,
            "ci_high": hi,
        })
    return rows
