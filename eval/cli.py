from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from typing import List, Optional

from .experiment_config import ExperimentConfig, ExperimentCondition
from .runner import run_experiment
from .aggregate import aggregate_win_rates
from .plots import save_winrate_vs_mc
from .profiles import get_baseline_kwargs  # noqa: F401  # imported to ensure profile names exist


class _SimpleProgress:
    def __init__(self, total: int, desc: str = "") -> None:
        import time
        self.total = max(1, int(total))
        self.done = 0
        self.t0 = time.time()
        self.desc = desc

    def update(self, n: int = 1) -> None:
        import sys, time
        self.done += int(n)
        now = time.time()
        elapsed = now - self.t0
        rate = self.done / elapsed if elapsed > 0 else 0.0
        remaining = (self.total - self.done) / rate if rate > 0 else float("inf")
        def _fmt(s: float) -> str:
            if s == float("inf"):
                return "--:--"
            m, s = divmod(int(s), 60)
            h, m = divmod(m, 60)
            return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
        pct = 100.0 * self.done / self.total
        msg = f"{self.desc} {self.done}/{self.total} ({pct:5.1f}%) | elapsed {_fmt(elapsed)} | eta {_fmt(remaining)}"
        print("\r" + msg, end="", file=sys.stderr, flush=True)
        if self.done >= self.total:
            self.close()

    def close(self) -> None:
        import sys
        print(file=sys.stderr)


def _make_progress(total: int, desc: str = "Running matches"):
    try:
        from tqdm import tqdm  # type: ignore
        bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        return tqdm(total=total, desc=desc, unit="match", dynamic_ncols=True, bar_format=bar_format)
    except Exception:
        return _SimpleProgress(total, desc)


DEFAULT_MC_GRID = [200, 400, 800, 1600, 3200]
DEFAULT_SEEDS = [1, 2, 3]
DEFAULT_MATCHES_PER_SEED = 30


MATCHES_SCHEMA = [
    "match_id",
    "condition_id",
    "seed",
    "layout",
    "mc_n",
    "baseline_profile",
    "player_count",
    "turns",
    "winner_team",
    "mc_team_win",
    "total_time_ms",
]

PLAYERS_SCHEMA = [
    "match_id",
    "seat",
    "agent_type",
    "team",
    "won",
    "finish_pos",
    "avg_decision_time_ms",
]


def _ensure_out_dir(out: str | None, name: str) -> str:
    if out:
        base = out
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = os.path.join("eval", "results", "experiments", name or ts)
    os.makedirs(base, exist_ok=True)
    return base


def _write_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _write_csv(path: str, rows: List[dict], schema: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=schema)
        writer.writeheader()
        for r in rows:
            # Only write known fields as per schema
            writer.writerow({k: r.get(k) for k in schema})


def _parse_int_list(s: str) -> List[int]:
    try:
        return [int(x) for x in s.split(",") if x.strip()]
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid int list: {s}") from e


def cmd_sweep_mc_n(args: argparse.Namespace) -> int:
    seeds = _parse_int_list(args.seeds) if isinstance(args.seeds, str) else DEFAULT_SEEDS
    matches_per_seed = int(args.matches_per_seed)
    grid = _parse_int_list(args.mc_grid) if isinstance(args.mc_grid, str) else DEFAULT_MC_GRID
    baseline_profiles = ["aggressive", "passive"]

    # Resolve jobs -> max_workers (0 means auto)
    jobs = int(args.jobs)
    if jobs <= 0:
        cpu = os.cpu_count() or 1
        jobs = max(1, cpu - 1)
    max_workers = jobs if jobs > 1 else None

    conditions = [
        ExperimentCondition(mc_n=n, baseline_profile=prof, layout="3v3", player_count=6)
        for n in grid for prof in baseline_profiles
    ]

    cfg = ExperimentConfig(
        name=args.name or "sweep_mc_n",
        seeds=seeds,
        num_matches_per_seed=matches_per_seed,
        conditions=conditions,
        out_dir=None,
    )

    out_dir = _ensure_out_dir(args.out, cfg.name)
    _write_json(os.path.join(out_dir, "config.json"), json.loads(cfg.to_json()))

    # Progress bar
    total_matches = len(conditions) * len(seeds) * matches_per_seed
    progress = _make_progress(total_matches, desc="Running matches")
    try:
        matches, players = run_experiment(cfg, out_dir=None, progress=progress, max_workers=max_workers)
    finally:
        try:
            progress.close()  # type: ignore[attr-defined]
        except Exception:
            pass

    # Persist CSVs per strict schema
    _write_csv(os.path.join(out_dir, "matches.csv"), matches, MATCHES_SCHEMA)
    _write_csv(os.path.join(out_dir, "players.csv"), players, PLAYERS_SCHEMA)

    # Aggregates and summary
    aggr = aggregate_win_rates(matches)
    summary = {
        "name": cfg.name,
        "layout": "3v3",
        "baseline_profiles": baseline_profiles,
        "mc_grid": grid,
        "seeds": seeds,
        "num_matches_per_seed": matches_per_seed,
        "aggregates": aggr,
    }
    _write_json(os.path.join(out_dir, "summary.json"), summary)

    # Plot
    if not args.no_plot:
        try:
            saved = save_winrate_vs_mc(aggr, out_dir, title=f"Win rate vs mc_n — {cfg.name}")
            if not saved:
                print("matplotlib not available; skipping plot", file=sys.stderr)
        except Exception as e:
            print(f"Plotting failed: {e}", file=sys.stderr)

    print(f"Done. Outputs saved to: {out_dir}")
    return 0


def cmd_one_vs_many(args: argparse.Namespace) -> int:
    seeds = _parse_int_list(args.seeds) if isinstance(args.seeds, str) else DEFAULT_SEEDS
    matches_per_seed = int(args.matches_per_seed)
    mc_n = int(args.mc_n)
    baseline_profiles = ["aggressive", "passive"]

    # Resolve jobs -> max_workers (0 means auto)
    jobs = int(args.jobs)
    if jobs <= 0:
        cpu = os.cpu_count() or 1
        jobs = max(1, cpu - 1)
    max_workers = jobs if jobs > 1 else None

    conditions = [
        ExperimentCondition(mc_n=mc_n, baseline_profile=prof, layout="1v5", player_count=6)
        for prof in baseline_profiles
    ]

    cfg = ExperimentConfig(
        name=args.name or f"one_vs_many_mc{mc_n}",
        seeds=seeds,
        num_matches_per_seed=matches_per_seed,
        conditions=conditions,
        out_dir=None,
    )

    out_dir = _ensure_out_dir(args.out, cfg.name)
    _write_json(os.path.join(out_dir, "config.json"), json.loads(cfg.to_json()))

    # Progress bar
    total_matches = len(conditions) * len(seeds) * matches_per_seed
    progress = _make_progress(total_matches, desc="Running matches")
    try:
        matches, players = run_experiment(cfg, out_dir=None, progress=progress, max_workers=max_workers)
    finally:
        try:
            progress.close()  # type: ignore[attr-defined]
        except Exception:
            pass

    _write_csv(os.path.join(out_dir, "matches.csv"), matches, MATCHES_SCHEMA)
    _write_csv(os.path.join(out_dir, "players.csv"), players, PLAYERS_SCHEMA)

    aggr = aggregate_win_rates(matches)
    summary = {
        "name": cfg.name,
        "layout": "1v5",
        "baseline_profiles": baseline_profiles,
        "mc_n": mc_n,
        "seeds": seeds,
        "num_matches_per_seed": matches_per_seed,
        "aggregates": aggr,
    }
    _write_json(os.path.join(out_dir, "summary.json"), summary)

    if not args.no_plot:
        try:
            saved = save_winrate_vs_mc(aggr, out_dir, title=f"Win rate vs mc_n — {cfg.name}")
            if not saved:
                print("matplotlib not available; skipping plot", file=sys.stderr)
        except Exception as e:
            print(f"Plotting failed: {e}", file=sys.stderr)

    print(f"Done. Outputs saved to: {out_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="perudo-eval", description="Perudo statistical evaluation CLI (MVP)")
    sub = p.add_subparsers(dest="cmd", required=True)

    # sweep_mc_n
    p_sweep = sub.add_parser("sweep_mc_n", help="Run 3v3 sweep over mc_n for both baseline profiles")
    p_sweep.add_argument("--mc-grid", default=",".join(str(x) for x in DEFAULT_MC_GRID), help="Comma-separated mc_n values")
    p_sweep.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS), help="Comma-separated seeds")
    p_sweep.add_argument("--matches-per-seed", type=int, default=DEFAULT_MATCHES_PER_SEED)
    p_sweep.add_argument("--out", default=None, help="Output directory (defaults to eval/results/experiments/<name>)")
    p_sweep.add_argument("--name", default="sweep_mc_n", help="Experiment name")
    p_sweep.add_argument("--no-plot", action="store_true", help="Do not generate plot")
    p_sweep.add_argument("--jobs", default="0", help="Parallel jobs for match-level parallelism (0=auto, 1=sequential)")
    p_sweep.set_defaults(func=cmd_sweep_mc_n)

    # one_vs_many
    p_one = sub.add_parser("one_vs_many", help="Run 1v5 for a single mc_n against both profiles")
    p_one.add_argument("--mc-n", type=int, default=100, help="Monte Carlo simulations per decision")
    p_one.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS), help="Comma-separated seeds")
    p_one.add_argument("--matches-per-seed", type=int, default=DEFAULT_MATCHES_PER_SEED)
    p_one.add_argument("--out", default=None, help="Output directory (defaults to eval/results/experiments/<name>)")
    p_one.add_argument("--name", default=None, help="Experiment name (auto if omitted)")
    p_one.add_argument("--no-plot", action="store_true", help="Do not generate plot")
    p_one.add_argument("--jobs", default="0", help="Parallel jobs for match-level parallelism (0=auto, 1=sequential)")
    p_one.set_defaults(func=cmd_one_vs_many)

    return p


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
