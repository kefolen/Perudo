from __future__ import annotations

import time
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

from sim.perudo import PerudoSimulator
from agents.baseline_agent import BaselineAgent
from agents.mc_agent import MonteCarloAgent

from .experiment_config import ExperimentConfig, ExperimentCondition
from .profiles import get_baseline_kwargs, get_mc_kwargs


def _build_seating(condition: ExperimentCondition, match_idx: int) -> List[str]:
    """
    Returns list of team tags per seat: "MC" or "BASE".
    Deterministic per spec:
    - 3v3: MC at seats [0,2,4]
    - 1v5: one MC whose seat rotates by match_idx modulo player_count
    """
    p = condition.player_count
    seats = ["BASE"] * p
    if condition.layout == "3v3":
        for i in [0, 2, 4]:
            if i < p:
                seats[i] = "MC"
    else:  # 1v5
        mc_seat = match_idx % p
        seats[mc_seat] = "MC"
    return seats


def run_match(condition: ExperimentCondition, seed: int, match_idx: int) -> Tuple[Dict, List[Dict]]:
    """
    Run a single match and return (match_row, players_rows).
    """
    condition.validate()

    sim = PerudoSimulator(num_players=condition.player_count, seed=seed + match_idx)

    # Seating and agent instantiation
    team_by_seat = _build_seating(condition, match_idx)
    mc_kwargs = get_mc_kwargs(condition.mc_n)
    base_kwargs = get_baseline_kwargs(condition.baseline_profile)

    agents = []
    for seat, team in enumerate(team_by_seat):
        if team == "MC":
            a = MonteCarloAgent(**mc_kwargs)
        else:
            a = BaselineAgent(**base_kwargs)
        agents.append(a)

    t0 = time.time()
    winner_player, _state = sim.play_game(agents)
    t1 = time.time()

    winner_team = team_by_seat[winner_player]
    mc_team_win = (winner_team == "MC")

    match_row: Dict = {
        "match_id": 0,  # will be filled by run_experiment with global counter
        "condition_id": f"{condition.layout}-{condition.baseline_profile}-{condition.mc_n}",
        "seed": seed,
        "layout": condition.layout,
        "mc_n": condition.mc_n,
        "baseline_profile": condition.baseline_profile,
        "player_count": condition.player_count,
        "turns": None,
        "winner_player": winner_player,
        "winner_team": winner_team,
        "mc_team_win": mc_team_win,
        "total_time_ms": int((t1 - t0) * 1000),
    }

    players_rows: List[Dict] = []
    for seat, team in enumerate(team_by_seat):
        players_rows.append({
            "match_id": 0,  # filled by run_experiment
            "seat": seat,
            "agent_type": team,
            "team": team,
            "won": seat == winner_player,
            "finish_pos": 1 if seat == winner_player else None,
            "avg_decision_time_ms": None,
        })

    return match_row, players_rows


def run_experiment(cfg: ExperimentConfig, out_dir: Optional[str] = None, progress: Optional[object] = None, max_workers: Optional[int] = None) -> Tuple[List[Dict], List[Dict]]:
    """
    Minimal runner used by tests. Returns (matches, players) as in-memory lists of dicts.
    If `progress` is provided, it can be either:
      - an object with an `update(int)` method (e.g., tqdm instance), or
      - a callable taking (done:int, total:int) to report progress.
    Parallelization:
      - If max_workers is None or <= 1, runs sequentially (default, test-friendly).
      - If max_workers > 1, dispatches matches to a ProcessPoolExecutor.
    """
    # Validation optional in tests, but safe to do
    # Do not require out_dir; tests pass None
    matches: List[Dict] = []
    players: List[Dict] = []

    # Build flat task list for deterministic ordering and progress
    tasks: List[Tuple[ExperimentCondition, int, int]] = []
    for cond in cfg.conditions:
        for seed in cfg.seeds:
            for m in range(cfg.num_matches_per_seed):
                tasks.append((cond, seed, m))

    total_matches = len(tasks)

    def _progress_update(done: int) -> None:
        if progress is None:
            return
        # tqdm-like object with update method
        upd = getattr(progress, "update", None)
        if callable(upd):
            try:
                upd(1)
                return
            except Exception:
                pass
        # Fallback: callable(done, total)
        if callable(progress):
            try:
                progress(done, total_matches)
            except Exception:
                pass

    # Sequential path (default)
    if not max_workers or max_workers <= 1:
        match_global_id = 0
        done = 0
        for cond, seed, m in tasks:
            match_row, players_rows = run_match(cond, seed, m)
            match_global_id += 1
            match_row["match_id"] = match_global_id
            for pr in players_rows:
                pr["match_id"] = match_global_id
            matches.append(match_row)
            players.extend(players_rows)
            done += 1
            _progress_update(done)
        return matches, players

    # Parallel path
    results: List[Optional[Tuple[Dict, List[Dict]]]] = [None] * total_matches
    done = 0
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        future_to_idx = {
            ex.submit(run_match, cond, seed, m): idx
            for idx, (cond, seed, m) in enumerate(tasks)
        }
        for fut in as_completed(future_to_idx):
            idx = future_to_idx[fut]
            match_row, players_rows = fut.result()
            results[idx] = (match_row, players_rows)
            done += 1
            _progress_update(done)

    # Assign deterministic match_id based on original task order
    match_global_id = 0
    for item in results:
        assert item is not None
        match_row, players_rows = item
        match_global_id += 1
        match_row["match_id"] = match_global_id
        for pr in players_rows:
            pr["match_id"] = match_global_id
        matches.append(match_row)
        players.extend(players_rows)

    # Tests currently do not require writing CSVs or JSON when out_dir is None.
    # If needed in future, implement file I/O following the spec.
    return matches, players
