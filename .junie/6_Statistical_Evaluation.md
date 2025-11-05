# Statistical Evaluation (Minimal) v1

Date: 2025-11-04 18:54

## Goal
Measure how MonteCarloAgent (MC) win rate changes with simulation budget (mc_n) against two BaselineAgent profiles, under two layouts, with reproducible, fast experiments suitable for TDD.

## Scope (MVP)
- Agents: MonteCarloAgent and BaselineAgent only; keep existing select_action(obs) interfaces.
- Layouts: 3v3 and 1v5 with 6 players.
- Profiles: aggressive and passive with fully specified parameters.
- Outputs: config.json, matches.csv, players.csv, summary.json only.
- KPIs: win rate (with Wilson 95% CI), mean match duration (ms), seat bias (1v5 only).

## Defaults and runtime budget
- Default mc_n grid: [100, 200, 400, 800, 1600].
- Default seeds: seeds = [1, 2, 3].
- Default num_matches_per_seed = 30 per condition.
- CI smoke target: a single mc_n=100 condition with seeds=[1], num_matches_per_seed=5 finishes in < 2 minutes on a typical laptop.

## Deterministic seating
- 3v3: Seats [0,2,4] are MC; [1,3,5] are Baseline. No shuffling in MVP.
- 1v5: Exactly one MC per match; the MC seat cycles 0→1→2→3→4→5→… across matches per condition.

## Baseline profiles
- aggressive: {"threshold_call": 0.25}
- passive: {"threshold_call": 0.55}
- If BaselineAgent exposes additional knobs, leave them at project defaults; do not vary in MVP.

## Monte Carlo agent parameters
- Only mc_n varies in MVP. Other MonteCarloAgent kwargs remain defaults.

## Data outputs
- Directory: eval/results/experiments/{timestamp_or_name}/
  - config.json: exact experiment configuration with resolved defaults.
  - matches.csv: one row per match.
  - players.csv: one row per player per match.
  - summary.json: computed KPIs and metadata.

Schemas (strict, minimal):
- matches.csv
  match_id,condition_id,seed,layout,mc_n,baseline_profile,player_count,turns,winner_team,mc_team_win,total_time_ms
- players.csv
  match_id,seat,agent_type,team,won,finish_pos,avg_decision_time_ms

Notes:
- team is MC or BASE.
- avg_decision_time_ms can be NaN if not measured; timing is optional, but total_time_ms in matches.csv is required.

## KPIs and formulas
- Win rate per condition: p̂ = wins_mc_team / total_matches.
- 95% Wilson CI for binomial proportion:
  
  def wilson_ci(k, n, z=1.959963984540054):
      if n == 0: return (float("nan"), float("nan"))
      p = k / n
      denom = 1 + z*z/n
      center = (p + z*z/(2*n)) / denom
      half = z * ((p*(1-p)/n + z*z/(4*n*n)) ** 0.5) / denom
      return (center - half, center + half)
  
- Mean match duration (ms) per condition: arithmetic mean of total_time_ms.
- Seat bias (1v5): MC win rate aggregated by seat (0–5) and reported alongside overall.
- Diminishing returns (reported in summary.json): discrete derivative over log2(mc_n) and the smallest mc_n where gain < 0.01 absolute.

## Module plan (eval/)
- experiment_config.py
  - ExperimentCondition: mc_n: int, baseline_profile: str, layout: Literal["3v3","1v5"], player_count: int = 6.
  - ExperimentConfig: name: str, seeds: List[int], num_matches_per_seed: int, conditions: List[ExperimentCondition], out_dir: Optional[str].
  - JSON (de)serialization and validation.
- profiles.py
  - get_baseline_kwargs(profile_name: str) -> dict.
  - get_mc_kwargs(mc_n: int) -> dict.
- runner.py
  - run_match(condition, seed, match_idx) -> MatchResult.
  - run_experiment(config) -> (matches_df, players_df).
  - Responsibilities: build lineup from layout; seat assignment per rules; seed RNGs; time the match; write rows.
- aggregate.py
  - compute_kpis(matches_df, players_df) -> dict returning per-condition win rate + CI, mean duration, seat bias (for 1v5), diminishing returns recommendation.
  - Writes summary.json.
- cli.py
  - Commands:
    - sweep_mc_n (3v3, both profiles) with default grid.
    - one_vs_many (1v5, both profiles) for a single mc_n.
  - Flags: --seeds, --matches-per-seed, --out.

## TDD test plan (minimal, fast)
- tests/test_experiment_config.py
  - JSON round‑trip, validation of invalid layout/profile.
- tests/test_profiles.py
  - Exact numeric mapping for aggressive and passive.
- tests/test_runner_smoke.py
  - With seeds=[1], num_matches_per_seed=1: exactly 1 row in matches.csv and 6 rows in players.csv for each condition.
  - Correct seat assignment for 3v3 and 1v5 (MC seats [0,2,4] vs rotating seat respectively).
- tests/test_aggregate.py
  - Toy matches_df producing known p̂ and Wilson CI; known mean duration; verify diminishing_returns.mc_n_recommended.
- tests/perf/test_runner_budget.py
  - mc_n=100, seeds=[1], num_matches_per_seed=1 completes under a small time budget; mark as @slow to skip on constrained CI.

## Out of scope for MVP (documented extensions)
- Per-decision logs (decisions.jsonl).
- Efficiency metric (win rate per 1k simulations) and plots requiring per-decision simulation counts.
- Elo ratings.
- Plot generation; can be added later as a thin layer over summary.json.

## Deliverables
- eval/: experiment_config.py, profiles.py, runner.py, aggregate.py, cli.py.
- Tests as above.
- Two example configs in README and CLI commands to reproduce them.
