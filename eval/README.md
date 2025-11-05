# eval/ Quickstart

Run Monte Carlo vs Baseline evaluation experiments from the command line and (optionally) save a simple plot.

## 1) Setup (Windows PowerShell)
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -U pip
py -m pip install -r requirements.txt
```

## 2) Run experiments
- 3v3 sweep (both baseline profiles) with a tiny grid:
```powershell
py -m eval.cli sweep_mc_n --mc-grid 100,200 --seeds 1 --matches-per-seed 5 --name demo_sweep --jobs 0
```
- 1v5 one-vs-many (both profiles): single mc_n or a grid
```powershell
# Single mc_n
py -m eval.cli one_vs_many --mc-n 100 --seeds 1 --matches-per-seed 5 --name demo_1v5 --jobs 0
# Or sweep multiple mc_n values
py -m eval.cli one_vs_many --mc-grid 100,200,400 --seeds 1 --matches-per-seed 5 --name demo_1v5_sweep --jobs 0
```

Outputs are saved to: `eval/results/experiments/<name>/`
- Files: `config.json`, `matches.csv`, `players.csv`, `summary.json`
- Plot: `winrate_vs_mc.png` (if matplotlib available; skip with `--no-plot`)

Live progress: CLI shows a progress bar with elapsed time and ETA in the terminal while matches run.

Parallelization:
- Use `--jobs N` to run matches in parallel (match-level). `--jobs 0` uses CPU-1 automatically; `--jobs 1` forces sequential.
- If you enable internal MC parallelism in `web/mc_config.json` ("enable_parallel": true), consider keeping `--jobs 1` to avoid nested process pools.

Notes:
- Seating is deterministic per spec (3v3: MC at seats 0,2,4; 1v5: single MC seat rotates).
- MC agent parameters are loaded from web/mc_config.json; only `n` (mc_n) is controlled by the CLI/experiment.
- Adjust `--mc-grid`, `--seeds`, `--matches-per-seed`, `--out` as needed.

## 3) (Optional) Legacy demo
```powershell
py eval\tournament.py
```