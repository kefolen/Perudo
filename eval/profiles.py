from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Dict


# Keys accepted by MonteCarloAgent __init__ (except 'n' which is overridden by mc_n)
_ALLOWED_MC_KEYS = {
    "name",
    "rng",
    "chunk_size",
    "early_stop_margin",
    "weighted_sampling",
    "enable_parallel",
    "num_workers",
    "enhanced_pruning",
    "variance_reduction",
    "betting_history_enabled",
    "player_trust_enabled",
    "trust_learning_rate",
    "history_memory_rounds",
    "bayesian_sampling",
}


def get_baseline_kwargs(profile_name: str) -> Dict:
    name = str(profile_name).lower()
    if name == "aggressive":
        return {"threshold_call": 0.25}
    if name == "passive":
        return {"threshold_call": 0.55}
    raise ValueError(f"Unknown baseline profile: {profile_name}")


@lru_cache(maxsize=1)
def _load_mc_config() -> Dict:
    """Load MC config JSON from web/mc_config.json. Returns empty dict on failure.
    Filters out private ('_'-prefixed) top-level keys.
    """
    try:
        # Resolve path relative to project root: eval/.. -> project root -> web/mc_config.json
        here = os.path.dirname(os.path.abspath(__file__))
        root = os.path.dirname(here)
        path = os.path.join(root, "web", "mc_config.json")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        # Drop private keys starting with '_' (like _description, _documentation)
        data = {k: v for k, v in data.items() if not str(k).startswith("_")}
        return data
    except Exception:
        # Graceful fallback: no external params
        return {}


def get_mc_kwargs(mc_n: int) -> Dict:
    if not isinstance(mc_n, int):
        raise ValueError("mc_n must be an int")
    if mc_n <= 0:
        raise ValueError("mc_n must be positive")

    cfg = _load_mc_config().copy()

    # Remove 'n' from file (we control it via mc_n)
    cfg.pop("n", None)

    # Keep only allowed keys to avoid unexpected kwargs errors
    cfg = {k: v for k, v in cfg.items() if k in _ALLOWED_MC_KEYS}

    # Finally, set the simulation budget from argument
    cfg["n"] = mc_n
    return cfg
