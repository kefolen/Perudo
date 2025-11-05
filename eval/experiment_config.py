from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Literal


VALID_LAYOUTS = {"3v3", "1v5"}
VALID_PROFILES = {"aggressive", "passive"}


@dataclass
class ExperimentCondition:
    mc_n: int
    baseline_profile: str
    layout: Literal["3v3", "1v5"]
    player_count: int = 6

    def validate(self) -> None:
        # mc_n must be positive integer
        if not isinstance(self.mc_n, int) or self.mc_n <= 0:
            raise ValueError("mc_n must be a positive integer")
        if self.layout not in VALID_LAYOUTS:
            raise ValueError(f"Invalid layout: {self.layout}")
        if self.baseline_profile not in VALID_PROFILES:
            raise ValueError(f"Invalid baseline_profile: {self.baseline_profile}")
        if self.player_count != 6:
            # MVP only supports 6 players as per spec/tests
            raise ValueError("player_count must be 6 for MVP")


@dataclass
class ExperimentConfig:
    name: str
    seeds: List[int]
    num_matches_per_seed: int
    conditions: List[ExperimentCondition]
    # Legacy flags kept for backward-compat with tests; not used in MVP
    log_decisions: bool = False
    shuffle_seats: bool = False
    out_dir: Optional[str] = None

    def validate(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string")
        if not isinstance(self.seeds, list) or not all(isinstance(s, int) for s in self.seeds):
            raise ValueError("seeds must be a list[int]")
        if len(self.seeds) == 0:
            raise ValueError("seeds list must be non-empty")
        if not isinstance(self.num_matches_per_seed, int) or self.num_matches_per_seed <= 0:
            raise ValueError("num_matches_per_seed must be a positive integer")
        if not isinstance(self.conditions, list) or len(self.conditions) == 0:
            raise ValueError("conditions must be a non-empty list")
        for c in self.conditions:
            if not isinstance(c, ExperimentCondition):
                raise ValueError("conditions must contain ExperimentCondition items")
            c.validate()

    def to_json(self) -> str:
        # Convert dataclasses to serializable dicts
        data = asdict(self)
        # dataclasses.asdict already recurses into nested dataclasses
        return json.dumps(data)

    @staticmethod
    def from_json(s: str) -> ExperimentConfig:
        obj = json.loads(s)
        # Extract conditions
        conds = [ExperimentCondition(**c) for c in obj.get("conditions", [])]
        cfg = ExperimentConfig(
            name=obj.get("name"),
            seeds=list(obj.get("seeds", [])),
            num_matches_per_seed=obj.get("num_matches_per_seed"),
            conditions=conds,
            log_decisions=obj.get("log_decisions", False),
            shuffle_seats=obj.get("shuffle_seats", False),
            out_dir=obj.get("out_dir"),
        )
        return cfg
