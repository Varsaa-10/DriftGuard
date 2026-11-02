#!/usr/bin/env python3
"""Generate all 576 experiment configurations for Phase 1.

Each configuration is a JSON object with the following fields:
- warmup: int (0, 50, 100, 250, 500, 1000)
- window: str ("1h", "6h", "24h", "all")
- seed: int (0, 1, 2)
- scenario: str ("S1" … "S8")

The script writes one JSON line per configuration to
`results/phase1/configurations/all_576.jsonl`.
"""
import json
from pathlib import Path

WARMUPS = [0, 50, 100, 250, 500, 1000]
WINDOWS = ["1h", "6h", "24h", "all"]
SEEDS = [0, 1, 2]
SCENARIOS = [f"S{i}" for i in range(1, 9)]

out_path = Path(__file__).resolve().parents[1] / "results" / "phase1" / "configurations" / "all_576.jsonl"
out_path.parent.mkdir(parents=True, exist_ok=True)

with out_path.open("w", encoding="utf-8") as f:
    for warm in WARMUPS:
        for win in WINDOWS:
            for seed in SEEDS:
                for scen in SCENARIOS:
                    cfg = {
                        "warmup": warm,
                        "window": win,
                        "seed": seed,
                        "scenario": scen,
                    }
                    f.write(json.dumps(cfg) + "\n")

print(f"Generated {len(WARMUPS)*len(WINDOWS)*len(SEEDS)*len(SCENARIOS)} configurations at {out_path}")
