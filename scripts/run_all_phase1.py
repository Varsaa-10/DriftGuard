#!/usr/bin/env python3
"""Execute the full Phase 1 warm‑up experiment (576 configurations).

For each configuration we run `run_all` (which internally calls the configurable
adaptive baseline) and write a per‑event trace CSV using the `trace_path`
parameter.  The trace schema matches the columns required for the warm‑up
semantic audit.

All traces are stored under `results/phase1/traces/` with a filename that
encodes the warm‑up, window, seed and scenario.

A summary CSV `results/phase1/summary.csv` is also produced, containing the
primary outcome metrics defined in the original request.
"""
import json
import csv
from pathlib import Path
import numpy as np
import pandas as pd

# Project root (two levels up from this script)
ROOT = Path(__file__).resolve().parents[1]
sys_path_added = False

# Ensure the src package is importable
import sys
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
    sys_path_added = True
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from driftguard import data, features, baselines as B
from driftguard.baselines_configurable import run_adaptive_configurable
from driftguard.experiments import run_all

OUT_DIR = ROOT / "results" / "phase1"
TRACE_DIR = OUT_DIR / "traces"
TRACE_DIR.mkdir(parents=True, exist_ok=True)

# Load data splits
base_raw = data.load_split(str(ROOT / "data"), "base")
train_raw = data.load_split(str(ROOT / "data"), "train")
test_raw = data.load_split(str(ROOT / "data"), "test")

# Convert to DataFrames / feature rows
base_df = features.build_frame(base_raw)
train_df = features.build_frame(train_raw)
test_df = features.build_frame(test_raw)

cols = features.ALL_FEATURES
nf = len(cols)

# Pre‑compute reference rows and timestamps
base_rows = list(base_df[cols].itertuples(index=False, name=None))
base_t = base_df["t"].values

# Fixed baseline scores for the test stream (used when window too small)
fixed_model = B.run_fixed(base_rows, list(test_df[cols].itertuples(index=False, name=None)), nf, alpha=0.01)
fixed_scores = fixed_model.scores

# Load configuration generator output
config_path = ROOT / "results" / "phase1" / "configurations" / "all_576.jsonl"
configs = []
with config_path.open() as f:
    for line in f:
        configs.append(json.loads(line))

summary_rows = []

for cfg in configs:
    warm = cfg["warmup"]
    win = cfg["window"]
    seed = cfg["seed"]
    scen = cfg["scenario"]
    # Build the synthetic stream for this scenario/seed using the same helper as Phase 5
    # Scenario definitions live in v11/scenarios.py
    from v11.scenarios import build_stream
    stream_df = build_stream(scenario=scen, seed=seed)
    stream_rows = list(stream_df[cols].itertuples(index=False, name=None))
    stream_t = stream_df["t"].values
    stream_y = stream_df["label"].values

    # Specification for guarded‑adaptive with the chosen window
    spec = B.Spec(kind="adaptive", window=win, mode="guarded")

    # Run the experiment for this configuration
    result = run_adaptive_configurable(
        spec,
        base_rows,
        base_t,
        stream_rows,
        stream_t,
        stream_y,
        n_features=nf,
        alpha=0.01,
        tau=tau_initial,  # calibrated guard threshold
        fixed_scores=fixed_scores,
        min_n=100,
        guard_q=0.99,
        guard_m=2000,
        guard_warm=warm,
        recalibration_interval=25,
        trace_path=str(TRACE_DIR / f"warm{warm}_win{win}_seed{seed}_{scen}.csv"),
        rng=np.random.default_rng(seed),
    )

    # Simple outcome metrics (admission rates etc.)
    total = len(stream_y)
    attacks = int(stream_y.sum())
    benign = total - attacks
    admitted = int(result.admitted.sum())
    admitted_attacks = int((result.admitted & (stream_y == 1)).sum())
    admitted_benign = admitted - admitted_attacks
    attack_rate = admitted_attacks / attacks if attacks else 0.0
    benign_rate = admitted_benign / benign if benign else 0.0
    overall_rate = admitted / total if total else 0.0

    summary_rows.append({
        "scenario": scen,
        "seed": seed,
        "warmup": warm,
        "window": win,
        "total_events": total,
        "attack_events": attacks,
        "benign_events": benign,
        "admitted_events": admitted,
        "admitted_attack_events": admitted_attacks,
        "admitted_benign_events": admitted_benign,
        "attack_admission_rate": attack_rate,
        "benign_admission_rate": benign_rate,
        "overall_admission_rate": overall_rate,
    })

# Write summary CSV
summary_path = OUT_DIR / "summary.csv"
with summary_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
    writer.writeheader()
    writer.writerows(summary_rows)

print(f"Executed {len(configs)} configurations, traces written to {TRACE_DIR}")
print(f"Summary CSV written to {summary_path}")
