"""Phase 5 Smoke Test Script

Runs a minimal deterministic pilot to validate the experiment runner before the full factorial execution.
It executes a single short run for each calibration strategy on the first 1,000 events of the test stream.
Outputs metadata and summary metrics to `results/v3.0/phase5/smoke/`.
"""

from __future__ import annotations

import os
import sys
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from driftguard import data, features, baselines as B
from driftguard.scoring import RarityModel
from driftguard.metrics import evaluate

SMOKE_OUT = ROOT / "results" / "v3.0" / "phase5" / "smoke"
SMOKE_OUT.mkdir(parents=True, exist_ok=True)

ALPHA = 0.1
GUARD_Q = 0.95
MIN_N = 100
ALERT_BUDGETS = [0.005]  # only 0.5% budget for quick test

def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def main():
    np.random.seed(0)
    # Load data splits
    base_raw = data.load_split(str(ROOT / "data"), "base")
    train_raw = data.load_split(str(ROOT / "data"), "train")
    test_raw = data.load_split(str(ROOT / "data"), "test")

    base_df = features.build_frame(base_raw)
    train_df = features.build_frame(train_raw)
    test_df = features.build_frame(test_raw)

    cols = features.ALL_FEATURES
    nf = len(cols)

    base_rows = list(base_df[cols].itertuples(index=False, name=None))
    base_t = base_df["t"].values
    train_rows = list(train_df[cols].itertuples(index=False, name=None))
    train_labels = train_df["label"].values
    test_rows_full = list(test_df[cols].itertuples(index=False, name=None))
    test_t_full = test_df["t"].values
    test_y_full = test_df["label"].values

    # Shortened stream: first 1000 events
    stream_rows = test_rows_full[:1000]
    stream_t = test_t_full[:1000]
    stream_y = test_y_full[:1000]
    n_stream = len(stream_rows)

    # Precompute fixed scores for fallback
    m_fixed = RarityModel(nf, ALPHA).fit(base_rows)
    fixed_scores = np.array(m_fixed.score_many(stream_rows))

    hour_ids = (stream_t // 3600).astype(int)

    # Calibrate thresholds
    tau_cin = B.calibrate_cv(base_rows, nf, ALPHA, q=GUARD_Q, k=5)
    tau_cho = B.calibrate_ho(base_rows, stream_rows, nf, ALPHA, q=GUARD_Q, frac=0.10)
    tau_csw_init = tau_cin
    tau_crq_oracle = B.calibrate_rq(base_rows, train_rows, nf, ALPHA, q=GUARD_Q, mode="oracle", val_labels=train_labels)
    tau_crq_robust = B.calibrate_rq(base_rows, train_rows, nf, ALPHA, q=GUARD_Q, mode="robust")

    thresholds = {
        "C-IN": tau_cin,
        "C-HO": tau_cho,
        "C-SW_init": tau_csw_init,
        "C-RQ-Oracle": tau_crq_oracle,
        "C-RQ-Robust": tau_crq_robust,
    }
    print("[SMOKE] Calibrated thresholds:", thresholds)

    spec = B.Spec("adaptive", "1h", "guarded")
    runs = {}
    runs["C-IN"] = B.run_adaptive(spec, base_rows, base_t, stream_rows, stream_t, stream_y,
                                 nf, ALPHA, tau_cin, fixed_scores, min_n=MIN_N, guard_q=None)
    runs["C-HO"] = B.run_adaptive(spec, base_rows, base_t, stream_rows, stream_t, stream_y,
                                 nf, ALPHA, tau_cho, fixed_scores, min_n=MIN_N, guard_q=None)
    runs["C-SW"] = B.run_adaptive(spec, base_rows, base_t, stream_rows, stream_t, stream_y,
                                 nf, ALPHA, tau_csw_init, fixed_scores, min_n=MIN_N,
                                 guard_q=GUARD_Q, window_frac=0.05)
    runs["C-RQ-Oracle"] = B.run_adaptive(spec, base_rows, base_t, stream_rows, stream_t, stream_y,
                                 nf, ALPHA, tau_crq_oracle, fixed_scores, min_n=MIN_N, guard_q=None)
    runs["C-RQ-Robust"] = B.run_adaptive(spec, base_rows, base_t, stream_rows, stream_t, stream_y,
                                 nf, ALPHA, tau_crq_robust, fixed_scores, min_n=MIN_N, guard_q=None)
    runs["Static-Baseline"] = B.run_fixed(base_rows, stream_rows, nf, ALPHA)

    summary_metrics = []
    total_attacks = int(stream_y.sum())
    total_benign = int((stream_y == 0).sum())

    for name, res in runs.items():
        assert not np.isnan(res.scores).any(), f"NaN in scores for {name}"
        metrics = evaluate(stream_y, res.scores, hour_ids=hour_ids, budgets=ALERT_BUDGETS)
        admitted_attacks = int((res.admitted & (stream_y == 1)).sum())
        admitted_benign = int((res.admitted & (stream_y == 0)).sum())
        total_admitted = int(res.admitted.sum())
        attack_admission_rate_pct = (admitted_attacks / total_attacks) * 100.0 if total_attacks > 0 else 0.0
        baseline_contamination_rate_pct = (admitted_attacks / total_admitted) * 100.0 if total_admitted > 0 else 0.0
        row = {
            "strategy": name,
            "tau_initial": thresholds.get(name, thresholds.get(f"{name}_init", np.nan)),
            "roc_auc": float(metrics["roc_auc"]),
            "pr_auc": float(metrics["pr_auc"]),
            "recall@0.5%": float(metrics.get("recall@0%", metrics.get("recall@1%", 0.0))),
            "admitted_attacks": admitted_attacks,
            "total_attacks": total_attacks,
            "attack_admission_rate_pct": attack_admission_rate_pct,
            "admitted_benign": admitted_benign,
            "total_admitted": total_admitted,
            "baseline_contamination_rate_pct": baseline_contamination_rate_pct,
        }
        summary_metrics.append(row)
        # Save per-strategy summary
        strategy_file = SMOKE_OUT / f"{name}_summary.json"
        with open(strategy_file, "w", encoding="utf-8") as f:
            json.dump({"strategy": name, "tau_initial": row["tau_initial"], "metrics": row}, f, indent=2, sort_keys=True)

    df = pd.DataFrame(summary_metrics)
    summary_csv = SMOKE_OUT / "summary_metrics.csv"
    df.to_csv(summary_csv, index=False)
    summary_json = SMOKE_OUT / "summary_metrics.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2, sort_keys=True)

    # Record metadata
    metadata = {
        "seed": 0,
        "stream_length": n_stream,
        "alert_budgets": ALERT_BUDGETS,
        "strategies": list(runs.keys()),
    }
    with open(SMOKE_OUT / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    # Hashes of smoke output files
    hashes = {p.name: file_sha256(p) for p in SMOKE_OUT.iterdir() if p.is_file()}
    with open(SMOKE_OUT / "hashes.json", "w", encoding="utf-8") as f:
        json.dump(hashes, f, indent=2, sort_keys=True)

    print("[SMOKE COMPLETE] Summary written to", summary_csv)

if __name__ == "__main__":
    main()
