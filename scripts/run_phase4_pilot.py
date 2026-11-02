"""Phase 4 Calibration Pilot Script.

Evaluates the 4 calibration strategies (C-IN, C-HO, C-SW, C-RQ) and the C-RQ-Robust variant
on identical deterministic stream data, seeds, and alert-budget denominators.
Saves all run artifacts and metric summaries to results/v2.0/phase4/pilot/.
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

PILOT_OUT = ROOT / "results" / "v2.0" / "phase4" / "pilot"
PILOT_OUT.mkdir(parents=True, exist_ok=True)

ALPHA = 0.1
GUARD_Q = 0.95
MIN_N = 100
ALERT_BUDGETS = [0.005, 0.01, 0.02, 0.05, 0.10]  # 0.5% budget = B=50 per 10k events


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main():
    np.random.seed(0)
    
    # 1. Load Data Splits
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
    
    # Stream: We use the full chronological test stream (N=24,087)
    # The first 10% (N=2,408) is benign, followed by natural drift and real attacks (2,585 attacks)
    stream_rows = list(test_df[cols].itertuples(index=False, name=None))
    stream_t = test_df["t"].values
    stream_y = test_df["label"].values
    n_stream = len(stream_rows)
    
    # Precompute fixed scores for fallback
    m_fixed = RarityModel(nf, ALPHA).fit(base_rows)
    fixed_scores = np.array(m_fixed.score_many(stream_rows))
    
    # Hour IDs for evaluate()
    hour_ids = (stream_t // 3600).astype(int)
    
    # 2. Compute Initial Thresholds for All Strategies
    tau_cin = B.calibrate_cv(base_rows, nf, ALPHA, q=GUARD_Q, k=5)
    tau_cho = B.calibrate_ho(base_rows, stream_rows, nf, ALPHA, q=GUARD_Q, frac=0.10)
    tau_csw_init = tau_cin  # C-SW initializes with in-sample CV threshold, updates dynamically
    tau_crq_oracle = B.calibrate_rq(base_rows, train_rows, nf, ALPHA, q=GUARD_Q, mode="oracle", val_labels=train_labels)
    tau_crq_robust = B.calibrate_rq(base_rows, train_rows, nf, ALPHA, q=GUARD_Q, mode="robust")
    
    thresholds = {
        "C-IN": tau_cin,
        "C-HO": tau_cho,
        "C-SW_init": tau_csw_init,
        "C-RQ-Oracle": tau_crq_oracle,
        "C-RQ-Robust": tau_crq_robust,
    }
    
    print("[PILOT] Calibrated Thresholds:", thresholds)
    
    # 3. Execute Strategies
    spec = B.Spec("adaptive", "1h", "guarded")
    
    runs = {}
    # C-IN: static tau_cin, no sliding window frac
    runs["C-IN"] = B.run_adaptive(spec, base_rows, base_t, stream_rows, stream_t, stream_y,
                                  nf, ALPHA, tau_cin, fixed_scores, min_n=MIN_N, guard_q=None)
    
    # C-HO: static tau_cho, no sliding window frac
    runs["C-HO"] = B.run_adaptive(spec, base_rows, base_t, stream_rows, stream_t, stream_y,
                                  nf, ALPHA, tau_cho, fixed_scores, min_n=MIN_N, guard_q=None)
    
    # C-SW: window_frac=0.05 (5% of stream length), guard_q=0.95
    runs["C-SW"] = B.run_adaptive(spec, base_rows, base_t, stream_rows, stream_t, stream_y,
                                  nf, ALPHA, tau_csw_init, fixed_scores, min_n=MIN_N,
                                  guard_q=GUARD_Q, window_frac=0.05)
    
    # C-RQ-Oracle: static tau_crq_oracle
    runs["C-RQ-Oracle"] = B.run_adaptive(spec, base_rows, base_t, stream_rows, stream_t, stream_y,
                                         nf, ALPHA, tau_crq_oracle, fixed_scores, min_n=MIN_N, guard_q=None)
    
    # C-RQ-Robust: static tau_crq_robust
    runs["C-RQ-Robust"] = B.run_adaptive(spec, base_rows, base_t, stream_rows, stream_t, stream_y,
                                         nf, ALPHA, tau_crq_robust, fixed_scores, min_n=MIN_N, guard_q=None)
    
    # Static Baseline for reference
    runs["Static-Baseline"] = B.run_fixed(base_rows, stream_rows, nf, ALPHA)
    
    # 4. Evaluate Metrics Under Identical Alert Budgets
    summary_metrics = []
    total_attacks = int(stream_y.sum())
    total_benign = int((stream_y == 0).sum())
    
    for name, res in runs.items():
        assert not np.isnan(res.scores).any(), f"NaN in scores for {name}"
        metrics = evaluate(stream_y, res.scores, hour_ids=hour_ids, budgets=ALERT_BUDGETS)
        
        # Explicit Contamination and Admission Breakdown
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
            "recall@1%": float(metrics.get("recall@1%", 0.0)),
            "recall@5%": float(metrics.get("recall@5%", 0.0)),
            "admitted_attacks": admitted_attacks,
            "total_attacks": total_attacks,
            "attack_admission_rate_pct": attack_admission_rate_pct,
            "admitted_benign": admitted_benign,
            "total_admitted": total_admitted,
            "baseline_contamination_rate_pct": baseline_contamination_rate_pct,
        }
        summary_metrics.append(row)
        
        # Save per-strategy run summaries
        strategy_file = PILOT_OUT / f"{name}_summary.json"
        with open(strategy_file, "w", encoding="utf-8") as f:
            json.dump({
                "strategy": name,
                "tau_initial": row["tau_initial"],
                "metrics": row,
                "score_distribution": {
                    "min": float(np.min(res.scores)),
                    "median": float(np.median(res.scores)),
                    "q95": float(np.quantile(res.scores, 0.95)),
                    "max": float(np.max(res.scores)),
                },
                "admitted_count": total_admitted,
                "fallback_count": int(res.fallback.sum()),
            }, f, indent=2, sort_keys=True)
            
    summary_df = pd.DataFrame(summary_metrics)
    summary_csv = PILOT_OUT / "pilot_summary_metrics.csv"
    summary_df.to_csv(summary_csv, index=False)
    
    summary_json = PILOT_OUT / "pilot_summary_metrics.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2, sort_keys=True)
        
    print("\n[PILOT RESULTS SUMMARY]")
    print(summary_df.to_string(index=False))
    
    # 5. Compute Hashes of All Artifacts (excluding hash tracking files)
    hashes = {}
    for p in sorted(PILOT_OUT.glob("*")):
        if p.is_file() and p.name not in ("hashes.json", "run1_hashes.json"):
            hashes[p.name] = file_sha256(p)
            
    with open(PILOT_OUT / "hashes.json", "w", encoding="utf-8") as f:
        json.dump(hashes, f, indent=2, sort_keys=True)
        
    print("\n[PILOT COMPLETE] Artifact hashes written to hashes.json")
    print(json.dumps(hashes, indent=2))


if __name__ == "__main__":
    main()
