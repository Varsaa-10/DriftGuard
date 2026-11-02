"""Phase 5 Full Factorial Experiment Matrix Runner.

Executes the approved Phase 5 factorial matrix defined in EXPERIMENTAL_PROTOCOL.md:
- 4 Attack Fractions: 0%, 1%, 5%, 10% (0.00, 0.01, 0.05, 0.10)
- 3 Window Sizes: 100, 500, 2000 events
- 3 Guard Sensitivities (quantiles): 90th, 95th, 99th (0.90, 0.95, 0.99)
- 5 Random Seeds: 0, 1, 2, 3, 4
Yielding 4 * 3 * 3 * 5 = 180 runs per calibration strategy.

Evaluated Strategies:
- Static-Baseline (SB)
- C-IN (In-Sample CV)
- C-HO (Chronological Held-Out)
- C-SW (Sliding-Window dynamic)
- C-RQ-Oracle (Validation Benign)
- C-RQ-Robust (Validation Unsupervised)

Outputs stored in results/v5.0/matrix/.
"""
from __future__ import annotations

import os
import sys
import json
import time
import hashlib
from pathlib import Path
from collections import deque
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from driftguard import data, features, baselines as B
from driftguard.scoring import RarityModel
from driftguard.metrics import evaluate

OUT_DIR = ROOT / "results" / "v5.0" / "matrix"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHA = 0.1
MIN_N = 100
ALERT_BUDGETS = [0.005, 0.01, 0.02, 0.05, 0.10]

ATTACK_FRACTIONS = [0.00, 0.01, 0.05, 0.10]
WINDOWS = [100, 500, 2000]
GUARD_QUANTILES = [0.90, 0.95, 0.99]
SEEDS = [0, 1, 2, 3, 4]
STRATEGIES = ["Static-Baseline", "C-IN", "C-HO", "C-SW", "C-RQ-Oracle", "C-RQ-Robust"]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def sample_stream(benign_df: pd.DataFrame, attack_df: pd.DataFrame, n_total: int, attack_fraction: float, seed: int) -> pd.DataFrame:
    """Sample stream preserving natural temporal ordering with replacement per protocol."""
    rng = np.random.default_rng(seed)
    n_attacks = int(round(attack_fraction * n_total))
    n_benign = n_total - n_attacks

    benign_idx = rng.choice(len(benign_df), size=n_benign, replace=True)
    sampled_benign = benign_df.iloc[benign_idx].copy()

    if n_attacks > 0 and len(attack_df) > 0:
        attack_idx = rng.choice(len(attack_df), size=n_attacks, replace=True)
        sampled_attacks = attack_df.iloc[attack_idx].copy()
        combined = pd.concat([sampled_benign, sampled_attacks], ignore_index=True)
    else:
        combined = sampled_benign

    # Sort strictly chronologically by timestamp t with stable tie-break
    combined = combined.sort_values("t", kind="stable").reset_index(drop=True)
    return combined


def run_adaptive_count_window(ref_rows, ref_t, stream_rows, stream_t, stream_y, nf, alpha, tau_init, fixed_scores,
                              window_count, guard_q, is_csw=False):
    """Run adaptive baseline with discrete count-based window (100, 500, 2000 events) per protocol."""
    model = RarityModel(nf, alpha)
    buf = deque()
    for r in ref_rows:
        model.add(r)
        buf.append(r)

    n = len(stream_rows)
    guard_m = max(1, window_count)
    guard_warm = min(500, max(10, guard_m // 2))
    recent = deque(maxlen=guard_m)
    tau_now = tau_init
    scores = np.empty(n)
    admitted = np.zeros(n, bool)

    for i, r in enumerate(stream_rows):
        # Evict oldest if buffer exceeds window_count
        while len(buf) > window_count:
            old = buf.popleft()
            model.remove(old)

        if model.n < MIN_N:
            s = fixed_scores[i]
        else:
            s = model.score(r)
        scores[i] = s

        if is_csw:
            if guard_q is not None and len(recent) >= guard_warm and i % 25 == 0:
                tau_now = float(np.quantile(np.fromiter(recent, float), guard_q))

        adm = (s < tau_now)
        if adm:
            model.add(r)
            buf.append(r)
            admitted[i] = True
            recent.append(s)

    return scores, admitted


def main():
    print("=== DriftGuard Phase 5 Full Factorial Matrix Execution ===")
    t0 = time.time()

    # Load Splits
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

    benign_test_df = test_df[test_df["label"] == 0].reset_index(drop=True)
    attack_test_df = test_df[test_df["label"] == 1].reset_index(drop=True)

    # Precompute thresholds for all quantiles
    print("[1/4] Precomputing calibration thresholds...")
    full_test_rows = list(test_df[cols].itertuples(index=False, name=None))

    threshold_table = {}
    for q in GUARD_QUANTILES:
        cin = B.calibrate_cv(base_rows, nf, ALPHA, q=q, k=5)
        cho = B.calibrate_ho(base_rows, full_test_rows, nf, ALPHA, q=q, frac=0.10)
        crq_oracle = B.calibrate_rq(base_rows, train_rows, nf, ALPHA, q=q, mode="oracle", val_labels=train_labels)
        crq_robust = B.calibrate_rq(base_rows, train_rows, nf, ALPHA, q=q, mode="robust")
        threshold_table[q] = {
            "C-IN": cin,
            "C-HO": cho,
            "C-SW": cin,  # initializes with CV
            "C-RQ-Oracle": crq_oracle,
            "C-RQ-Robust": crq_robust,
            "Static-Baseline": np.nan,
        }
        print(f"  q={q:.2f} -> C-IN={cin:.4f}, C-HO={cho:.4f}, C-RQ-Oracle={crq_oracle:.4f}, C-RQ-Robust={crq_robust:.4f}")

    # Build Fixed Model
    m_fixed_base = RarityModel(nf, ALPHA).fit(base_rows)

    # Factorial Matrix: 4 attack fractions x 3 windows x 3 quantiles x 5 seeds = 180 runs per strategy
    # Across 6 strategies = 1080 total runs
    total_runs = len(ATTACK_FRACTIONS) * len(WINDOWS) * len(GUARD_QUANTILES) * len(SEEDS) * len(STRATEGIES)
    print(f"[2/4] Starting factorial matrix execution ({total_runs} total strategy executions)...")

    stream_cache = {}
    results = []
    run_idx = 0

    N_STREAM_EVENTS = 10000  # 10,000 events per stream condition for matrix runs

    for af in ATTACK_FRACTIONS:
        for seed in SEEDS:
            # Generate deterministic stream for (af, seed)
            stream_key = (af, seed)
            if stream_key not in stream_cache:
                st_df = sample_stream(benign_test_df, attack_test_df, N_STREAM_EVENTS, af, seed)
                st_rows = list(st_df[cols].itertuples(index=False, name=None))
                st_t = st_df["t"].values
                st_y = st_df["label"].values
                st_fixed_scores = np.array(m_fixed_base.score_many(st_rows))
                stream_cache[stream_key] = (st_df, st_rows, st_t, st_y, st_fixed_scores)
            else:
                st_df, st_rows, st_t, st_y, st_fixed_scores = stream_cache[stream_key]

            hour_ids = (st_t // 3600).astype(int)
            total_attacks = int(st_y.sum())
            total_benign = int((st_y == 0).sum())

            for win in WINDOWS:
                for q in GUARD_QUANTILES:
                    for strat in STRATEGIES:
                        run_idx += 1
                        run_id = f"{strat}_af{int(af*100):02d}_win{win}_q{int(q*100):02d}_seed{seed}"

                        # Execute strategy
                        if strat == "Static-Baseline":
                            scores = st_fixed_scores
                            admitted = np.zeros(len(st_y), bool)
                            tau_val = np.nan
                        else:
                            tau_val = threshold_table[q][strat]
                            is_csw = (strat == "C-SW")
                            scores, admitted = run_adaptive_count_window(
                                base_rows, base_t, st_rows, st_t, st_y, nf, ALPHA, tau_val, st_fixed_scores,
                                window_count=win, guard_q=q, is_csw=is_csw
                            )

                        # Evaluate metrics
                        metrics = evaluate(st_y, scores, hour_ids=hour_ids, budgets=ALERT_BUDGETS)

                        admitted_attacks = int((admitted & (st_y == 1)).sum())
                        admitted_benign = int((admitted & (st_y == 0)).sum())
                        total_admitted = int(admitted.sum())

                        att_admit_rate = (admitted_attacks / total_attacks * 100.0) if total_attacks > 0 else 0.0
                        contamination_rate = (admitted_attacks / total_admitted * 100.0) if total_admitted > 0 else 0.0

                        record = {
                            "run_id": run_id,
                            "strategy": strat,
                            "attack_fraction": af,
                            "window_size": win,
                            "guard_quantile": q,
                            "seed": seed,
                            "tau_initial": tau_val,
                            "n_events": len(st_y),
                            "n_attacks": total_attacks,
                            "roc_auc": metrics["roc_auc"],
                            "pr_auc": metrics["pr_auc"],
                            "r_precision": metrics["r_precision"],
                            "recall@0.5%": metrics.get("recall@0%"),
                            "precision@0.5%": metrics.get("precision@0%"),
                            "recall@1%": metrics.get("recall@1%"),
                            "precision@1%": metrics.get("precision@1%"),
                            "recall@2%": metrics.get("recall@2%"),
                            "precision@2%": metrics.get("precision@2%"),
                            "recall@5%": metrics.get("recall@5%"),
                            "precision@5%": metrics.get("precision@5%"),
                            "recall@10%": metrics.get("recall@10%"),
                            "precision@10%": metrics.get("precision@10%"),
                            "admitted_attacks": admitted_attacks,
                            "admitted_benign": admitted_benign,
                            "total_admitted": total_admitted,
                            "attack_admission_rate_pct": att_admit_rate,
                            "baseline_contamination_rate_pct": contamination_rate,
                        }
                        results.append(record)

                        if run_idx % 180 == 0:
                            print(f"  Processed {run_idx}/{total_runs} runs ({time.time()-t0:.1f}s elapsed)...")

    # Save Results
    print("[3/4] Saving matrix results and computing aggregates...")
    df_results = pd.DataFrame(results)
    results_csv = OUT_DIR / "phase5_factorial_results.csv"
    df_results.to_csv(results_csv, index=False)

    # Compute Grouped Summary: Strategy x Attack Fraction x Window x Quantile
    group_cols = ["strategy", "attack_fraction", "window_size", "guard_quantile"]
    metric_cols = ["roc_auc", "pr_auc", "recall@0.5%", "precision@0.5%", "recall@1%", "precision@1%", "recall@5%",
                   "attack_admission_rate_pct", "baseline_contamination_rate_pct"]

    summary_agg = df_results.groupby(group_cols)[metric_cols].agg(["mean", "std", "count"]).reset_index()
    summary_csv = OUT_DIR / "phase5_aggregate_summary.csv"
    summary_agg.to_csv(summary_csv, index=False)

    # Generate Run Manifest
    manifest = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_runs_executed": len(df_results),
        "strategies": STRATEGIES,
        "attack_fractions": ATTACK_FRACTIONS,
        "windows": WINDOWS,
        "guard_quantiles": GUARD_QUANTILES,
        "seeds": SEEDS,
        "input_hashes": {
            "base.json": file_sha256(ROOT / "data" / "base.json"),
            "train.json": file_sha256(ROOT / "data" / "train.json"),
            "test.json": file_sha256(ROOT / "data" / "test.json"),
        },
        "output_files": {
            "results_csv": file_sha256(results_csv),
            "summary_csv": file_sha256(summary_csv),
        }
    }
    with open(OUT_DIR / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[4/4] Execution completed successfully in {time.time()-t0:.1f}s.")
    print(f"Results CSV: {results_csv}")
    print(f"Aggregate Summary: {summary_csv}")
    print(f"Manifest: {OUT_DIR / 'run_manifest.json'}")


if __name__ == "__main__":
    main()
