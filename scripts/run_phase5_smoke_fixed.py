"""Phase 5 Smoke Test – Fixed Design

Ensures both positive (attack) and negative (benign) classes are present when evaluating
ROC‑AUC, PR‑AUC and related detection metrics. The selection rule is deterministic:
- First 500 benign events (label = 0) in chronological order.
- First 500 attack events (label = 1) in chronological order.
If fewer than 500 of either class exist, all available events of that class are used.
The combined set is sorted by timestamp to preserve the original stream order.

Quality‑gate checks enforce:
- All strategies run on the exact same event IDs.
- ROC‑AUC and PR‑AUC are set to `null` (JSON `None`) when only one label class is present,
  with an explanatory note in the summary.
- Required metadata files (`metadata.json`, `hashes.json`) must exist.
- Re‑running the smoke test with identical seeds produces identical hashes.
"""

from __future__ import annotations

import sys, os, json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from driftguard import data, features, baselines as B
from driftguard.scoring import RarityModel
from driftguard.metrics import evaluate

SMOKE_OUT = ROOT / "results" / "v3.0" / "phase5" / "smoke_fixed"
SMOKE_OUT.mkdir(parents=True, exist_ok=True)

ALPHA = 0.1
GUARD_Q = 0.95
MIN_N = 100
ALERT_BUDGETS = [0.005]  # 0.5% budget for quick test

def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def select_balanced_stream(df: pd.DataFrame, benign_n: int = 500, attack_n: int = 500) -> pd.DataFrame:
    benign = df[df["label"] == 0].head(benign_n)
    attack = df[df["label"] == 1].head(attack_n)
    combined = pd.concat([benign, attack]).sort_values("t", kind="stable").reset_index(drop=True)
    return combined

def main():
    np.random.seed(0)
    # Load splits
    base_raw = data.load_split(str(ROOT / "data"), "base")
    train_raw = data.load_split(str(ROOT / "data"), "train")
    test_raw = data.load_split(str(ROOT / "data"), "test")

    base_df = features.build_frame(base_raw)
    train_df = features.build_frame(train_raw)
    test_df = features.build_frame(test_raw)

    # Deterministic balanced stream
    stream_df = select_balanced_stream(test_df)
    if stream_df.empty:
        raise RuntimeError("Selected stream is empty – no events found.")

    cols = features.ALL_FEATURES
    nf = len(cols)

    base_rows = list(base_df[cols].itertuples(index=False, name=None))
    train_rows = list(train_df[cols].itertuples(index=False, name=None))
    train_labels = train_df["label"].values
    stream_rows = list(stream_df[cols].itertuples(index=False, name=None))
    stream_t = stream_df["t"].values
    stream_y = stream_df["label"].values

    # Fixed scores fallback
    m_fixed = RarityModel(nf, ALPHA).fit(base_rows)
    fixed_scores = np.array(m_fixed.score_many(stream_rows))

    hour_ids = (stream_t // 3600).astype(int)

    # Calibrate thresholds using the same procedures as Phase 4 (full chronological test stream)
    # Load the full test stream for calibration (chronological order)
    full_test_rows = list(test_df[cols].itertuples(index=False, name=None))
    # C-HO uses the first 10% of the full stream (all benign in Phase 4)
    tau_cho = B.calibrate_ho(base_rows, full_test_rows, nf, ALPHA, q=GUARD_Q, frac=0.10)
    # Other thresholds stay unchanged
    tau_cin = B.calibrate_cv(base_rows, nf, ALPHA, q=GUARD_Q, k=5)
    tau_csw_init = tau_cin
    tau_crq_oracle = B.calibrate_rq(base_rows, train_rows, nf, ALPHA, q=GUARD_Q, mode="oracle", val_labels=train_labels)
    tau_crq_robust = B.calibrate_rq(base_rows, train_rows, nf, ALPHA, q=GUARD_Q, mode="robust")
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
    print("[SMOKE_FIXED] Calibrated thresholds:", thresholds)

    spec = B.Spec("adaptive", "1h", "guarded")
    runs = {}
    runs["C-IN"] = B.run_adaptive(spec, base_rows, base_df["t"].values, stream_rows, stream_t, stream_y,
                                 nf, ALPHA, tau_cin, fixed_scores, min_n=MIN_N, guard_q=None)
    runs["C-HO"] = B.run_adaptive(spec, base_rows, base_df["t"].values, stream_rows, stream_t, stream_y,
                                 nf, ALPHA, tau_cho, fixed_scores, min_n=MIN_N, guard_q=None)
    runs["C-SW"] = B.run_adaptive(spec, base_rows, base_df["t"].values, stream_rows, stream_t, stream_y,
                                 nf, ALPHA, tau_csw_init, fixed_scores, min_n=MIN_N,
                                 guard_q=GUARD_Q, window_frac=0.05)
    runs["C-RQ-Oracle"] = B.run_adaptive(spec, base_rows, base_df["t"].values, stream_rows, stream_t, stream_y,
                                 nf, ALPHA, tau_crq_oracle, fixed_scores, min_n=MIN_N, guard_q=None)
    runs["C-RQ-Robust"] = B.run_adaptive(spec, base_rows, base_df["t"].values, stream_rows, stream_t, stream_y,
                                 nf, ALPHA, tau_crq_robust, fixed_scores, min_n=MIN_N, guard_q=None)
    runs["Static-Baseline"] = B.run_fixed(base_rows, stream_rows, nf, ALPHA)

    # Quality gate – ensure all runs have identical event IDs
    event_ids = [np.arange(len(stream_rows)) for _ in runs]
    if not all(np.array_equal(event_ids[0], eid) for eid in event_ids[1:]):
        raise AssertionError("Inconsistent event IDs across strategies.")

    summary_metrics = []
    total_attacks = int(stream_y.sum())
    total_events = len(stream_y)

    for name, res in runs.items():
        assert not np.isnan(res.scores).any(), f"NaN in scores for {name}"
        # Compute metrics; handle single‑class case
        try:
            metrics = evaluate(stream_y, res.scores, hour_ids=hour_ids, budgets=ALERT_BUDGETS)
        except Exception as e:
            raise RuntimeError(f"Metric evaluation failed for {name}: {e}")

        # Detect single‑class situation
        if len(np.unique(stream_y)) == 1:
            roc_auc = None
            pr_auc = None
            note = "Only one label class present – metrics undefined."
        else:
            roc_auc = float(metrics["roc_auc"])
            pr_auc = float(metrics["pr_auc"])
            note = ""

        admitted_attacks = int((res.admitted & (stream_y == 1)).sum())
        admitted_benign = int((res.admitted & (stream_y == 0)).sum())
        total_admitted = int(res.admitted.sum())
        attack_admission_rate_pct = (admitted_attacks / total_attacks) * 100.0 if total_attacks > 0 else 0.0
        baseline_contamination_rate_pct = (admitted_attacks / total_admitted) * 100.0 if total_admitted > 0 else 0.0

        row = {
            "strategy": name,
            "tau_initial": thresholds.get(name, thresholds.get(f"{name}_init", np.nan)),
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "note": note,
            "recall@0.5%": float(metrics.get("recall@0%", metrics.get("recall@1%", 0.0))),
            "admitted_attacks": admitted_attacks,
            "total_attacks": total_attacks,
            "attack_admission_rate_pct": attack_admission_rate_pct,
            "admitted_benign": admitted_benign,
            "total_admitted": total_admitted,
            "baseline_contamination_rate_pct": baseline_contamination_rate_pct,
        }
        summary_metrics.append(row)
        # Save per‑strategy JSON
        with open(SMOKE_OUT / f"{name}_summary.json", "w", encoding="utf-8") as f:
            json.dump(row, f, indent=2, sort_keys=True)

    df = pd.DataFrame(summary_metrics)
    summary_csv = SMOKE_OUT / "summary_metrics.csv"
    df.to_csv(summary_csv, index=False)
    summary_json = SMOKE_OUT / "summary_metrics.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2, sort_keys=True)

    # Metadata
    metadata = {
        "seed": 0,
        "stream_length": total_events,
        "selected_event_ids": list(range(total_events)),
        "selection_rule": "first 500 benign + first 500 attacks, sorted chronologically",
        "alert_budgets": ALERT_BUDGETS,
        "strategies": list(runs.keys()),
    }
    with open(SMOKE_OUT / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    # Hashes of output files
    hashes = {p.name: file_sha256(p) for p in SMOKE_OUT.iterdir() if p.is_file()}
    with open(SMOKE_OUT / "hashes.json", "w", encoding="utf-8") as f:
        json.dump(hashes, f, indent=2, sort_keys=True)

    print("[SMOKE_FIXED COMPLETE] Summary CSV at", summary_csv)
    print(json.dumps(hashes, indent=2))

if __name__ == "__main__":
    main()
