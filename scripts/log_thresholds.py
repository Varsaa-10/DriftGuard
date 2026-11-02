# """Log tau (guard threshold) trajectory for adaptive and fixed guarded runs.
#
# This script runs the adaptive guarded baseline on the full test stream and records:
#   - event index
#   - timestamp
#   - score
#   - tau (current guard threshold)
#   - admitted flag
# It produces two CSV files:
#   - results/v5.6/robust_validation/threshold_log_adaptive.csv
#   - results/v5.6/robust_validation/threshold_log_fixed.csv
#
# Fixed run is simulated by disabling the adaptive guard recalibration (guard_q=None) so
# tau stays at its initial calibrated value for the whole stream.
#"""

from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from driftguard import data, features, baselines as B
from driftguard.scoring import RarityModel

OUT_DIR = ROOT / "results" / "v5.6" / "robust_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHA = 0.1
GUARD_Q = 0.95
MIN_N = 100
WINDOW = "1h"
MODE = "guarded"

def file_sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def run_and_log(guard_q: float | None, log_path: Path):
    # Load data
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
    stream_rows = list(test_df[cols].itertuples(index=False, name=None))
    stream_t = test_df["t"].values
    stream_y = test_df["label"].values

    # Calibration (use C-IN as example)
    tau_init = B.calibrate_cv(base_rows, nf, ALPHA, q=GUARD_Q, k=5)

    # Fixed fallback scores
    m_fixed = RarityModel(nf, ALPHA).fit(base_rows)
    fixed_scores = np.array(m_fixed.score_many(stream_rows))

    spec = B.Spec("adaptive", WINDOW, MODE)

    # Run adaptive with logging of tau_now per event
    # We'll re‑implement a small wrapper around run_adaptive to capture tau.
    model = RarityModel(nf, ALPHA)
    buf = []  # not used for logging
    # Initialise reference data
    for r, t in zip(base_rows, base_t):
        model.add(r)
        buf.append((t, r))

    win = B.WINDOWS[WINDOW]
    ref_t_aligned = B.align_reference(base_t, stream_t[0])
    # Align reference buffer
    ref_buf = []
    for r, t in zip(base_rows, ref_t_aligned):
        ref_buf.append((t, r))
        model.add(r)
    # Note: run_adaptive internally does similar alignment; we mirror only for tau logging.

    recent = []
    recent_maxlen = 2000  # guard_m default for C‑SW; will be used when guard_q is set
    # In our experiment we use the same guard_m/window logic as run_adaptive's guard_warm=500, guard_m=2000.
    guard_m = 2000
    recent = []
    recent = []
    recent_deque = []  # will use list as simple buffer
    recent_len = 0
    # Use deque from collections for proper maxlen
    from collections import deque
    recent_deque = deque(maxlen=guard_m)
    tau_now = tau_init

    logs = []
    for i, (r, t) in enumerate(zip(stream_rows, stream_t)):
        # Evict old reference if window specified
        if win is not None:
            while ref_buf and ref_buf[0][0] < t - win:
                _, old = ref_buf.pop(0)
                model.remove(old)
        # Record window size
        # Compute score
        if model.n < MIN_N:
            s = fixed_scores[i]
        else:
            s = model.score(r)
        # Guard recalibration
        if guard_q is not None and len(recent_deque) >= 500 and i % 25 == 0:
            tau_now = float(np.quantile(np.fromiter(recent_deque, float), guard_q))
        adm = s < tau_now if MODE == "guarded" else True
        if adm:
            model.add(r)
            recent_deque.append(s)
        logs.append({
            "event_index": i,
            "timestamp": int(t),
            "score": float(s),
            "tau": float(tau_now),
            "admitted": bool(adm),
            "label": int(stream_y[i]),
        })
    df = pd.DataFrame(logs)
    df.to_csv(log_path, index=False)

if __name__ == "__main__":
    adaptive_path = OUT_DIR / "threshold_log_adaptive.csv"
    fixed_path = OUT_DIR / "threshold_log_fixed.csv"
    run_and_log(GUARD_Q, adaptive_path)
    run_and_log(None, fixed_path)
    # Record hashes for provenance
    hashes = {
        adaptive_path.name: file_sha256(adaptive_path),
        fixed_path.name: file_sha256(fixed_path),
    }
    with open(OUT_DIR / "threshold_logs_hashes.json", "w", encoding="utf-8") as f:
        json.dump(hashes, f, indent=2, sort_keys=True)
    print("[LOG] Threshold logs written to", OUT_DIR)
