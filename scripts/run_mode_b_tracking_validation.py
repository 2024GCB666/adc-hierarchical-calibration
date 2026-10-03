"""Cold-start and reacquisition test for time-varying Mode-B mismatch."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from calibration import run_background_nlms_stream  # noqa: E402
from model import MismatchConfig  # noqa: E402
from scripts.run_mode_b_timing_validation import (  # noqa: E402
    FINE_SCALE,
    FS,
    SLOPE_SCALE,
    build_mode_b_features,
    build_record,
    generate_signals,
)


N_TRACK = 2**17
SWITCH_FULL_RATE = N_TRACK // 2
SEED = 20260716
TRACKING_MU = 0.01
DT_BEFORE = 10e-12
DT_AFTER = -15e-12
MISMATCH_BEFORE = MismatchConfig(
    g0=1.0,
    o0=0.0,
    g1=1.01,
    o1=1.0,
    g2=1.0,
    o2=0.0,
    g3=1.01,
    o3=1.0,
    noise_std=0.08,
)
MISMATCH_AFTER = MismatchConfig(
    g0=1.0,
    o0=0.0,
    g1=1.03,
    o1=-1.5,
    g2=1.0,
    o2=0.0,
    g3=1.03,
    o3=-1.5,
    noise_std=0.08,
)


def build_step_record() -> dict[str, np.ndarray]:
    x_nominal, x_mdac_before = generate_signals(
        "qam16_dmt", SEED, n_samples=N_TRACK, dt_seconds=DT_BEFORE
    )
    x_nominal_after, x_mdac_after = generate_signals(
        "qam16_dmt", SEED, n_samples=N_TRACK, dt_seconds=DT_AFTER
    )
    if not np.array_equal(x_nominal, x_nominal_after):
        raise RuntimeError("Nominal records differ across timing cases.")
    before = build_record(x_nominal, x_mdac_before, SEED, MISMATCH_BEFORE)
    after = build_record(x_nominal, x_mdac_after, SEED, MISMATCH_AFTER)
    record = {key: value.copy() for key, value in before.items()}
    record["F_raw"][SWITCH_FULL_RATE:] = after["F_raw"][SWITCH_FULL_RATE:]
    return record


def _settling_after_step(
    trace: list[dict[str, Any]], step_updates: int
) -> int | None:
    for index, row in enumerate(trace):
        if int(row["update_count"]) <= step_updates:
            continue
        window = trace[index : index + 5]
        if len(window) < 5:
            return None
        within = [
            abs(float(point["g_B"]) - MISMATCH_AFTER.g1) < 0.001
            and abs(float(point["o_B"]) - MISMATCH_AFTER.o1) < 0.10
            and abs(
                -float(point["k_B"]) / (SLOPE_SCALE * FS) * 1e12
                - DT_AFTER * 1e12
            )
            < 2.0
            for point in window
        ]
        if all(within):
            return int(row["update_count"]) - step_updates
    return None


def _window_evm(
    corrected: np.ndarray,
    ideal: np.ndarray,
    safe_b: np.ndarray,
    start: int,
    stop: int,
) -> float:
    valid = (np.arange(ideal.size) >= start) & (np.arange(ideal.size) < stop)
    valid[1::2] &= safe_b
    error = corrected[valid] - ideal[valid]
    return 100.0 * float(np.sqrt(np.mean(error * error))) / float(
        np.std(ideal[valid])
    )


def run_tracking() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data = build_step_record()
    features = build_mode_b_features(data)
    fine_online, coeffs, diagnostics = run_background_nlms_stream(
        features["F_B"],
        features["Fhat_B"],
        features["slope_B"],
        features["safe"],
        mu=TRACKING_MU,
        fine_scale=FINE_SCALE,
        trace_block_size=128,
    )
    corrected = np.empty(N_TRACK, dtype=np.float64)
    corrected[0::2] = features["U_A"]
    corrected[1::2] = features["D_coarse_B"] + fine_online
    switch_b = SWITCH_FULL_RATE // 2
    step_updates = int(np.sum(features["safe"][:switch_b]))
    trace = diagnostics["block_trace"]
    for row in trace:
        row["estimated_dt_ps"] = (
            -float(row["k_B"]) / (SLOPE_SCALE * FS) * 1e12
        )
    reacquisition = _settling_after_step(trace, step_updates)
    quarter = N_TRACK // 4
    metrics = {
        "signal": "16-QAM DMT",
        "samples": N_TRACK,
        "switch_full_rate_sample": SWITCH_FULL_RATE,
        "safe_updates_before_switch": step_updates,
        "total_safe_updates": int(diagnostics["update_count"]),
        "nlms_mu": TRACKING_MU,
        "reacquisition_safe_updates": reacquisition,
        "evm_before_step_settled_percent": _window_evm(
            corrected,
            data["D_ideal"],
            features["safe"],
            SWITCH_FULL_RATE - quarter // 2,
            SWITCH_FULL_RATE,
        ),
        "evm_after_step_settled_percent": _window_evm(
            corrected,
            data["D_ideal"],
            features["safe"],
            N_TRACK - quarter // 2,
            N_TRACK,
        ),
        "final_g_B": float(coeffs["g_B"]),
        "final_o_B_lsb": float(coeffs["o_B"]),
        "final_estimated_dt_ps": (
            -float(coeffs["k_B"]) / (SLOPE_SCALE * FS) * 1e12
        ),
        "before": {
            "g_B": MISMATCH_BEFORE.g1,
            "o_B_lsb": MISMATCH_BEFORE.o1,
            "dt_ps": DT_BEFORE * 1e12,
        },
        "after": {
            "g_B": MISMATCH_AFTER.g1,
            "o_B_lsb": MISMATCH_AFTER.o1,
            "dt_ps": DT_AFTER * 1e12,
        },
    }
    return metrics, trace


def main() -> None:
    metrics, _ = run_tracking()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
