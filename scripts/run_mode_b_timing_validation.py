"""Leakage-free Mode-B validation with unknown, band-limited inputs.

The primary protocol is a physical cold start: ``g_B=1``, ``o_B=0``, and
``Delta t_BA=0``.  Fixed ADC full-scale constants normalize the NLMS features,
so no record-level means, variances, or least-squares state are used. Every
safe sample is corrected with the pre-update coefficient state and is then
used once for the update.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from calibration import (  # noqa: E402
    apply_fir_differentiator,
    build_fir_differentiator,
    build_group_streams,
    fractional_delay_predict_B_from_A,
    run_background_nlms_stream,
)
from model import (  # noqa: E402
    MismatchConfig,
    coarse_to_final_lsb,
    default_2p5b_redundant_model,
    flash_2p5b_quantize,
    inject_fine_path_mismatch,
    quantize_and_clip_fine_code,
)


FS = 1.0e9
N = 2**16
N_FINE = 8
FINE_SCALE = float(2 ** (N_FINE - 1))
SLOPE_SCALE = FINE_SCALE
EVALUATION_FULL_RATE_SAMPLES = N // 4
SEEDS = (20260711, 20260712, 20260713, 20260714, 20260715)
MISMATCH = MismatchConfig(
    g0=1.0,
    o0=0.0,
    g1=1.02,
    o1=2.0,
    g2=1.0,
    o2=0.0,
    g3=1.02,
    o3=2.0,
    noise_std=0.08,
)
DT_MDAC_B = 20e-12
PREDICTOR_TAPS = 15
DIFFERENTIATOR_TAPS = 31
PREDICTOR_DELAY_A = (PREDICTOR_TAPS - 1) // 2
DIFFERENTIATOR_DELAY_A = (DIFFERENTIATOR_TAPS - 1) // 2
TOTAL_DELAY_A = PREDICTOR_DELAY_A + DIFFERENTIATOR_DELAY_A
NLMS_MU = 0.005
TRACE_BLOCK_SIZE = 256


def _time_shifted_signal(
    spectrum: np.ndarray,
    n_samples: int,
    dt_seconds: float,
) -> np.ndarray:
    frequency = np.fft.rfftfreq(n_samples, d=1.0 / FS)
    return np.fft.irfft(
        spectrum * np.exp(1j * 2.0 * np.pi * frequency * dt_seconds),
        n=n_samples,
    )


def generate_signals(
    generator_type: str,
    seed: int,
    *,
    n_samples: int = N,
    dt_seconds: float = DT_MDAC_B,
) -> tuple[np.ndarray, np.ndarray]:
    """Return nominal-grid and group-B-skewed full-rate input records."""

    rng = np.random.default_rng(seed)
    n = np.arange(n_samples, dtype=np.float64)

    if generator_type == "multitone":
        # Preserve approximately the same physical frequencies for any N.
        frequencies = np.asarray([23.0e6, 71.0e6, 137.5e6])
        amplitudes = np.asarray([1.0, 0.72, 0.48])
        phases = rng.uniform(0.0, 2.0 * np.pi, frequencies.size)
        t_nominal = n / FS
        x_nominal = sum(
            amplitude * np.sin(2.0 * np.pi * frequency * t_nominal + phase)
            for amplitude, frequency, phase in zip(amplitudes, frequencies, phases)
        )
        x_group_b = sum(
            amplitude
            * np.sin(
                2.0 * np.pi * frequency * (t_nominal + dt_seconds) + phase
            )
            for amplitude, frequency, phase in zip(amplitudes, frequencies, phases)
        )
    elif generator_type == "bandlimited_random":
        spectrum = np.zeros(n_samples // 2 + 1, dtype=np.complex128)
        max_bin = int(0.15 * n_samples)
        spectrum[1 : max_bin + 1] = (
            rng.normal(size=max_bin) + 1j * rng.normal(size=max_bin)
        )
        edge_start = int(0.12 * n_samples)
        edge = np.arange(edge_start, max_bin + 1)
        spectrum[edge] *= 0.5 * (
            1.0
            + np.cos(
                np.pi * (edge - edge_start) / max(1, max_bin - edge_start)
            )
        )
        x_nominal = np.fft.irfft(spectrum, n=n_samples)
        x_group_b = _time_shifted_signal(spectrum, n_samples, dt_seconds)
    elif generator_type == "qam16_dmt":
        active = np.rint(
            np.linspace(0.0205 * n_samples, 0.1797 * n_samples, 164)
        ).astype(int)
        spectrum = np.zeros(n_samples // 2 + 1, dtype=np.complex128)
        levels = np.asarray([-3.0, -1.0, 1.0, 3.0])
        spectrum[active] = rng.choice(levels, active.size) + 1j * rng.choice(
            levels, active.size
        )
        x_nominal = np.fft.irfft(spectrum, n=n_samples)
        x_group_b = _time_shifted_signal(spectrum, n_samples, dt_seconds)
    else:
        raise ValueError(f"Unsupported signal generator {generator_type!r}.")

    mean_nominal = float(np.mean(x_nominal))
    centered_nominal = x_nominal - mean_nominal
    scale = 500.0 / float(np.max(np.abs(centered_nominal)))
    x_nominal = centered_nominal * scale
    x_group_b = (x_group_b - mean_nominal) * scale
    x_mdac = x_nominal.copy()
    x_mdac[1::2] = x_group_b[1::2]
    return x_nominal, x_mdac


def build_record(
    x_nominal: np.ndarray,
    x_mdac: np.ndarray,
    seed: int,
    mismatch: MismatchConfig = MISMATCH,
) -> dict[str, np.ndarray]:
    model = default_2p5b_redundant_model(N_FINE)
    n = np.arange(x_nominal.size, dtype=np.int64)
    sar_id = n % 4
    mdac_id = n % 2
    coarse_code = flash_2p5b_quantize(x_nominal, model)
    coarse = coarse_to_final_lsb(coarse_code, model)
    fine_ideal = x_mdac - coarse
    slope = np.gradient(x_nominal)
    slope_b = slope[1::2]
    slope_norm = (slope - float(np.mean(slope_b))) / (
        float(np.std(slope_b)) or 1.0
    )
    fine_float = inject_fine_path_mismatch(
        fine_ideal, sar_id, mdac_id, slope_norm, mismatch
    )
    fine_raw, _, clipped = quantize_and_clip_fine_code(
        fine_float,
        rng=np.random.default_rng(seed + 1000),
        N_fine=N_FINE,
        noise_std=mismatch.noise_std,
    )
    if np.any(clipped):
        raise RuntimeError(
            f"Fine-code clipping occurred in {int(np.sum(clipped))} samples"
        )
    return {
        "D_ideal": x_nominal,
        "D_coarse": coarse,
        "F_raw": fine_raw,
        "C_raw": coarse_code,
        "sar_id": sar_id,
    }


def build_mode_b_features(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    # The unknown-input experiment isolates inter-group Mode B; paired SAR
    # channels within each MDAC group have identical injected mismatch.
    group = build_group_streams(data["D_coarse"], data["F_raw"].astype(float))
    d_hat, mask_ref, _ = fractional_delay_predict_B_from_A(
        group["U_A"], num_taps=PREDICTOR_TAPS
    )
    h_diff = build_fir_differentiator(DIFFERENTIATOR_TAPS)
    slope_raw = apply_fir_differentiator(d_hat, h_diff) / 2.0
    f_hat = d_hat - group["D_coarse_B"]
    fine_limit = 0.8 * FINE_SCALE
    safe = (
        mask_ref
        & np.isfinite(slope_raw)
        & np.isfinite(f_hat)
        & (np.abs(group["F_B"]) < fine_limit)
        & (np.abs(f_hat) < fine_limit)
    )
    return {
        **group,
        "Dhat_B": d_hat,
        "Fhat_B": f_hat,
        "slope_raw": slope_raw,
        "slope_B": slope_raw / SLOPE_SCALE,
        "safe": safe,
    }




def _settling_updates(block_trace: list[dict[str, Any]]) -> int | None:
    """Return a robust 15%-of-final block-RMS settling estimate."""

    if len(block_trace) < 8:
        return None
    rms = np.asarray([row["block_rms_res_B"] for row in block_trace])
    final_floor = float(np.median(rms[-8:]))
    threshold = 1.15 * final_floor
    rolling = np.convolve(rms, np.ones(3) / 3.0, mode="valid")
    for i in range(rolling.size):
        if np.all(rolling[i:] <= threshold):
            return int(block_trace[i + 2]["update_count"])
    return None


def evaluate_protocol(
    signal_name: str,
    data: dict[str, np.ndarray],
    seed: int,
) -> dict[str, Any]:
    features = build_mode_b_features(data)
    update_mask = features["safe"].copy()
    initial = {"g_B": 1.0, "o_B": 0.0, "k_B": 0.0}
    fine_online, coeffs, diagnostics = run_background_nlms_stream(
        features["F_B"],
        features["Fhat_B"],
        features["slope_B"],
        update_mask,
        mu=NLMS_MU,
        fine_scale=FINE_SCALE,
        initial_g_B=initial["g_B"],
        initial_o_B=initial["o_B"],
        initial_k_B=initial["k_B"],
        trace_block_size=TRACE_BLOCK_SIZE,
    )
    corrected = np.empty(data["D_ideal"].size, dtype=np.float64)
    corrected[0::2] = features["U_A"]
    corrected[1::2] = features["D_coarse_B"] + fine_online
    raw = data["D_coarse"] + data["F_raw"]

    valid = np.arange(data["D_ideal"].size) >= (
        data["D_ideal"].size - EVALUATION_FULL_RATE_SAMPLES
    )
    valid[1::2] &= features["safe"]
    signal_rms = float(np.std(data["D_ideal"][valid]))

    def rms_error(values: np.ndarray) -> float:
        error = values[valid] - data["D_ideal"][valid]
        return float(np.sqrt(np.mean(error * error)))

    rms_raw = rms_error(raw)
    rms_full = rms_error(corrected)
    estimated_dt_ps = -coeffs["k_B"] / (SLOPE_SCALE * FS) * 1e12
    return {
        "signal": signal_name,
        "seed": seed,
        "protocol": "cold_start",
        "samples": int(data["D_ideal"].size),
        "data_fitted_initialization": False,
        "unavailable_full_rate_samples": 0,
        "joint_ls_initialization_safe_B_samples": 0,
        "nlms_updates": int(diagnostics["update_count"]),
        "settling_updates_1p15x": _settling_updates(diagnostics["block_trace"]),
        "evaluation_full_rate_samples": int(np.sum(valid)),
        "rms_raw_lsb": rms_raw,
        "rms_after_full_lsb": rms_full,
        "evm_raw_percent": 100.0 * rms_raw / signal_rms,
        "evm_after_full_percent": 100.0 * rms_full / signal_rms,
        "g_B": float(coeffs["g_B"]),
        "o_B_lsb": float(coeffs["o_B"]),
        "k_B_lsb": float(coeffs["k_B"]),
        "estimated_dt_ps": float(estimated_dt_ps),
        "trace": diagnostics["block_trace"],
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for signal in ("Multitone", "Band-limited random", "16-QAM DMT"):
        for protocol in ("cold_start",):
            selected = [
                row
                for row in rows
                if row["signal"] == signal and row["protocol"] == protocol
            ]
            if not selected:
                continue
            group_summary: dict[str, float] = {}
            for key in (
                "nlms_updates",
                "settling_updates_1p15x",
                "rms_raw_lsb",
                "rms_after_full_lsb",
                "evm_raw_percent",
                "evm_after_full_percent",
                "g_B",
                "o_B_lsb",
                "k_B_lsb",
                "estimated_dt_ps",
            ):
                values = np.asarray(
                    [float(row[key]) for row in selected if row[key] is not None]
                )
                group_summary[f"{key}_mean"] = float(np.mean(values))
                group_summary[f"{key}_std"] = (
                    float(np.std(values, ddof=1)) if values.size > 1 else 0.0
                )
            summary[f"{signal}|{protocol}"] = group_summary
    return summary


def main() -> None:
    rows: list[dict[str, Any]] = []
    generators = (("Multitone", "multitone"), ("Band-limited random", "bandlimited_random"), ("16-QAM DMT", "qam16_dmt"))
    for signal_name, generator_type in generators:
        for seed in SEEDS:
            x_nominal, x_mdac = generate_signals(generator_type, seed)
            data = build_record(x_nominal, x_mdac, seed)
            rows.append(evaluate_protocol(signal_name, data, seed))
    print(json.dumps({"summary": _summarize(rows), "records": [{key: value for key, value in row.items() if key != "trace"} for row in rows]}, indent=2))


if __name__ == "__main__":
    main()
