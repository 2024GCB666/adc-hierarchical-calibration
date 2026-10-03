"""Numerical power-matching bias and aligned target-construction ablation."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from calibration import CalibrationConfig, apply_intra_group_calibration, build_B_reference, build_group_streams, build_safe_mask, calibrate_adc
from fft_output_code import compute_fft_metrics
from model import DatasetConfig, MismatchConfig, SignalConfig, TimingConfig, build_synthetic_adc_dataset

def ideal_fine_path_mismatch() -> MismatchConfig:
    """Remove fine-path gain/offset/noise so any inferred gain is false."""

    return MismatchConfig(
        g0=1.0,
        o0=0.0,
        g1=1.0,
        o1=0.0,
        g2=1.0,
        o2=0.0,
        g3=1.0,
        o3=0.0,
        noise_std=0.0,
    )

def _fft(values: np.ndarray, fs: float, fin: float) -> dict[str, float]:
    metric, _ = compute_fft_metrics(values, fs=fs, fin=fin, window="auto")
    return metric

def simulate_power_matching_ablation(
    skew_ps: float,
    *,
    n_samples: int = 2**16,
    seed: int = 20260620,
) -> dict[str, float]:
    """Compare direct power matching with group-predictive joint LS."""

    fs = 1.0e9
    tone_bin = 32703
    fin = tone_bin / n_samples * fs
    cfg = DatasetConfig(
        N=n_samples,
        seed=seed,
        signal=SignalConfig(
            fs=fs,
            fin=fin,
            # Slight back-off keeps the entire 0--40 ps sweep free of fine-code
            # clipping while remaining near full scale.
            amplitude=505.0,
            phase=0.31,
            dc=0.0,
        ),
        timing=TimingConfig(dt_mdac_A=0.0, dt_mdac_B=skew_ps * 1.0e-12),
        mismatch=ideal_fine_path_mismatch(),
    )
    data, truth, _ = build_synthetic_adc_dataset(cfg)

    D_coarse = data["D_coarse"].astype(np.float64)
    F_raw = data["F_raw"].astype(np.float64)
    F_intra, _, _ = apply_intra_group_calibration(F_raw, data["sar_id"])
    fine_a = F_intra[0::2]
    fine_b = F_intra[1::2]

    var_a = float(np.var(fine_a, ddof=1))
    var_b = float(np.var(fine_b, ddof=1))
    g_power = float(np.sqrt(var_a / var_b))

    F_power = F_intra.copy()
    F_power[1::2] = g_power * fine_b
    D_power = D_coarse + F_power
    D_raw = D_coarse + F_raw

    calibration_cfg = CalibrationConfig(
        reference_method="known_tone",
        known_tone_source="A",
        inter_method="joint",
    )
    D_proposed, coeffs, _, _ = calibrate_adc(
        D_coarse,
        F_raw,
        sar_id=data["sar_id"],
        C_raw=data["C_raw"],
        fs=fs,
        fin=fin,
        D_ideal=data["D_ideal"],
        truth=truth,
        config=calibration_cfg,
    )

    raw_metric = _fft(D_raw, fs, fin)
    power_metric = _fft(D_power, fs, fin)
    proposed_metric = _fft(D_proposed, fs, fin)
    g_proposed = float(coeffs["g_B"])

    return {
        "mdac_a_skew_ps": 0.0,
        "mdac_b_skew_ps": float(skew_ps),
        "sample_count": int(n_samples),
        "fs_mhz": fs / 1.0e6,
        "fin_mhz": fin / 1.0e6,
        "true_fine_gain": 1.0,
        "var_F_A_lsb2": var_a,
        "var_F_B_lsb2": var_b,
        "var_ratio_B_over_A": var_b / var_a,
        "g_power": g_power,
        "g_power_error_percent": 100.0 * (g_power - 1.0),
        "g_proposed": g_proposed,
        "g_proposed_error_percent": 100.0 * (g_proposed - 1.0),
        "raw_sndr_db": float(raw_metric["SNDR_dB"]),
        "power_sndr_db": float(power_metric["SNDR_dB"]),
        "proposed_sndr_db": float(proposed_metric["SNDR_dB"]),
        "raw_sfdr_dbc": float(raw_metric["SFDR_dBc_single_bin"]),
        "power_sfdr_dbc": float(power_metric["SFDR_dBc_single_bin"]),
        "proposed_sfdr_dbc": float(proposed_metric["SFDR_dBc_single_bin"]),
        "sndr_improvement_vs_power_db": float(
            proposed_metric["SNDR_dB"] - power_metric["SNDR_dB"]
        ),
        "sfdr_improvement_vs_power_db": float(
            proposed_metric["SFDR_dBc_single_bin"]
            - power_metric["SFDR_dBc_single_bin"]
        ),
        "clipped_fraction_A": float(np.mean(data["F_raw_clipped"][0::2])),
        "clipped_fraction_B": float(np.mean(data["F_raw_clipped"][1::2])),
    }

def build_sweep() -> list[dict[str, float]]:
    sweep_ps = np.arange(0.0, 40.0 + 1.0e-9, 2.0)
    return [simulate_power_matching_ablation(float(skew_ps)) for skew_ps in sweep_ps]

def _joint_ls(
    regressor: np.ndarray,
    target: np.ndarray,
    slope: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    X = np.column_stack(
        [regressor[mask], np.ones(int(np.sum(mask))), slope[mask]]
    )
    theta, _, _, _ = np.linalg.lstsq(X, target[mask], rcond=None)
    fitted = theta[0] * regressor + theta[1] + theta[2] * slope
    return theta, fitted

def _assemble_output(U_A: np.ndarray, U_B_corr: np.ndarray) -> np.ndarray:
    output = np.empty(2 * U_A.size, dtype=np.float64)
    output[0::2] = U_A
    output[1::2] = U_B_corr
    return output

def build_aligned_ablation(
    skew_ps: float,
) -> list[dict[str, float | int | str]]:
    """Run four baselines with an identical predictor, mask, slope, and joint LS."""

    fs = 1.0e9
    n_samples = 2**16
    fin = 32703 / n_samples * fs
    injected_gain = 1.02
    data, _, _ = build_synthetic_adc_dataset(
        DatasetConfig(
            N=n_samples,
            seed=20260620,
            signal=SignalConfig(
                fs=fs,
                fin=fin,
                amplitude=505.0,
                phase=0.31,
                dc=0.0,
            ),
            timing=TimingConfig(dt_mdac_A=0.0, dt_mdac_B=skew_ps * 1.0e-12),
            mismatch=MismatchConfig(
                g0=1.0,
                o0=0.0,
                g1=injected_gain,
                o1=0.0,
                g2=1.0,
                o2=0.0,
                g3=injected_gain,
                o3=0.0,
                noise_std=0.0,
            ),
        )
    )
    D_coarse = data["D_coarse"].astype(np.float64)
    F_raw = data["F_raw"].astype(np.float64)
    F_intra, _, _ = apply_intra_group_calibration(F_raw, data["sar_id"])
    group = build_group_streams(D_coarse, F_intra)
    cfg = CalibrationConfig(
        reference_method="known_tone",
        known_tone_source="A",
        inter_method="joint",
    )
    Dhat_B, Fhat_B, slope_B, mask_ref, _ = build_B_reference(
        group, fs, fin, cfg
    )
    mask_safe, _ = build_safe_mask(
        group["F_B"],
        Fhat_B,
        slope_B,
        data["C_raw"][1::2],
        cfg,
    )
    mask_safe &= mask_ref

    fine_a = group["F_A"]
    fine_b = group["F_B"]
    g_power = float(
        np.sqrt(
            np.var(fine_a[mask_safe], ddof=1)
            / np.var(fine_b[mask_safe], ddof=1)
        )
    )
    power_fine = g_power * fine_b

    theta_full, full_code_fit = _joint_ls(
        group["U_B"], Dhat_B, slope_B, mask_safe
    )
    theta_no_sub, no_sub_fine_fit = _joint_ls(
        fine_b, Dhat_B, slope_B, mask_safe
    )
    theta_proposed, proposed_fine_fit = _joint_ls(
        fine_b, Fhat_B, slope_B, mask_safe
    )

    methods = [
        (
            "power_matching",
            "Power matching",
            np.asarray([g_power, 0.0, 0.0]),
            group["D_coarse_B"] + power_fine,
            "Variance-ratio gain on the common safe mask",
        ),
        (
            "full_code_joint_ls",
            "Full-code joint LS",
            theta_full,
            full_code_fit,
            "Dhat_B fitted from [U_B, 1, slope]",
        ),
        (
            "fine_ls_no_local_subtraction",
            "Fine LS, no local subtraction",
            theta_no_sub,
            group["D_coarse_B"] + no_sub_fine_fit,
            "Dhat_B used directly as the fine-domain target",
        ),
        (
            "proposed_decoupled_joint_ls",
            "Proposed decoupled joint LS",
            theta_proposed,
            group["D_coarse_B"] + proposed_fine_fit,
            "Fhat_B = Dhat_B - D_coarse,B",
        ),
    ]
    rows: list[dict[str, float | int | str]] = []
    for key, label, theta, U_B_corr, definition in methods:
        metric = _fft(_assemble_output(group["U_A"], U_B_corr), fs, fin)
        rows.append(
            {
                "method": key,
                "label": label,
                "definition": definition,
                "sample_count": n_samples,
                "safe_sample_count": int(np.sum(mask_safe)),
                "fs_mhz": fs / 1.0e6,
                "fin_mhz": fin / 1.0e6,
                "b_group_timing_skew_ps": float(skew_ps),
                "injected_b_group_fine_gain": injected_gain,
                "estimated_gain": float(theta[0]),
                "estimated_offset_lsb": float(theta[1]),
                "estimated_timing_coefficient": float(theta[2]),
                "sndr_db": float(metric["SNDR_dB"]),
                "sfdr_dbc": float(metric["SFDR_dBc_single_bin"]),
            }
        )
    return rows

def main() -> None:
    print(json.dumps({"variance_sweep": build_sweep(), "target_construction_ablation": build_aligned_ablation(20.0)}, indent=2))


if __name__ == "__main__":
    main()
