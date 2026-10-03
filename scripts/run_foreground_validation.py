"""Numerical foreground behavioral validation and frequency robustness sweep."""
from __future__ import annotations
import argparse
import json
from dataclasses import replace
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from calibration import CalibrationConfig, add_fft_metrics, calibrate_adc
from model import DatasetConfig, MismatchConfig, SignalConfig, TimingConfig, build_synthetic_adc_dataset

def static_mismatch_case() -> MismatchConfig:
    """Same static fine-path mismatch case used for Table II."""

    return MismatchConfig(
        g0=1.0,
        o0=0.0,
        g1=1.02,
        o1=2.0,
        g2=1.012,
        o2=-1.5,
        g3=1.0076,
        o3=3.2,
        noise_std=0.08,
    )

def nearest_odd_tone_bin(target_mhz: float, n_samples: int, fs: float) -> int:
    """Choose a coherent odd FFT bin near the requested input frequency."""

    tone_bin = int(round(target_mhz * 1.0e6 / fs * n_samples))
    if tone_bin % 2 == 0:
        tone_bin += 1
    return max(1, min(tone_bin, n_samples // 2 - 1))

def simulate_frequency_point(
    target_mhz: float,
    *,
    n_samples: int = 2**16,
    fs: float = 1.0e9,
    seed: int = 20260614,
) -> dict[str, float]:
    """Run one behavioral-model calibration point for the input-frequency sweep."""

    tone_bin = nearest_odd_tone_bin(target_mhz, n_samples, fs)
    fin = tone_bin / n_samples * fs
    cfg = DatasetConfig(
        N=n_samples,
        seed=seed,
        signal=SignalConfig(
            fs=fs,
            fin=fin,
            amplitude=511.0,
            phase=0.31,
            dc=0.0,
        ),
        # Fig. 5 isolates frequency robustness under fixed static mismatch.
        # MDAC timing-skew robustness is intentionally left to a separate stress case.
        timing=TimingConfig(dt_mdac_A=0.0, dt_mdac_B=0.0),
        mismatch=static_mismatch_case(),
    )
    data, truth, _ = build_synthetic_adc_dataset(cfg)
    calibration_cfg = CalibrationConfig(
        reference_method="known_tone",
        known_tone_source="A",
        inter_method="two_stage",
    )
    D_corr, coeffs, debug, metrics = calibrate_adc(
        data["D_coarse"],
        data["F_raw"],
        sar_id=data["sar_id"],
        C_raw=data["C_raw"],
        fs=float(data["fs"]),
        fin=float(data["fin"]),
        D_ideal=data["D_ideal"],
        truth=truth,
        config=calibration_cfg,
    )
    D_raw = data["D_coarse"].astype(np.float64) + data["F_raw"].astype(np.float64)
    add_fft_metrics(
        metrics,
        D_raw,
        debug["D_after_intra"],
        D_corr,
        fs=float(data["fs"]),
        fin=float(data["fin"]),
    )

    fft = metrics["fft"]
    err = metrics["error_vs_ideal"]
    return {
        "target_fin_mhz": float(target_mhz),
        "coherent_tone_bin": float(tone_bin),
        "fin_mhz": float(fin / 1.0e6),
        "raw_sndr_db": float(fft["raw"]["SNDR_dB"]),
        "intra_sndr_db": float(fft["after_intra"]["SNDR_dB"]),
        "full_sndr_db": float(fft["after_full"]["SNDR_dB"]),
        "raw_sfdr_dbc": float(fft["raw"]["SFDR_dBc_single_bin"]),
        "intra_sfdr_dbc": float(fft["after_intra"]["SFDR_dBc_single_bin"]),
        "full_sfdr_dbc": float(fft["after_full"]["SFDR_dBc_single_bin"]),
        "raw_enob_bit": float(fft["raw"]["ENOB_bits"]),
        "intra_enob_bit": float(fft["after_intra"]["ENOB_bits"]),
        "full_enob_bit": float(fft["after_full"]["ENOB_bits"]),
        "raw_rms_lsb": float(err["rms_raw_lsb"]),
        "intra_rms_lsb": float(err["rms_after_intra_lsb"]),
        "full_rms_lsb": float(err["rms_after_full_lsb"]),
        "g20": float(coeffs["g20"]),
        "g31": float(coeffs["g31"]),
        "g_B": float(coeffs["g_B"]),
        "o_B_lsb": float(coeffs["o_B"]),
        "k_B": float(coeffs["k_B"]),
    }

def build_sweep() -> list[dict[str, float]]:
    target_mhz = [
        10.0,
        30.0,
        61.05,
        100.0,
        150.0,
        200.0,
        250.0,
        300.0,
        350.0,
        400.0,
        450.0,
        490.0,
    ]
    return [simulate_frequency_point(freq) for freq in target_mhz]

def run_single_tone(skew_ps: float = 0.0) -> dict:
    mismatch = static_mismatch_case()
    if skew_ps:
        mismatch = replace(mismatch, g3=1.008)
    skew_a_ps = 30.0 if skew_ps else 0.0
    cfg = DatasetConfig(
        N=65536,
        seed=20260614,
        signal=SignalConfig(fs=1e9, fin=4001 / 65536 * 1e9, amplitude=511.0, phase=0.31),
        timing=TimingConfig(dt_mdac_A=skew_a_ps * 1e-12, dt_mdac_B=(skew_a_ps + skew_ps) * 1e-12),
        mismatch=mismatch,
    )
    data, truth, _ = build_synthetic_adc_dataset(cfg)
    corrected, coeffs, debug, metrics = calibrate_adc(
        data["D_coarse"], data["F_raw"], sar_id=data["sar_id"], C_raw=data["C_raw"],
        fs=float(data["fs"]), fin=float(data["fin"]), D_ideal=data["D_ideal"], truth=truth,
        config=CalibrationConfig(reference_method="known_tone", inter_method="two_stage"),
    )
    add_fft_metrics(metrics, data["D_coarse"] + data["F_raw"], debug["D_after_intra"], corrected,
                    fs=float(data["fs"]), fin=float(data["fin"]))
    return {"coefficients": coeffs, "rms_error": metrics["error_vs_ideal"],
            "fft": metrics["fft"], "timing": metrics["timing_compare"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper Mode-A numerical validation.")
    parser.add_argument("--frequency-sweep", action="store_true", help="Include the 12 input-frequency points used in the paper.")
    args = parser.parse_args()
    report = {"behavioral_single_tone": run_single_tone(), "aperture_skew_20ps": run_single_tone(20.0)}
    if args.frequency_sweep:
        report["frequency_sweep"] = build_sweep()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
