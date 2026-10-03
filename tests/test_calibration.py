"""Smoke tests for the layered ADC calibration implementation."""

from __future__ import annotations

import pytest

import numpy as np

from calibration import (
    CalibrationConfig,
    apply_fir_differentiator,
    build_fir_differentiator,
    calibrate_adc,
    calibrate_adc_background,
    run_background_nlms_stream,
)
from calibration import build_timing_mismatch_comparison
from model import (
    DatasetConfig,
    MismatchConfig,
    SignalConfig,
    TimingConfig,
    build_synthetic_adc_dataset,
    timing_case,
)


def _build_mismatch_dataset(fin: float, N: int = 2**16):
    fs = 1.0e9
    cfg = DatasetConfig(
        N=N,
        seed=20260614,
        signal=SignalConfig(fs=fs, fin=fin, amplitude=511.0),
        timing=timing_case("case_0"),
        mismatch=MismatchConfig(
            g0=1.0,
            o0=0.0,
            g1=1.02,
            o1=2.0,
            g2=1.012,
            o2=-1.5,
            g3=1.008,
            o3=3.2,
            noise_std=0.08,
        ),
    )
    return build_synthetic_adc_dataset(cfg)


def test_streaming_nlms_applies_state_before_using_current_error() -> None:
    corrected, coeffs, diagnostics = run_background_nlms_stream(
        np.asarray([10.0]),
        np.asarray([20.0]),
        np.asarray([0.0]),
        np.asarray([True]),
        mu=0.5,
        fine_scale=128.0,
        trace_block_size=1,
    )
    assert corrected[0] == pytest.approx(10.0)
    assert coeffs["g_B"] != pytest.approx(1.0)
    assert diagnostics["apply_then_update"] is True
    assert diagnostics["initialization"] == "explicit_state_no_data_fit"


def test_streaming_nlms_cold_start_recovers_fixed_scale_coefficients() -> None:
    rng = np.random.default_rng(20260712)
    fine = rng.uniform(-100.0, 100.0, 20000)
    slope = rng.uniform(-1.0, 1.0, fine.size)
    target = 1.02 * fine + 2.0 - 2.56 * slope
    _, coeffs, diagnostics = run_background_nlms_stream(
        fine,
        target,
        slope,
        np.ones(fine.size, dtype=bool),
        mu=0.01,
        fine_scale=128.0,
    )
    assert coeffs["g_B"] == pytest.approx(1.02, abs=2e-4)
    assert coeffs["o_B"] == pytest.approx(2.0, abs=2e-3)
    assert coeffs["k_B"] == pytest.approx(-2.56, abs=2e-3)
    assert diagnostics["fine_scale"] == pytest.approx(128.0)


def test_known_tone_layered_calibration_reduces_error() -> None:
    fs = 1.0e9
    N = 2**16
    fin = 4001 / N * fs
    cfg = DatasetConfig(
        N=N,
        seed=20260614,
        signal=SignalConfig(fs=fs, fin=fin, amplitude=511.0),
        timing=timing_case("case_0"),
        mismatch=MismatchConfig(
            g0=1.0,
            o0=0.0,
            g1=1.02,
            o1=2.0,
            g2=1.012,
            o2=-1.5,
            g3=1.0076,
            o3=3.2,
            noise_std=0.08,
        ),
    )
    data, truth, _ = build_synthetic_adc_dataset(cfg)
    _, coeffs, _, metrics = calibrate_adc(
        data["D_coarse"],
        data["F_raw"],
        sar_id=data["sar_id"],
        C_raw=data["C_raw"],
        fs=data["fs"],
        fin=data["fin"],
        D_ideal=data["D_ideal"],
        truth=truth,
        config=CalibrationConfig(reference_method="known_tone"),
    )

    err = metrics["error_vs_ideal"]
    assert err["rms_after_full_lsb"] < 0.4
    assert err["rms_after_full_lsb"] < 0.2 * err["rms_raw_lsb"]
    assert abs(coeffs["g20"] - 1.012) < 2e-3
    assert abs(coeffs["g_B"] - 1.02) < 3e-3
    assert abs(coeffs["o_B"] - 2.0) < 0.05


def test_high_frequency_fractional_delay_fir_auto_taps() -> None:
    fs = 1.0e9
    N = 2**16
    fin = 15000 / N * fs
    data, truth, _ = _build_mismatch_dataset(fin, N)

    _, coeffs, _, metrics = calibrate_adc(
        data["D_coarse"],
        data["F_raw"],
        sar_id=data["sar_id"],
        C_raw=data["C_raw"],
        fs=data["fs"],
        fin=data["fin"],
        D_ideal=data["D_ideal"],
        truth=truth,
        config=CalibrationConfig(
            reference_method="fractional_delay_fir",
            fir_taps=15,
            inter_method="two_stage",
        ),
    )

    config = metrics["config"]
    err = metrics["error_vs_ideal"]
    assert config["fir_auto_taps_applied"] is True
    assert config["fir_effective_taps"] > config["fir_requested_taps"]
    assert coeffs["g_B"] > 0.9
    assert err["rms_after_full_lsb"] < 0.25 * err["rms_raw_lsb"]


def test_background_nlms_layered_calibration_reduces_error() -> None:
    fs = 1.0e9
    N = 2**16
    fin = 4001 / N * fs
    data, truth, _ = _build_mismatch_dataset(fin, N)

    _, coeffs, _, metrics = calibrate_adc_background(
        data["D_coarse"],
        data["F_raw"],
        sar_id=data["sar_id"],
        C_raw=data["C_raw"],
        fs=data["fs"],
        fin=data["fin"],
        D_ideal=data["D_ideal"],
        truth=truth,
        config=CalibrationConfig(
            reference_method="known_tone",
            inter_method="nlms",
            lms_trace_block_size=512,
            lms_settling_ratio=1.1,
        ),
    )

    err = metrics["error_vs_ideal"]
    diagnostics = metrics["inter"]["diagnostics"]
    assert diagnostics["adaptive_method"] == "nlms"
    assert diagnostics["lms_update_count"] > 0
    assert diagnostics["lms_trace_block_size"] == 512
    assert len(diagnostics["lms_block_trace"]) > 3
    assert diagnostics["convergence_reached"] is True
    assert diagnostics["convergence_samples"] is not None
    assert diagnostics["convergence_samples"] <= diagnostics["lms_update_count"]
    assert diagnostics["rms_decay_db_per_sample"] is not None
    # The full-record prequential RMS includes the physical cold-start
    # transient.  Steady-state quality is evaluated on the final quarter.
    assert err["rms_after_full_tail_quarter_lsb"] < 0.4
    assert err["rms_after_full_lsb"] < err["rms_raw_lsb"]
    assert abs(coeffs["g_B"] - 1.02) < 4e-3
    assert abs(coeffs["o_B"] - 2.0) < 0.08


def test_fractional_delay_fir_rejects_A_group_nyquist() -> None:
    fs = 1.0e9
    N = 2**16
    fin = 17000 / N * fs
    data, truth, _ = _build_mismatch_dataset(fin, N)

    with pytest.raises(ValueError, match="A-group Nyquist"):
        calibrate_adc(
            data["D_coarse"],
            data["F_raw"],
            sar_id=data["sar_id"],
            C_raw=data["C_raw"],
            fs=data["fs"],
            fin=data["fin"],
            D_ideal=data["D_ideal"],
            truth=truth,
            config=CalibrationConfig(reference_method="fractional_delay_fir"),
        )


def test_fir_differentiator_has_explicit_full_rate_units() -> None:
    h = build_fir_differentiator(31)
    ramp = 3.25 * np.arange(128, dtype=np.float64) + 7.0
    derivative_per_a_sample = apply_fir_differentiator(ramp, h)
    assert derivative_per_a_sample[15:-15] == pytest.approx(3.25, abs=1e-12)


def test_mode_b_fir_derivative_reports_cascade_latency() -> None:
    fs = 1.0e9
    N = 2**16
    fin = 4001 / N * fs
    data, truth, _ = _build_mismatch_dataset(fin, N)

    _, _, debug, metrics = calibrate_adc(
        data["D_coarse"],
        data["F_raw"],
        sar_id=data["sar_id"],
        C_raw=data["C_raw"],
        fs=data["fs"],
        fin=None,
        D_ideal=data["D_ideal"],
        truth=truth,
        config=CalibrationConfig(
            reference_method="fractional_delay_fir",
            fir_taps=15,
            differentiator_taps=31,
            inter_method="joint",
        ),
    )

    info = debug["reference"]
    assert info["slope_unit"] == "LSB/full-rate-sample"
    assert info["predictor_group_delay_A_samples"] == 7
    assert info["differentiator_group_delay_A_samples"] == 15
    assert info["total_group_delay_A_samples"] == 22
    assert info["total_group_delay_full_rate_samples"] == 44
    assert metrics["config"]["total_group_delay_full_rate_samples"] == 44
    assert metrics["timing_compare"]["slope_basis_factor"] == pytest.approx(1.0)


def test_timing_mismatch_comparison_reports_equivalent_skew() -> None:
    fs = 1.0e9
    N = 2**16
    fin = 4001 / N * fs
    cfg = DatasetConfig(
        N=N,
        seed=20260614,
        signal=SignalConfig(fs=fs, fin=fin, amplitude=511.0),
        timing=TimingConfig(dt_mdac_A=30e-12, dt_mdac_B=50e-12),
        mismatch=MismatchConfig(
            g0=1.0,
            o0=0.0,
            g1=1.02,
            o1=2.0,
            g2=1.012,
            o2=-1.5,
            g3=1.008,
            o3=3.2,
            noise_std=0.08,
        ),
    )
    data, truth, _ = build_synthetic_adc_dataset(cfg)
    _, coeffs, debug, metrics = calibrate_adc(
        data["D_coarse"],
        data["F_raw"],
        sar_id=data["sar_id"],
        C_raw=data["C_raw"],
        fs=data["fs"],
        fin=data["fin"],
        D_ideal=data["D_ideal"],
        truth=truth,
        config=CalibrationConfig(reference_method="known_tone"),
    )

    comparison = build_timing_mismatch_comparison(
        coeffs,
        truth,
        debug["reference"],
        "known_tone",
        data["fs"],
        data["fin"],
    )
    assert comparison == metrics["timing_compare"]
    assert comparison["actual_B_minus_A_ps"] == pytest.approx(20.0)
    assert comparison["fitted_B_minus_A_ps"] == pytest.approx(20.0, abs=1.0)
    assert comparison["estimated_k_B"] == pytest.approx(
        comparison["expected_k_B"],
        abs=0.2,
    )


if __name__ == "__main__":
    test_known_tone_layered_calibration_reduces_error()
    test_high_frequency_fractional_delay_fir_auto_taps()
    test_background_nlms_layered_calibration_reduces_error()
    test_fractional_delay_fir_rejects_A_group_nyquist()
    test_timing_mismatch_comparison_reports_equivalent_skew()
    print("Calibration smoke test passed.")
