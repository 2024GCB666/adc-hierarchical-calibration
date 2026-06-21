import math

from app import SimulationParams, _build_dataset, _resolve_fft_setup
from fft_output_code import compute_fft_metrics


def _simulation_params(tone_bin: int) -> SimulationParams:
    return SimulationParams(
        fs=1.0e9,
        fin=61.0504e6,
        amplitude=511.0,
        sample_count=65536,
        auto_coherent=False,
        frequency_mode="bin",
        tone_bin=tone_bin,
        enable_mismatch=True,
        noise_std=0.08,
        g0=1.0,
        o0=0.0,
        g1=1.02,
        o1=2.0,
        g2=1.012,
        o2=-1.5,
        g3=1.008,
        o3=3.2,
        dt_mdac_A_ps=0.0,
        dt_mdac_B_ps=38.0,
    )


def test_bin_mode_keeps_above_nyquist_bin_and_reports_alias() -> None:
    n, fin, setup = _resolve_fft_setup(_simulation_params(65000))

    assert n == 65536
    assert setup["tone_bin"] == 65000
    assert setup["alias_tone_bin"] == 536
    assert setup["above_nyquist"] is True
    assert math.isclose(fin, 65000 / 65536 * 1.0e9)
    assert math.isclose(setup["analysis_fin_Hz"], 536 / 65536 * 1.0e9)


def test_fft_metrics_use_alias_for_above_nyquist_input_bin() -> None:
    simulation = _build_dataset(_simulation_params(65000))
    data = simulation["data"]
    metrics, _ = compute_fft_metrics(
        data["D_raw"],
        fs=float(data["fs"]),
        fin=float(simulation["fin"]),
        window="auto",
    )

    assert metrics["fundamental_bin"] == 536
    assert metrics["largest_spur_bin"] == 32232
    assert metrics["fin_requested_Hz"] > 0.5 * float(data["fs"])
    assert math.isclose(metrics["fin_analysis_Hz"], 536 / 65536 * 1.0e9)
    assert metrics["ENOB_bits"] < 4.0


def test_fft_metrics_count_bin_one_spur_near_nyquist() -> None:
    simulation = _build_dataset(_simulation_params(32767))
    data = simulation["data"]
    metrics, _ = compute_fft_metrics(
        data["D_raw"],
        fs=float(data["fs"]),
        fin=float(simulation["fin"]),
        window="auto",
    )

    assert metrics["fundamental_bin"] == 32767
    assert metrics["largest_spur_bin"] == 1
    assert metrics["ENOB_bits"] < 5.0
