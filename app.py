import io
import base64
import math
import os
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import numpy as np

from model import (
    DatasetConfig, SignalConfig, MismatchConfig, TimingConfig,
    build_synthetic_adc_dataset, default_2p5b_redundant_model
)
from fft_output_code import compute_fft_metrics, plot_spectrum
from calibration import (
    CalibrationConfig,
    add_fft_metrics,
    build_comparison_tables,
    calibrate_adc,
    calibrate_adc_background,
    save_calibration_outputs,
)

app = FastAPI(title="ADC Calibration UI")
DEFAULT_SAMPLE_COUNT = 2**13
MIN_SAMPLE_COUNT = 16
MAX_SAMPLE_COUNT = 2**20
MAX_AUTO_COHERENT_SAMPLE_COUNT = 2**18
CHANNEL_COUNT_MULTIPLE = 4
LAST_SIMULATION: dict[str, Any] | None = None

# Ensure static directory exists
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def read_root():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/calibration", response_class=HTMLResponse)
def read_calibration_page():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


class SimulationParams(BaseModel):
    # Signal
    fs: float
    fin: float
    amplitude: float
    sample_count: int = DEFAULT_SAMPLE_COUNT
    auto_coherent: bool = True
    frequency_mode: str = "hz"
    tone_bin: int | None = None
    # Mismatch
    noise_std: float
    g0: float
    o0: float
    g1: float
    o1: float
    g2: float
    o2: float
    g3: float
    o3: float
    # Timing
    dt_mdac_A_ps: float
    dt_mdac_B_ps: float
    # Checkbox
    enable_mismatch: bool


class CalibrationParams(BaseModel):
    reference_method: Literal["known_tone", "fractional_delay_fir", "linear"] = "known_tone"
    fir_taps: int = 15
    known_tone_source: Literal["A", "full"] = "A"
    inter_method: Literal["two_stage", "joint", "nlms"] = "two_stage"
    lms_mu: float = 0.005
    lms_epochs: int = 1
    lms_trace_block_size: int = 1024
    lms_settling_ratio: float = 1.1


def _nearest_multiple(value: int, step: int) -> int:
    lower = (value // step) * step
    upper = lower + step
    if lower < MIN_SAMPLE_COUNT:
        return upper
    return lower if value - lower <= upper - value else upper


def _normalize_sample_count(value: int) -> int:
    if not math.isfinite(float(value)):
        return DEFAULT_SAMPLE_COUNT
    n = int(round(float(value)))
    n = max(MIN_SAMPLE_COUNT, min(MAX_SAMPLE_COUNT, n))
    return _nearest_multiple(n, CHANNEL_COUNT_MULTIPLE)


def _clamp_tone_bin(bin_value: int | None, fs: float, fin: float, n: int) -> int:
    max_bin = max(1, n - 1)
    if bin_value is None:
        bin_value = int(round(fin / fs * n))
    return max(1, min(max_bin, int(bin_value)))


def _alias_tone_bin(tone_bin: int, n: int) -> int:
    wrapped = int(tone_bin) % int(n)
    if wrapped == 0:
        return 0
    return min(wrapped, int(n) - wrapped)


def _coherent_sample_count(
    fs: float,
    fin: float,
    target_n: int,
) -> tuple[int, bool, int | None, str]:
    if fs <= 0.0 or fin <= 0.0:
        return target_n, False, None, "Invalid fs/fin for coherent sizing."

    ratio = Fraction(str(fin)) / Fraction(str(fs))
    if ratio <= 0 or ratio >= Fraction(1, 2):
        return target_n, False, None, "Tone is outside the single-sided FFT range."

    required_multiple = math.lcm(ratio.denominator, CHANNEL_COUNT_MULTIPLE)
    if required_multiple > MAX_AUTO_COHERENT_SAMPLE_COUNT:
        return (
            target_n,
            False,
            None,
            (
                "Exact coherent N would be too large for automatic simulation; "
                "use bin mode to force fin = bin / N * fs."
            ),
        )

    lower = (target_n // required_multiple) * required_multiple
    upper = lower + required_multiple
    candidates = [
        n
        for n in (lower, upper)
        if MIN_SAMPLE_COUNT <= n <= MAX_AUTO_COHERENT_SAMPLE_COUNT
    ]
    if not candidates:
        return target_n, False, None, "No nearby coherent N is within limits."

    n = min(candidates, key=lambda candidate: abs(candidate - target_n))
    tone_bin = int(ratio * n)
    return n, True, tone_bin, "Exact coherent sample count selected."


def _resolve_fft_setup(params: SimulationParams) -> tuple[int, float, dict[str, object]]:
    fs = float(params.fs)
    requested_fin = float(params.fin)
    target_n = _normalize_sample_count(params.sample_count)

    if params.frequency_mode == "bin":
        n = target_n
        tone_bin = _clamp_tone_bin(params.tone_bin, fs, requested_fin, n)
        fin = tone_bin / n * fs
        alias_tone_bin = _alias_tone_bin(tone_bin, n)
        analysis_fin = alias_tone_bin / n * fs
        if tone_bin == alias_tone_bin:
            note = "Frequency set by fin = tone_bin / N * fs."
        else:
            note = (
                "Input bin is above Nyquist for a real sampled sequence; "
                "FFT analysis uses alias_tone_bin = N - tone_bin."
            )
        return n, fin, {
            "frequency_mode": "bin",
            "target_sample_count": target_n,
            "sample_count": n,
            "requested_fin_Hz": requested_fin,
            "fin_Hz": fin,
            "tone_bin": tone_bin,
            "alias_tone_bin": alias_tone_bin,
            "analysis_fin_Hz": analysis_fin,
            "above_nyquist": bool(tone_bin > n // 2),
            "auto_coherent_requested": False,
            "auto_coherent_applied": False,
            "note": note,
        }

    n = target_n
    tone_bin = None
    auto_applied = False
    note = "Using requested sample count."
    if params.auto_coherent:
        n, auto_applied, tone_bin, note = _coherent_sample_count(fs, requested_fin, target_n)

    return n, requested_fin, {
        "frequency_mode": "hz",
        "target_sample_count": target_n,
        "sample_count": n,
        "requested_fin_Hz": requested_fin,
        "fin_Hz": requested_fin,
        "tone_bin": tone_bin,
        "auto_coherent_requested": bool(params.auto_coherent),
        "auto_coherent_applied": auto_applied,
        "note": note,
    }


def _json_ready(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v) for v in obj]
    return obj


def _image_to_base64(buffer: io.BytesIO) -> str:
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def _build_dataset(params: SimulationParams) -> dict[str, Any]:
    sample_count, fin, fft_setup = _resolve_fft_setup(params)
    cfg = DatasetConfig(
        N=sample_count,
        signal=SignalConfig(
            fs=params.fs,
            fin=fin,
            amplitude=params.amplitude,
        ),
        timing=TimingConfig(
            dt_mdac_A=params.dt_mdac_A_ps * 1e-12 if params.enable_mismatch else 0.0,
            dt_mdac_B=params.dt_mdac_B_ps * 1e-12 if params.enable_mismatch else 0.0,
        ),
        mismatch=MismatchConfig(
            noise_std=params.noise_std if params.enable_mismatch else 0.0,
            g0=params.g0 if params.enable_mismatch else 1.0,
            o0=params.o0 if params.enable_mismatch else 0.0,
            g1=params.g1 if params.enable_mismatch else 1.0,
            o1=params.o1 if params.enable_mismatch else 0.0,
            g2=params.g2 if params.enable_mismatch else 1.0,
            o2=params.o2 if params.enable_mismatch else 0.0,
            g3=params.g3 if params.enable_mismatch else 1.0,
            o3=params.o3 if params.enable_mismatch else 0.0,
        ),
    )

    coarse_model = default_2p5b_redundant_model()
    data, truth, model_metrics = build_synthetic_adc_dataset(cfg, coarse_model)
    params_payload = params.model_dump() if hasattr(params, "model_dump") else params.dict()
    return {
        "params": params_payload,
        "sample_count": sample_count,
        "fin": fin,
        "fft_setup": fft_setup,
        "data": data,
        "truth": truth,
        "model_metrics": model_metrics,
    }


def _compute_fft_image(values: np.ndarray, fs: float, fin: float, title: str) -> tuple[dict[str, Any], str]:
    metrics, debug = compute_fft_metrics(
        np.asarray(values, dtype=np.float64),
        fs=fs,
        fin=fin,
        window="auto",
    )
    buf = io.BytesIO()
    plot_spectrum(debug, metrics, output_path=buf, title=title)
    return metrics, _image_to_base64(buf)


def _format_fft_title(prefix: str, sample_count: int, fin: float, fft_setup: dict[str, Any]) -> str:
    analysis_fin = float(fft_setup.get("analysis_fin_Hz", fin))
    if not math.isclose(fin, analysis_fin, rel_tol=0.0, abs_tol=1e-9):
        return (
            f"{prefix} (N={sample_count}, fin={fin / 1e6:.6g} MHz, "
            f"FFT alias={analysis_fin / 1e6:.6g} MHz)"
        )
    return f"{prefix} (N={sample_count}, fin={fin / 1e6:.6g} MHz)"


def _plot_calibration_summary(
    metrics: dict[str, Any],
    coeffs: dict[str, float],
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stages = [("raw", "Raw"), ("after_intra", "Intra"), ("after_full", "Full")]
    err = metrics.get("error_vs_ideal", {})
    rms_values = [
        err.get("rms_raw_lsb"),
        err.get("rms_after_intra_lsb"),
        err.get("rms_after_full_lsb"),
    ]
    fft = metrics.get("fft", {})
    sndr_values = [fft.get(key, {}).get("SNDR_dB") for key, _ in stages]
    sfdr_values = [fft.get(key, {}).get("SFDR_dBc_single_bin") for key, _ in stages]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.patch.set_facecolor("#ffffff")

    labels = [label for _, label in stages]
    ax = axes[0, 0]
    ax.bar(labels, [0.0 if value is None else value for value in rms_values], color=["#8a8f98", "#2f80ed", "#10a37f"])
    ax.set_title("RMS Error vs D_ideal")
    ax.set_ylabel("LSB")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[0, 1]
    x = np.arange(len(labels))
    width = 0.36
    ax.bar(x - width / 2, [0.0 if value is None else value for value in sndr_values], width, label="SNDR", color="#2f80ed")
    ax.bar(x + width / 2, [0.0 if value is None else value for value in sfdr_values], width, label="SFDR", color="#f59e0b")
    ax.set_title("FFT Metrics")
    ax.set_ylabel("dB")
    ax.set_xticks(x, labels)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    truth_compare = metrics.get("truth_compare", {})
    coeff_names = ["g20", "g31", "g_B"]
    estimated = [coeffs.get(name, np.nan) for name in coeff_names]
    truth = [truth_compare.get(name, {}).get("true", np.nan) for name in coeff_names]
    ax = axes[1, 0]
    x = np.arange(len(coeff_names))
    ax.bar(x - width / 2, estimated, width, label="Estimated", color="#10a37f")
    ax.bar(x + width / 2, truth, width, label="Truth", color="#8a8f98")
    ax.set_title("Gain Coefficients")
    ax.set_xticks(x, coeff_names)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    ref = metrics.get("inter", {}).get("reference_tracking", {})
    before = ref.get("rms_U_B_minus_Dhat_safe_before_lsb")
    after = ref.get("rms_U_B_corr_minus_Dhat_safe_after_lsb")
    ax = axes[1, 1]
    ax.bar(["Before", "After"], [0.0 if before is None else before, 0.0 if after is None else after], color=["#8a8f98", "#10a37f"])
    ax.set_title("B Stream Reference Tracking")
    ax.set_ylabel("RMS LSB")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, dpi=160, format="png")
    plt.close(fig)
    return _image_to_base64(buf)


def _build_timing_mismatch_table(
    truth: dict[str, Any],
    coeffs: dict[str, float],
    debug: dict[str, Any],
    metrics: dict[str, Any],
    cfg: CalibrationConfig,
) -> dict[str, Any]:
    reference = debug.get("reference", {})
    timing_compare = metrics.get("timing_compare", {})
    dt_a_ps = float(truth.get("dt_mdac_A", 0.0)) * 1e12
    dt_b_ps = float(truth.get("dt_mdac_B", 0.0)) * 1e12
    reference_detail = cfg.reference_method
    if cfg.reference_method == "known_tone":
        reference_detail += f" / source={cfg.known_tone_source}"
    elif cfg.reference_method == "fractional_delay_fir":
        effective_taps = reference.get("fir_effective_taps", cfg.fir_taps)
        requested_taps = reference.get("fir_requested_taps", cfg.fir_taps)
        if effective_taps != requested_taps:
            reference_detail += f" / taps={requested_taps}->{effective_taps}"
        else:
            reference_detail += f" / taps={effective_taps}"

    return {
        "title": "Timing-like mismatch summary",
        "headers": ["Item", "Actual / expected", "Fitted / observed"],
        "rows": [
            ["MDAC-A skew", dt_a_ps, "-"],
            ["MDAC-B skew", dt_b_ps, "-"],
            [
                "B-A aperture skew (ps)",
                timing_compare.get("actual_B_minus_A_ps", dt_b_ps - dt_a_ps),
                timing_compare.get("fitted_B_minus_A_ps"),
            ],
            ["B-A skew fit error (ps)", "-", timing_compare.get("B_minus_A_error_ps")],
            [
                "B-A skew fit error (%)",
                "-",
                timing_compare.get("B_minus_A_error_percent"),
            ],
            [
                "k_B timing-like coefficient",
                timing_compare.get("expected_k_B"),
                coeffs.get("k_B"),
            ],
            ["slope basis", truth.get("slope_basis", "-"), "-"],
            ["slope center", reference.get("slope_center"), "-"],
            ["slope scale", reference.get("slope_scale"), "-"],
            ["slope basis factor", timing_compare.get("slope_basis_factor"), "-"],
            ["reference method", reference_detail, "-"],
        ],
    }


@app.post("/api/simulate")
def simulate(params: SimulationParams):
    global LAST_SIMULATION

    simulation = _build_dataset(params)
    data = simulation["data"]
    fin = float(simulation["fin"])
    fft_setup = simulation["fft_setup"]
    sample_count = int(simulation["sample_count"])
    y = data["D_raw"] if params.enable_mismatch else data["D_no_mismatch"]

    mode = "Mismatch" if params.enable_mismatch else "Ideal"
    title = _format_fft_title(f"FFT Spectrum ({mode})", sample_count, fin, fft_setup)
    metrics, img_b64 = _compute_fft_image(y, fs=float(data["fs"]), fin=fin, title=title)
    LAST_SIMULATION = simulation

    return {
        "metrics": _json_ready(metrics),
        "fft_setup": _json_ready(simulation["fft_setup"]),
        "model_metrics": _json_ready(simulation["model_metrics"]),
        "image_b64": img_b64
    }


@app.post("/api/calibrate")
def calibrate_latest(params: CalibrationParams):
    if LAST_SIMULATION is None:
        raise HTTPException(status_code=400, detail="请先运行一次仿真，再启动校准。")

    data = LAST_SIMULATION["data"]
    truth = LAST_SIMULATION["truth"]
    fs = float(data["fs"])
    fin = float(data["fin"])
    fft_setup = LAST_SIMULATION["fft_setup"]
    sample_count = int(LAST_SIMULATION["sample_count"])
    cfg = CalibrationConfig(
        reference_method=params.reference_method,
        fir_taps=params.fir_taps,
        known_tone_source=params.known_tone_source,
        inter_method=params.inter_method,
        lms_mu=params.lms_mu,
        lms_epochs=params.lms_epochs,
        lms_trace_block_size=params.lms_trace_block_size,
        lms_settling_ratio=params.lms_settling_ratio,
    )

    try:
        calibrator = calibrate_adc_background if cfg.inter_method == "nlms" else calibrate_adc
        D_corr, coeffs, debug, metrics = calibrator(
            data["D_coarse"],
            data["F_raw"],
            sar_id=data.get("sar_id"),
            C_raw=data.get("C_raw"),
            fs=fs,
            fin=fin,
            D_ideal=data.get("D_ideal"),
            truth=truth,
            config=cfg,
        )
        D_raw = np.asarray(data["D_coarse"], dtype=np.float64) + np.asarray(data["F_raw"], dtype=np.float64)
        add_fft_metrics(
            metrics,
            D_raw,
            debug["D_after_intra"],
            D_corr,
            fs=fs,
            fin=fin,
        )
        metrics["tables"] = build_comparison_tables(metrics, coeffs, truth)
        metrics["tables"]["timing_mismatch"] = _build_timing_mismatch_table(
            truth,
            coeffs,
            debug,
            metrics,
            cfg,
        )
        raw_fft_metrics, raw_fft_b64 = _compute_fft_image(
            D_raw,
            fs=fs,
            fin=fin,
            title=_format_fft_title("Raw FFT Spectrum", sample_count, fin, fft_setup),
        )
        calibrated_fft_metrics, calibrated_fft_b64 = _compute_fft_image(
            D_corr,
            fs=fs,
            fin=fin,
            title=_format_fft_title("Calibrated FFT Spectrum", sample_count, fin, fft_setup),
        )
        stem = "ui_latest_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        files = save_calibration_outputs(
            Path("outputs") / "calibration",
            stem,
            D_raw,
            D_corr,
            coeffs,
            debug,
            metrics,
        )
        chart_b64 = _plot_calibration_summary(metrics, coeffs)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"校准参数不适用：{exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"校准失败：{exc}") from exc

    err = metrics.get("error_vs_ideal", {})
    raw_rms = err.get("rms_raw_lsb")
    full_rms = err.get("rms_after_full_lsb")
    improvement = None
    if raw_rms not in (None, 0) and full_rms is not None:
        improvement = float(raw_rms / full_rms)

    return {
        "config": _json_ready(metrics.get("config", {})),
        "sample_count": sample_count,
        "fin_Hz": fin,
        "fft_setup": _json_ready(fft_setup),
        "coeffs": _json_ready(coeffs),
        "metrics": _json_ready(metrics),
        "tables": _json_ready(metrics.get("tables", {})),
        "files": _json_ready(files),
        "summary": {
            "rms_raw_lsb": raw_rms,
            "rms_after_full_lsb": full_rms,
            "rms_improvement_x": improvement,
            "sndr_raw_dB": metrics.get("fft", {}).get("raw", {}).get("SNDR_dB"),
            "sndr_after_full_dB": metrics.get("fft", {}).get("after_full", {}).get("SNDR_dB"),
            "enob_raw_bits": metrics.get("fft", {}).get("raw", {}).get("ENOB_bits"),
            "enob_after_full_bits": metrics.get("fft", {}).get("after_full", {}).get("ENOB_bits"),
            "enob_improvement_bits": (metrics.get("fft", {}).get("after_full", {}).get("ENOB_bits", 0) or 0) - (metrics.get("fft", {}).get("raw", {}).get("ENOB_bits", 0) or 0),
            "sfdr_raw_dBc": metrics.get("fft", {}).get("raw", {}).get("SFDR_dBc_single_bin"),
            "sfdr_after_full_dBc": metrics.get("fft", {}).get("after_full", {}).get("SFDR_dBc_single_bin"),
            "safe_samples": metrics.get("inter", {}).get("mask_counts", {}).get("safe_after_reference"),
        },
        "image_b64": chart_b64,
        "spectrum_images": {
            "raw": raw_fft_b64,
            "calibrated": calibrated_fft_b64,
        },
        "spectrum_metrics": {
            "raw": _json_ready(raw_fft_metrics),
            "calibrated": _json_ready(calibrated_fft_metrics),
        },
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
