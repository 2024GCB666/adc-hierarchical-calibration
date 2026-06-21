from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from calibration import CalibrationConfig, calibrate_adc  # noqa: E402
from fft_output_code import compute_fft_metrics  # noqa: E402
from model import (  # noqa: E402
    DatasetConfig,
    MismatchConfig,
    SignalConfig,
    TimingConfig,
    build_synthetic_adc_dataset,
)


# Editable text for SVG/PDF post-processing.
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42


BLACK = "#111111"
GRAY = "#6F6F6F"
LIGHT_GRAY = "#E6E6E6"
BLUE = "#0F4D92"
GREEN = "#0B7A53"
RED = "#B64342"
FUND = "#2E7D32"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 11.0,
            "axes.linewidth": 0.9,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "savefig.facecolor": "white",
        }
    )


def table_ii_static_mismatch() -> MismatchConfig:
    """Same static fine-path mismatch case used in Table II."""

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


def build_behavioral_case() -> tuple[dict[str, np.ndarray], dict[str, dict[str, float]], dict[str, dict[str, np.ndarray]]]:
    """Generate the three waveform stages and FFT spectra for Fig. 4."""

    fs = 1.0e9
    n_samples = 2**16
    tone_bin = 4001
    fin = tone_bin / n_samples * fs
    cfg = DatasetConfig(
        N=n_samples,
        seed=20260614,
        signal=SignalConfig(
            fs=fs,
            fin=fin,
            amplitude=511.0,
            phase=0.31,
            dc=0.0,
        ),
        timing=TimingConfig(dt_mdac_A=0.0, dt_mdac_B=0.0),
        mismatch=table_ii_static_mismatch(),
    )
    data, truth, _ = build_synthetic_adc_dataset(cfg)
    calibration_cfg = CalibrationConfig(
        reference_method="known_tone",
        known_tone_source="A",
        inter_method="two_stage",
    )
    D_corr, _, debug, _ = calibrate_adc(
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
    stages = {
        "raw": D_raw,
        "after_intra": debug["D_after_intra"].astype(np.float64),
        "after_full": D_corr.astype(np.float64),
    }
    metrics: dict[str, dict[str, float]] = {}
    spectra: dict[str, dict[str, np.ndarray]] = {}
    for stage, values in stages.items():
        metric, spectrum_debug = compute_fft_metrics(
            values,
            fs=float(data["fs"]),
            fin=float(data["fin"]),
            window="auto",
        )
        metrics[stage] = metric
        spectra[stage] = spectrum_debug
    return stages, metrics, spectra


def write_source_data(
    metrics: dict[str, dict[str, float]],
    spectra: dict[str, dict[str, np.ndarray]],
    spectrum_path: Path,
    metrics_path: Path,
) -> None:
    with spectrum_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["stage", "fft_bin", "freq_mhz", "spectrum_dbc"])
        for stage, debug in spectra.items():
            freq_mhz = debug["freq"] / 1.0e6
            spectrum_dbc = debug["bin_power_dbc"]
            for bin_idx, (freq, dbc) in enumerate(zip(freq_mhz, spectrum_dbc)):
                writer.writerow([stage, bin_idx, f"{freq:.9f}", f"{dbc:.9f}"])

    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "stage",
            "SNDR_dB",
            "SFDR_dBc_single_bin",
            "SFDR_dBc_integrated",
            "ENOB_bits",
            "fundamental_bin",
            "fundamental_freq_MHz",
            "largest_spur_bin",
            "largest_spur_freq_MHz",
            "largest_spur_dBc_single_bin",
            "window",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for stage, metric in metrics.items():
            writer.writerow(
                {
                    "stage": stage,
                    "SNDR_dB": f"{metric['SNDR_dB']:.9f}",
                    "SFDR_dBc_single_bin": f"{metric['SFDR_dBc_single_bin']:.9f}",
                    "SFDR_dBc_integrated": f"{metric['SFDR_dBc_integrated']:.9f}",
                    "ENOB_bits": f"{metric['ENOB_bits']:.9f}",
                    "fundamental_bin": int(metric["fundamental_bin"]),
                    "fundamental_freq_MHz": f"{metric['fundamental_bin_freq_Hz'] / 1.0e6:.9f}",
                    "largest_spur_bin": int(metric["largest_spur_bin"]),
                    "largest_spur_freq_MHz": f"{metric['largest_spur_freq_Hz'] / 1.0e6:.9f}",
                    "largest_spur_dBc_single_bin": f"{metric['largest_spur_dBc_single_bin']:.9f}",
                    "window": metric["window"],
                }
            )


def plot_stage_spectrum(
    ax: plt.Axes,
    stage_key: str,
    stage_label: str,
    color: str,
    metric: dict[str, float],
    debug: dict[str, np.ndarray],
    *,
    floor_db: float,
) -> None:
    freq_mhz = debug["freq"] / 1.0e6
    spectrum_dbc = debug["bin_power_dbc"]
    display = np.maximum(spectrum_dbc, floor_db)
    ax.plot(freq_mhz, display, color=color, lw=0.45, alpha=0.95)
    ax.fill_between(freq_mhz, floor_db, display, color=color, alpha=0.08, lw=0)

    fund_bin = int(metric["fundamental_bin"])
    spur_bin = int(metric["largest_spur_bin"])
    ax.plot(
        freq_mhz[fund_bin],
        spectrum_dbc[fund_bin],
        marker="o",
        ms=5.0,
        mfc="white",
        mec=FUND,
        mew=1.1,
        linestyle="none",
        zorder=5,
    )
    ax.plot(
        freq_mhz[spur_bin],
        max(spectrum_dbc[spur_bin], floor_db),
        marker="o",
        ms=5.2,
        mfc="white",
        mec=RED,
        mew=1.1,
        linestyle="none",
        zorder=5,
    )
    ax.text(
        0.012,
        0.86,
        stage_label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12.2,
        color=BLACK,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.92, "pad": 1.5},
    )
    metric_text = (
        f"SNDR {metric['SNDR_dB']:.2f} dB\n"
        f"SFDR {metric['SFDR_dBc_single_bin']:.2f} dBc\n"
        f"ENOB {metric['ENOB_bits']:.2f} bit"
    )
    ax.text(
        0.985,
        0.86,
        metric_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10.8,
        color=BLACK,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "#DDDDDD",
            "alpha": 0.92,
        },
    )
    ax.set_ylim(floor_db, 5)
    ax.set_xlim(0, 505)
    ax.set_yticks([-100, -75, -50, -25, 0])
    ax.grid(axis="y", color=LIGHT_GRAY, lw=0.55)


def build_figure(
    metrics: dict[str, dict[str, float]],
    spectra: dict[str, dict[str, np.ndarray]],
) -> plt.Figure:
    setup_style()
    floor_db = -115.0
    fig, axes = plt.subplots(3, 1, figsize=(7.05, 6.9), sharex=True, sharey=True)
    stage_specs = [
        ("raw", "a  Raw output", GRAY),
        ("after_intra", "b  After intra-group calibration", BLUE),
        ("after_full", "c  After full group-predictive calibration", GREEN),
    ]
    for ax, (stage_key, label, color) in zip(axes, stage_specs):
        plot_stage_spectrum(
            ax,
            stage_key,
            label,
            color,
            metrics[stage_key],
            spectra[stage_key],
            floor_db=floor_db,
        )

    axes[1].set_ylabel("Power / fundamental (dBc/bin)", fontsize=12.5)
    axes[-1].set_xlabel("Frequency (MHz)", fontsize=12.5)
    axes[-1].set_xticks([0, 100, 200, 300, 400, 500])
    for ax in axes:
        ax.tick_params(labelsize=11.0)

    handles = [
        plt.Line2D([0], [0], color=GRAY, lw=1.2, label="Spectrum"),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            mfc="white",
            mec=FUND,
            mew=1.1,
            color="none",
            label="Fundamental",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            mfc="white",
            mec=RED,
            mew=1.1,
            color="none",
            label="Largest spur",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.005),
        handlelength=1.5,
        columnspacing=1.5,
        fontsize=11.5,
    )
    fig.subplots_adjust(left=0.12, right=0.985, top=0.935, bottom=0.085, hspace=0.16)
    return fig


def save_figure(fig: plt.Figure, base: Path) -> None:
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(
        base.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    _, metrics, spectra = build_behavioral_case()
    write_source_data(
        metrics,
        spectra,
        out_dir / "fig4_behavioral_spectrum_data.csv",
        out_dir / "fig4_behavioral_spectrum_metrics.csv",
    )
    fig = build_figure(metrics, spectra)
    save_figure(fig, out_dir / "fig4_behavioral_spectrum")
    plt.close(fig)


if __name__ == "__main__":
    main()
