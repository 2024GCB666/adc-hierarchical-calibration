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

from calibration import (  # noqa: E402
    CalibrationConfig,
    add_fft_metrics,
    calibrate_adc,
)
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
BLUE_SOFT = "#7EA6D8"
GREEN = "#0B7A53"
GREEN_SOFT = "#93C9B1"
RED = "#B64342"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10.8,
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


def write_source_data(rows: list[dict[str, float]], path: Path) -> None:
    fieldnames = [
        "target_fin_mhz",
        "coherent_tone_bin",
        "fin_mhz",
        "raw_sndr_db",
        "intra_sndr_db",
        "full_sndr_db",
        "raw_sfdr_dbc",
        "intra_sfdr_dbc",
        "full_sfdr_dbc",
        "raw_enob_bit",
        "intra_enob_bit",
        "full_enob_bit",
        "raw_rms_lsb",
        "intra_rms_lsb",
        "full_rms_lsb",
        "g20",
        "g31",
        "g_B",
        "o_B_lsb",
        "k_B",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_metric_lines(
    ax: plt.Axes,
    x_mhz: np.ndarray,
    raw: np.ndarray,
    intra: np.ndarray,
    full: np.ndarray,
    *,
    label_lines: bool = False,
) -> None:
    ax.plot(
        x_mhz,
        raw,
        color=GRAY,
        lw=1.35,
        marker="o",
        ms=3.1,
        mfc="white",
        mec=GRAY,
        label="Raw" if label_lines else None,
    )
    ax.plot(
        x_mhz,
        intra,
        color=BLUE,
        lw=1.45,
        marker="s",
        ms=3.0,
        mfc=BLUE_SOFT,
        mec=BLUE,
        label="After intra-group" if label_lines else None,
    )
    ax.plot(
        x_mhz,
        full,
        color=GREEN,
        lw=1.65,
        marker="o",
        ms=3.2,
        mfc=GREEN_SOFT,
        mec=GREEN,
        label="After full calibration" if label_lines else None,
    )


def add_y_break_marks(ax_high: plt.Axes, ax_low: plt.Axes) -> None:
    """Draw compact diagonal marks to indicate the omitted y-axis interval."""

    d = 0.008
    kwargs = {"color": BLACK, "clip_on": False, "lw": 0.9}
    ax_high.plot((-d, +d), (-d, +d), transform=ax_high.transAxes, **kwargs)
    ax_low.plot((-d, +d), (1 - d, 1 + d), transform=ax_low.transAxes, **kwargs)


def plot_broken_metric(
    ax_high: plt.Axes,
    ax_low: plt.Axes,
    x_mhz: np.ndarray,
    raw: np.ndarray,
    intra: np.ndarray,
    full: np.ndarray,
    *,
    lower_ylim: tuple[float, float],
    upper_ylim: tuple[float, float],
    lower_ticks: list[float],
    upper_ticks: list[float],
    show_xlabel: bool = False,
    label_lines: bool = False,
) -> None:
    for ax in (ax_high, ax_low):
        plot_metric_lines(
            ax,
            x_mhz,
            raw,
            intra,
            full,
            label_lines=label_lines,
        )
        ax.set_xlim(0, 510)
        ax.set_xticks([0, 100, 200, 300, 400, 500])
        ax.grid(axis="y", color=LIGHT_GRAY, lw=0.6)

    ax_high.set_ylim(*upper_ylim)
    ax_low.set_ylim(*lower_ylim)
    ax_high.set_yticks(upper_ticks)
    ax_low.set_yticks(lower_ticks)

    ax_high.spines["bottom"].set_visible(False)
    ax_low.spines["top"].set_visible(False)
    ax_high.tick_params(labelbottom=False, bottom=False)
    if show_xlabel:
        ax_low.set_xlabel("Input frequency (MHz)")
    else:
        ax_low.tick_params(labelbottom=False)

    add_y_break_marks(ax_high, ax_low)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.03,
        label,
        transform=ax.transAxes,
        fontsize=12.2,
        fontweight="bold",
        va="bottom",
    )


def make_broken_axes_figure() -> tuple[plt.Figure, dict[str, plt.Axes]]:
    fig = plt.figure(figsize=(7.05, 5.1), constrained_layout=False)
    gs = fig.add_gridspec(
        5,
        1,
        height_ratios=[0.68, 1.0, 0.22, 0.68, 1.0],
        hspace=0.07,
    )
    ax_sndr_high = fig.add_subplot(gs[0])
    ax_sndr_low = fig.add_subplot(gs[1], sharex=ax_sndr_high)
    ax_spacer = fig.add_subplot(gs[2])
    ax_sfdr_high = fig.add_subplot(gs[3], sharex=ax_sndr_high)
    ax_sfdr_low = fig.add_subplot(gs[4], sharex=ax_sndr_high)
    ax_spacer.axis("off")
    return fig, {
        "sndr_high": ax_sndr_high,
        "sndr_low": ax_sndr_low,
        "sfdr_high": ax_sfdr_high,
        "sfdr_low": ax_sfdr_low,
    }


def build_figure(rows: list[dict[str, float]]) -> plt.Figure:
    setup_style()
    x_mhz = np.array([row["fin_mhz"] for row in rows])
    raw_sndr = np.array([row["raw_sndr_db"] for row in rows])
    intra_sndr = np.array([row["intra_sndr_db"] for row in rows])
    full_sndr = np.array([row["full_sndr_db"] for row in rows])
    raw_sfdr = np.array([row["raw_sfdr_dbc"] for row in rows])
    intra_sfdr = np.array([row["intra_sfdr_dbc"] for row in rows])
    full_sfdr = np.array([row["full_sfdr_dbc"] for row in rows])

    fig, axes = make_broken_axes_figure()
    plot_broken_metric(
        axes["sndr_high"],
        axes["sndr_low"],
        x_mhz,
        raw_sndr,
        intra_sndr,
        full_sndr,
        lower_ylim=(42.6, 48.4),
        upper_ylim=(60.9, 61.75),
        lower_ticks=[43, 45, 47],
        upper_ticks=[61.0, 61.4],
        show_xlabel=False,
        label_lines=True,
    )
    plot_broken_metric(
        axes["sfdr_high"],
        axes["sfdr_low"],
        x_mhz,
        raw_sfdr,
        intra_sfdr,
        full_sfdr,
        lower_ylim=(43.2, 49.1),
        upper_ylim=(89.0, 92.4),
        lower_ticks=[44, 46, 48],
        upper_ticks=[90, 91, 92],
        show_xlabel=True,
    )

    add_panel_label(axes["sndr_high"], "a")
    add_panel_label(axes["sfdr_high"], "b")

    axes["sndr_high"].annotate(
        "+18.3 dB",
        xy=(61.05, 61.37),
        xytext=(112, 61.62),
        textcoords="data",
        fontsize=10.6,
        color=GREEN,
        arrowprops={
            "arrowstyle": "->",
            "lw": 0.75,
            "color": GREEN,
            "shrinkA": 1,
            "shrinkB": 2,
        },
    )
    axes["sfdr_high"].annotate(
        "near-Nyquist\npoint retained",
        xy=(x_mhz[-1], full_sfdr[-1]),
        xytext=(335, 91.95),
        textcoords="data",
        fontsize=10.6,
        color=RED,
        ha="left",
        arrowprops={
            "arrowstyle": "->",
            "lw": 0.75,
            "color": RED,
            "shrinkA": 1,
            "shrinkB": 2,
        },
    )

    handles, labels = axes["sndr_high"].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.995),
        columnspacing=1.4,
        handlelength=1.7,
        fontsize=11.0,
    )
    fig.text(0.026, 0.705, "SNDR (dB)", rotation=90, va="center", fontsize=12.0)
    fig.text(0.026, 0.295, "SFDR (dBc)", rotation=90, va="center", fontsize=12.0)
    for ax in axes.values():
        ax.tick_params(labelsize=11.0)
        ax.xaxis.label.set_size(12.0)
    fig.subplots_adjust(left=0.105, right=0.985, top=0.905, bottom=0.095)
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
    rows = build_sweep()
    write_source_data(rows, out_dir / "fig5_frequency_sweep_data.csv")
    fig = build_figure(rows)
    save_figure(fig, out_dir / "fig5_frequency_sweep")
    plt.close(fig)


if __name__ == "__main__":
    main()
