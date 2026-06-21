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
from model import (  # noqa: E402
    DatasetConfig,
    MismatchConfig,
    SignalConfig,
    TimingConfig,
    build_synthetic_adc_dataset,
)


# Keep the visual language consistent with the existing IEEE paper figures.
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
    """Same static fine-path mismatch case used in Table II and Fig. 4."""

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


def build_behavioral_dataset() -> tuple[dict[str, np.ndarray], dict[str, float]]:
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
        mismatch=static_mismatch_case(),
    )
    data, truth, _ = build_synthetic_adc_dataset(cfg)
    return data, truth


def run_ls_reference(
    data: dict[str, np.ndarray],
    truth: dict[str, float],
) -> tuple[dict[str, float], dict[str, object]]:
    cfg = CalibrationConfig(
        reference_method="known_tone",
        known_tone_source="A",
        inter_method="two_stage",
    )
    _, coeffs, _, metrics = calibrate_adc(
        data["D_coarse"],
        data["F_raw"],
        sar_id=data["sar_id"],
        C_raw=data["C_raw"],
        fs=float(data["fs"]),
        fin=float(data["fin"]),
        D_ideal=data["D_ideal"],
        truth=truth,
        config=cfg,
    )
    return coeffs, metrics


def run_nlms_trace(
    data: dict[str, np.ndarray],
    truth: dict[str, float],
) -> tuple[dict[str, float], dict[str, object], dict[str, np.ndarray]]:
    cfg = CalibrationConfig(
        reference_method="known_tone",
        known_tone_source="A",
        inter_method="nlms",
        lms_mu=0.005,
        lms_epochs=1,
        lms_trace_block_size=256,
        lms_settling_ratio=1.1,
    )
    _, coeffs, debug, metrics = calibrate_adc(
        data["D_coarse"],
        data["F_raw"],
        sar_id=data["sar_id"],
        C_raw=data["C_raw"],
        fs=float(data["fs"]),
        fin=float(data["fin"]),
        D_ideal=data["D_ideal"],
        truth=truth,
        config=cfg,
    )
    return coeffs, metrics, debug


def rms(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(v * v)))


def build_trace_rows(
    nlms_metrics: dict[str, object],
    nlms_debug: dict[str, np.ndarray],
) -> list[dict[str, float]]:
    diagnostics = nlms_metrics["inter"]["diagnostics"]
    block_trace = diagnostics["lms_block_trace"]

    f_b = nlms_debug["F_intra_corr"][1::2].astype(np.float64)
    fhat_b = nlms_debug["Fhat_B"].astype(np.float64)
    mask_safe = nlms_debug["mask_safe"].astype(bool)
    initial_rms = rms((fhat_b - f_b)[mask_safe])

    rows: list[dict[str, float]] = [
        {
            "update_count": 0.0,
            "rms_res_B_lsb": initial_rms,
            "g_B": 1.0,
            "o_B_lsb": 0.0,
            "k_B": 0.0,
            "gain_correction_percent": 0.0,
        }
    ]
    for row in block_trace:
        g_b = float(row["g_B"])
        rows.append(
            {
                "update_count": float(row["update_count"]),
                "rms_res_B_lsb": float(row["rms_res_B"]),
                "g_B": g_b,
                "o_B_lsb": float(row["o_B"]),
                "k_B": float(row["k_B"]),
                "gain_correction_percent": 100.0 * (g_b - 1.0),
            }
        )
    return rows


def write_source_data(
    trace_rows: list[dict[str, float]],
    summary_rows: list[dict[str, float | str]],
    trace_path: Path,
    summary_path: Path,
) -> None:
    with trace_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "update_count",
            "rms_res_B_lsb",
            "g_B",
            "o_B_lsb",
            "k_B",
            "gain_correction_percent",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in trace_rows:
            writer.writerow({key: f"{float(row[key]):.9g}" for key in fieldnames})

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["metric", "value"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)


def build_summary_rows(
    truth: dict[str, float],
    ls_coeffs: dict[str, float],
    ls_metrics: dict[str, object],
    nlms_coeffs: dict[str, float],
    nlms_metrics: dict[str, object],
    trace_rows: list[dict[str, float]],
) -> list[dict[str, float | str]]:
    nlms_diag = nlms_metrics["inter"]["diagnostics"]
    truth_compare = nlms_metrics["truth_compare"]
    return [
        {"metric": "N", "value": int(truth["N"])},
        {"metric": "fs_Hz", "value": float(truth["signal"]["fs"])},
        {"metric": "fin_Hz", "value": float(truth["signal"]["fin"])},
        {"metric": "lms_mu", "value": nlms_diag["lms_mu"]},
        {"metric": "lms_trace_block_size", "value": nlms_diag["lms_trace_block_size"]},
        {"metric": "safe_B_samples", "value": nlms_diag["lms_safe_sample_count"]},
        {"metric": "convergence_updates_1p1x_final", "value": nlms_diag["convergence_samples"]},
        {"metric": "initial_reference_rms_lsb", "value": trace_rows[0]["rms_res_B_lsb"]},
        {"metric": "final_nlms_reference_rms_lsb", "value": nlms_diag["rms_res_B"]},
        {
            "metric": "ls_reference_rms_lsb",
            "value": ls_metrics["inter"]["diagnostics"]["rms_res_B"],
        },
        {"metric": "true_g_B", "value": truth_compare["g_B"]["true"]},
        {"metric": "nlms_g_B", "value": nlms_coeffs["g_B"]},
        {"metric": "ls_g_B", "value": ls_coeffs["g_B"]},
        {"metric": "true_o_B_lsb", "value": truth_compare["o_B"]["true"]},
        {"metric": "nlms_o_B_lsb", "value": nlms_coeffs["o_B"]},
        {"metric": "ls_o_B_lsb", "value": ls_coeffs["o_B"]},
        {"metric": "nlms_k_B", "value": nlms_coeffs["k_B"]},
        {
            "metric": "rms_after_full_nlms_lsb",
            "value": nlms_metrics["error_vs_ideal"]["rms_after_full_lsb"],
        },
        {
            "metric": "rms_after_full_ls_lsb",
            "value": ls_metrics["error_vs_ideal"]["rms_after_full_lsb"],
        },
    ]


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


def build_figure(
    trace_rows: list[dict[str, float]],
    ls_metrics: dict[str, object],
    nlms_metrics: dict[str, object],
) -> plt.Figure:
    setup_style()
    updates = np.asarray([row["update_count"] for row in trace_rows], dtype=np.float64)
    rms_trace = np.asarray([row["rms_res_B_lsb"] for row in trace_rows], dtype=np.float64)
    gain_percent = np.asarray(
        [row["gain_correction_percent"] for row in trace_rows],
        dtype=np.float64,
    )
    offset_lsb = np.asarray([row["o_B_lsb"] for row in trace_rows], dtype=np.float64)

    truth_compare = nlms_metrics["truth_compare"]
    true_gain_percent = 100.0 * (float(truth_compare["g_B"]["true"]) - 1.0)
    true_offset = float(truth_compare["o_B"]["true"])
    final_rms = float(nlms_metrics["inter"]["diagnostics"]["rms_res_B"])
    ls_rms = float(ls_metrics["inter"]["diagnostics"]["rms_res_B"])
    convergence_updates = nlms_metrics["inter"]["diagnostics"]["convergence_samples"]

    fig, axes = plt.subplots(2, 1, figsize=(7.05, 4.8), sharex=True)

    axes[0].plot(
        updates,
        rms_trace,
        color=BLUE,
        lw=1.8,
        marker="o",
        ms=3.0,
        mfc=BLUE_SOFT,
        mec=BLUE,
        label="NLMS trace",
    )
    axes[0].axhline(
        ls_rms,
        color=GREEN,
        lw=1.2,
        ls="--",
        label="LS residual floor",
    )
    axes[0].set_ylabel("B-reference RMS (LSB)")
    axes[0].set_ylim(0.0, max(2.35, float(np.max(rms_trace)) * 1.08))
    axes[0].grid(axis="y", color=LIGHT_GRAY, lw=0.6)
    axes[0].legend(loc="upper right", fontsize=10.4, handlelength=1.7)
    if convergence_updates is not None:
        axes[0].annotate(
            "within 10% of final\nat 2048 updates",
            xy=(float(convergence_updates), final_rms * 1.1),
            xytext=(7600, 1.05),
            textcoords="data",
            fontsize=10.0,
            color=BLUE,
            ha="left",
            arrowprops={
                "arrowstyle": "->",
                "lw": 0.75,
                "color": BLUE,
                "shrinkA": 1,
                "shrinkB": 2,
            },
        )

    axes[1].plot(
        updates,
        gain_percent,
        color=GREEN,
        lw=1.8,
        marker="o",
        ms=3.0,
        mfc=GREEN_SOFT,
        mec=GREEN,
        label=r"$100(g_B-1)$",
    )
    axes[1].plot(
        updates,
        offset_lsb,
        color=RED,
        lw=1.65,
        marker="s",
        ms=2.8,
        mfc="#E6AAA9",
        mec=RED,
        label=r"$o_B$",
    )
    axes[1].axhline(true_gain_percent, color=GREEN, lw=1.0, ls="--")
    axes[1].axhline(true_offset, color=RED, lw=1.0, ls=":")
    axes[1].set_xlabel("Safe B-group coefficient updates")
    axes[1].set_ylabel("Gain correction (%) / offset (LSB)")
    axes[1].set_xlim(0, max(27000, float(np.max(updates)) * 1.02))
    axes[1].set_ylim(-0.2, 2.45)
    axes[1].grid(axis="y", color=LIGHT_GRAY, lw=0.6)
    axes[1].legend(loc="lower right", ncol=2, fontsize=10.4, handlelength=1.7)
    axes[1].text(
        0.02,
        0.96,
        "truth: dashed gain, dotted offset",
        transform=axes[1].transAxes,
        fontsize=9.8,
        color=GRAY,
        va="top",
    )

    add_panel_label(axes[0], "a")
    add_panel_label(axes[1], "b")
    for ax in axes:
        ax.tick_params(labelsize=10.6)
        ax.xaxis.label.set_size(11.6)
        ax.yaxis.label.set_size(11.6)
    fig.subplots_adjust(left=0.11, right=0.985, top=0.97, bottom=0.12, hspace=0.18)
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
    data, truth = build_behavioral_dataset()
    ls_coeffs, ls_metrics = run_ls_reference(data, truth)
    nlms_coeffs, nlms_metrics, nlms_debug = run_nlms_trace(data, truth)
    trace_rows = build_trace_rows(nlms_metrics, nlms_debug)
    summary_rows = build_summary_rows(
        truth,
        ls_coeffs,
        ls_metrics,
        nlms_coeffs,
        nlms_metrics,
        trace_rows,
    )
    write_source_data(
        trace_rows,
        summary_rows,
        out_dir / "fig6_nlms_convergence_data.csv",
        out_dir / "fig6_nlms_convergence_summary.csv",
    )
    fig = build_figure(trace_rows, ls_metrics, nlms_metrics)
    save_figure(fig, out_dir / "fig6_nlms_convergence")
    plt.close(fig)


if __name__ == "__main__":
    main()
