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
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Times New Roman"
plt.rcParams["mathtext.it"] = "Times New Roman:italic"
plt.rcParams["mathtext.bf"] = "Times New Roman:bold"


BLACK = "#111111"
GRAY = "#666666"
LIGHT_GRAY = "#E6E6E6"
BLUE = "#0F4D92"
BLUE_SOFT = "#7EA6D8"
RED = "#B64342"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 7.2,
            "axes.linewidth": 0.8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "savefig.facecolor": "white",
        }
    )


def ideal_fine_path_mismatch() -> MismatchConfig:
    """No gain/offset/noise mismatch: isolate timing-induced residue statistics."""

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


def simulate_variance_ratio(
    skew_ps: float,
    *,
    n_samples: int = 2**16,
    seed: int = 20260620,
) -> dict[str, float]:
    """Return A/B backend fine-code variance statistics for one MDAC-B skew."""

    cfg = DatasetConfig(
        N=n_samples,
        seed=seed,
        signal=SignalConfig(
            fs=1.0e9,
            fin=499.0e6,
            amplitude=511.0,
            phase=0.31,
            dc=0.0,
        ),
        # MDAC-A is used as the reference.  MDAC-B is swept relative to the
        # common Flash/coarse decision instant.
        timing=TimingConfig(dt_mdac_A=0.0, dt_mdac_B=skew_ps * 1.0e-12),
        mismatch=ideal_fine_path_mismatch(),
    )
    data, _, _ = build_synthetic_adc_dataset(cfg)

    fine_code = data["F_raw"].astype(np.float64)
    mdac_id = data["mdac_id"]
    mask_a = mdac_id == 0
    mask_b = mdac_id == 1

    fine_a = fine_code[mask_a]
    fine_b = fine_code[mask_b]
    var_a = float(np.var(fine_a, ddof=1))
    var_b = float(np.var(fine_b, ddof=1))
    ratio = var_b / var_a

    return {
        "mdac_a_skew_ps": 0.0,
        "mdac_b_skew_ps": float(skew_ps),
        "var_F_A_lsb2": var_a,
        "var_F_B_lsb2": var_b,
        "var_ratio_B_over_A": ratio,
        "rms_ratio_B_over_A": float(np.sqrt(ratio)),
        "mean_F_A_lsb": float(np.mean(fine_a)),
        "mean_F_B_lsb": float(np.mean(fine_b)),
        "clipped_fraction_A": float(np.mean(data["F_raw_clipped"][mask_a])),
        "clipped_fraction_B": float(np.mean(data["F_raw_clipped"][mask_b])),
    }


def build_sweep() -> list[dict[str, float]]:
    sweep_ps = np.arange(0.0, 40.0 + 1.0e-9, 2.0)
    return [simulate_variance_ratio(float(skew_ps)) for skew_ps in sweep_ps]


def write_source_data(rows: list[dict[str, float]], path: Path) -> None:
    fieldnames = [
        "mdac_a_skew_ps",
        "mdac_b_skew_ps",
        "var_F_A_lsb2",
        "var_F_B_lsb2",
        "var_ratio_B_over_A",
        "rms_ratio_B_over_A",
        "mean_F_A_lsb",
        "mean_F_B_lsb",
        "clipped_fraction_A",
        "clipped_fraction_B",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_figure(rows: list[dict[str, float]]) -> plt.Figure:
    setup_style()
    skew_ps = np.array([row["mdac_b_skew_ps"] for row in rows])
    var_a = np.array([row["var_F_A_lsb2"] for row in rows])
    var_b = np.array([row["var_F_B_lsb2"] for row in rows])
    ratio = np.array([row["var_ratio_B_over_A"] for row in rows])

    fig, ax = plt.subplots(1, 1, figsize=(3.65, 2.55))

    ax.plot(
        skew_ps,
        var_a,
        color=GRAY,
        lw=1.5,
        marker="o",
        ms=3.3,
        mfc="white",
        mec=GRAY,
        label="MDAC-A reference",
    )
    ax.plot(
        skew_ps,
        var_b,
        color=BLUE,
        lw=1.8,
        marker="o",
        ms=3.5,
        mfc=BLUE_SOFT,
        mec=BLUE,
        label="MDAC-B swept",
    )
    ax.set_xlabel("MDAC-B timing skew (ps)")
    ax.set_ylabel(r"Fine-code variance (LSB$^2$)")
    ax.set_xlim(-1, 41)
    ax.set_xticks([0, 10, 20, 30, 40])
    ax.set_ylim(4300, 6600)
    ax.grid(axis="y", color=LIGHT_GRAY, lw=0.6)
    ax.legend(loc="upper left", handlelength=1.5, borderaxespad=0.2)
    ax.annotate(
        f"{ratio[-1]:.2f}x at 40 ps",
        xy=(skew_ps[-1], var_b[-1]),
        xytext=(22.5, 6350),
        textcoords="data",
        fontsize=6.7,
        color=RED,
        arrowprops={
            "arrowstyle": "->",
            "lw": 0.8,
            "color": RED,
            "shrinkA": 1,
            "shrinkB": 2,
        },
    )
    fig.subplots_adjust(left=0.17, right=0.98, top=0.97, bottom=0.22)
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
    write_source_data(rows, out_dir / "fig2_mdac_variance_ratio_data.csv")
    fig = build_figure(rows)
    save_figure(fig, out_dir / "fig2_mdac_variance_ratio")
    plt.close(fig)


if __name__ == "__main__":
    main()
