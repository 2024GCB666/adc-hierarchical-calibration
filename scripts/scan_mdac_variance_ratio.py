from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

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


def ideal_fine_path_mismatch() -> MismatchConfig:
    """Keep fine-path gain/offset/noise ideal to isolate timing-induced variance."""

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


def simulate_one_point(
    *,
    mdac_b_skew_ps: float,
    mdac_a_skew_ps: float,
    n_samples: int,
    seed: int,
    fs: float,
    fin: float,
    amplitude: float,
    phase: float,
) -> dict[str, float]:
    cfg = DatasetConfig(
        N=n_samples,
        seed=seed,
        signal=SignalConfig(
            fs=fs,
            fin=fin,
            amplitude=amplitude,
            phase=phase,
            dc=0.0,
        ),
        timing=TimingConfig(
            dt_mdac_A=mdac_a_skew_ps * 1.0e-12,
            dt_mdac_B=mdac_b_skew_ps * 1.0e-12,
        ),
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
        "mdac_a_skew_ps": float(mdac_a_skew_ps),
        "mdac_b_skew_ps": float(mdac_b_skew_ps),
        "var_F_A_lsb2": var_a,
        "var_F_B_lsb2": var_b,
        "var_ratio_B_over_A": ratio,
        "rms_ratio_B_over_A": float(np.sqrt(ratio)),
        "mean_F_A_lsb": float(np.mean(fine_a)),
        "mean_F_B_lsb": float(np.mean(fine_b)),
        "clipped_fraction_A": float(np.mean(data["F_raw_clipped"][mask_a])),
        "clipped_fraction_B": float(np.mean(data["F_raw_clipped"][mask_b])),
    }


def build_sweep(args: argparse.Namespace) -> list[dict[str, float]]:
    skew_values = np.arange(args.start_ps, args.stop_ps + 0.5 * args.step_ps, args.step_ps)
    return [
        simulate_one_point(
            mdac_b_skew_ps=float(skew_ps),
            mdac_a_skew_ps=args.mdac_a_skew_ps,
            n_samples=args.samples,
            seed=args.seed,
            fs=args.fs,
            fin=args.fin,
            amplitude=args.amplitude,
            phase=args.phase,
        )
        for skew_ps in skew_values
    ]


def write_csv(rows: list[dict[str, float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
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
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep MDAC-B timing skew and compute Var(F_B)/Var(F_A) with ideal "
            "fine-path gain/offset/noise."
        )
    )
    parser.add_argument("--start-ps", type=float, default=0.0, help="Sweep start for MDAC-B skew in ps.")
    parser.add_argument("--stop-ps", type=float, default=40.0, help="Sweep stop for MDAC-B skew in ps.")
    parser.add_argument("--step-ps", type=float, default=2.0, help="Sweep step for MDAC-B skew in ps.")
    parser.add_argument("--mdac-a-skew-ps", type=float, default=0.0, help="Reference MDAC-A skew in ps.")
    parser.add_argument("--samples", type=int, default=2**16, help="Number of ADC samples; must be divisible by 4.")
    parser.add_argument("--seed", type=int, default=20260620, help="Random seed used by the model.")
    parser.add_argument("--fs", type=float, default=1.0e9, help="Sampling frequency in Hz.")
    parser.add_argument("--fin", type=float, default=499.0e6, help="Input sine frequency in Hz.")
    parser.add_argument("--amplitude", type=float, default=511.0, help="Input sine amplitude in final-output LSB.")
    parser.add_argument("--phase", type=float, default=0.31, help="Input sine phase in rad.")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "figures" / "fig2_mdac_variance_ratio_data.csv",
        help="CSV output path.",
    )
    args = parser.parse_args()
    if args.step_ps <= 0:
        raise ValueError("--step-ps must be positive.")
    if args.samples % 4 != 0:
        raise ValueError("--samples must be divisible by 4.")
    return args


def main() -> None:
    args = parse_args()
    rows = build_sweep(args)
    write_csv(rows, args.output)

    final = rows[-1]
    print(f"Wrote {len(rows)} sweep points to {args.output}")
    print(
        "Final point: "
        f"MDAC-B skew={final['mdac_b_skew_ps']:.2f} ps, "
        f"Var(F_B)/Var(F_A)={final['var_ratio_B_over_A']:.6f}, "
        f"RMS ratio={final['rms_ratio_B_over_A']:.6f}"
    )


if __name__ == "__main__":
    main()
