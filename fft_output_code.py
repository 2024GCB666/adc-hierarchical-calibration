"""FFT metrics and spectrum plot for ADC output-code data.

Examples:
    python fft_output_code.py outputs/adc_synthetic_modeling_data_mdac4_30ps_50ps_fixed.npz
    python fft_output_code.py outputs/adc_synthetic_modeling_data_mdac4_30ps_50ps_fixed.csv --column D_raw --fs 1e9 --fin 73.123e6
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, IO, Literal

import numpy as np

WindowMode = Literal["auto", "blackman_harris", "rectangular"]


def blackman_harris(N: int) -> np.ndarray:
    n = np.arange(N, dtype=np.float64)
    a0, a1, a2, a3 = 0.35875, 0.48829, 0.14128, 0.01168
    return (
        a0
        - a1 * np.cos(2.0 * np.pi * n / (N - 1))
        + a2 * np.cos(4.0 * np.pi * n / (N - 1))
        - a3 * np.cos(6.0 * np.pi * n / (N - 1))
    )


def is_coherent_tone(fs: float, fin: float | None, N: int, tol: float = 1e-9) -> bool:
    if fin is None or fs <= 0.0:
        return False
    cycles = float(fin) * float(N) / float(fs)
    return abs(cycles - round(cycles)) <= tol


def alias_frequency_to_rfft(fin: float, fs: float) -> float:
    """Return the positive-frequency rFFT alias for a sampled real tone."""

    if fs <= 0.0:
        raise ValueError("fs must be positive.")
    folded = float(fin) % float(fs)
    if folded > 0.5 * float(fs):
        folded = float(fs) - folded
    return float(folded)


def select_window(N: int, fs: float, fin: float | None, window: WindowMode) -> tuple[np.ndarray, str]:
    if window == "auto":
        window = "rectangular" if is_coherent_tone(fs, fin, N) else "blackman_harris"

    if window == "rectangular":
        return np.ones(N, dtype=np.float64), "Rectangular"
    if window == "blackman_harris":
        return blackman_harris(N), "4-term Blackman-Harris"
    raise ValueError(f"Unsupported window mode: {window!r}")


def load_output_code(
    path: Path,
    column: str,
    fs_arg: float | None,
    fin_arg: float | None,
) -> tuple[np.ndarray, float, float | None]:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        data = np.load(path)
        if column not in data.files:
            raise KeyError(f"Column {column!r} not found. Available: {data.files}")
        y = data[column].astype(np.float64)
        fs = float(fs_arg if fs_arg is not None else data["fs"])
        fin = float(fin_arg if fin_arg is not None else data["fin"]) if (
            fin_arg is not None or "fin" in data.files
        ) else None
        return y, fs, fin

    if suffix == ".csv":
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or column not in reader.fieldnames:
                raise KeyError(
                    f"Column {column!r} not found. Available: {reader.fieldnames}"
                )
            values = [float(row[column]) for row in reader]
        if fs_arg is None:
            raise ValueError("CSV input has no metadata; pass --fs.")
        return np.asarray(values, dtype=np.float64), float(fs_arg), fin_arg

    raise ValueError("Input must be .npz or .csv.")


def contiguous_bins(center: int, half_width: int, length: int) -> np.ndarray:
    return np.arange(max(1, center - half_width), min(length, center + half_width + 1))


def compute_fft_metrics(
    y: np.ndarray,
    fs: float,
    fin: float | None = None,
    window: WindowMode = "blackman_harris",
    signal_half_width: int = 4,
    guard_half_width: int = 8,
    spur_half_width: int = 2,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    y = np.asarray(y, dtype=np.float64)
    N = y.size
    if N < 16:
        raise ValueError("Need at least 16 samples for FFT metrics.")

    y0 = y - np.mean(y)
    window_values, window_name = select_window(N, fs, fin, window)
    Y = np.fft.rfft(y0 * window_values)
    power = np.abs(Y) ** 2
    freq = np.fft.rfftfreq(N, 1.0 / fs)

    power[0] = 0.0

    if fin is None:
        fund_bin = int(np.argmax(power))
        fin_used = float(freq[fund_bin])
        fin_analysis = fin_used
        input_cycles = None
    else:
        fin_analysis = alias_frequency_to_rfft(float(fin), fs)
        fund_bin = int(np.argmin(np.abs(freq - fin_analysis)))
        fin_used = float(fin_analysis)
        input_cycles = float(fin) * float(N) / float(fs)

    cycles = float(fin_used) * float(N) / float(fs)
    cycle_error = cycles - round(cycles)
    signal_bins = contiguous_bins(fund_bin, signal_half_width, power.size)
    signal_power = float(np.sum(power[signal_bins]))
    if signal_power <= 0.0:
        raise ValueError("Fundamental signal power is zero.")

    noise_mask = np.ones(power.size, dtype=bool)
    noise_mask[0] = False
    guard_bins = contiguous_bins(fund_bin, guard_half_width, power.size)
    noise_mask[guard_bins] = False

    noise_dist_power = float(np.sum(power[noise_mask]))
    sndr_db = 10.0 * math.log10(signal_power / noise_dist_power)
    enob_bits = (sndr_db - 1.76) / 6.02

    spur_search = np.where(noise_mask, power, 0.0)
    spur_bin = int(np.argmax(spur_search))
    spur_power_single = float(power[spur_bin])
    spur_bins = contiguous_bins(spur_bin, spur_half_width, power.size)
    spur_bins = spur_bins[noise_mask[spur_bins]]
    spur_power_integrated = float(np.sum(power[spur_bins]))

    sfdr_single_db = 10.0 * math.log10(signal_power / spur_power_single)
    sfdr_integrated_db = 10.0 * math.log10(signal_power / spur_power_integrated)

    bin_power_dbc = 10.0 * np.log10(np.maximum(power, 1e-300) / signal_power)
    metrics = {
        "N": int(N),
        "fs_Hz": float(fs),
        "fin_requested_Hz": None if fin is None else float(fin),
        "fin_analysis_Hz": float(fin_analysis),
        "fin_used_Hz": fin_used,
        "fundamental_bin": int(fund_bin),
        "fundamental_bin_freq_Hz": float(freq[fund_bin]),
        "bin_width_Hz": float(fs / N),
        "window": window_name,
        "coherent_sampling": bool(abs(cycle_error) <= 1e-9),
        "input_cycles_in_record": None if input_cycles is None else float(input_cycles),
        "cycles_in_record": cycles,
        "cycles_fractional_error": cycle_error,
        "signal_bin_range": [int(signal_bins[0]), int(signal_bins[-1])],
        "guard_half_width_bins": int(guard_half_width),
        "SNDR_dB": float(sndr_db),
        "SNFR_dB": float(sndr_db),
        "SFDR_dBc_single_bin": float(sfdr_single_db),
        "SFDR_dBc_integrated": float(sfdr_integrated_db),
        "ENOB_bits": float(enob_bits),
        "largest_spur_bin": int(spur_bin),
        "largest_spur_freq_Hz": float(freq[spur_bin]),
        "largest_spur_dBc_single_bin": float(
            10.0 * math.log10(spur_power_single / signal_power)
        ),
        "largest_spur_dBc_integrated": float(
            10.0 * math.log10(spur_power_integrated / signal_power)
        ),
    }
    debug = {
        "freq": freq,
        "power": power,
        "bin_power_dbc": bin_power_dbc,
        "noise_mask": noise_mask,
        "signal_bins": signal_bins,
    }
    return metrics, debug


def plot_spectrum(
    debug: dict[str, np.ndarray],
    metrics: dict[str, Any],
    output_path: Path | IO[bytes],
    title: str,
    floor_db: float = -160.0,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    freq_mhz = debug["freq"] / 1e6
    spectrum_dbc = debug["bin_power_dbc"]
    
    # Adaptive lower threshold calculation
    min_dbc = np.min(spectrum_dbc)
    p05_dbc = np.percentile(spectrum_dbc, 5)
    max_dbc = np.max(spectrum_dbc)
    
    # Base the floor on the 5th percentile to ignore deep numerical nulls
    adaptive_floor = np.floor((p05_dbc - 10) / 10) * 10
    # Ensure minimum dynamic range of 100 dB, but don't clip below -200 for ideal ADCs
    adaptive_floor = min(adaptive_floor, np.floor((max_dbc - 100) / 10) * 10)
    adaptive_floor = max(adaptive_floor, -200.0)
    
    # Overwrite floor_db with the adaptive one
    floor_db = adaptive_floor
    spectrum_display = np.maximum(spectrum_dbc, floor_db)

    plt.figure(figsize=(11, 6))
    plt.vlines(
        freq_mhz,
        floor_db,
        spectrum_display,
        linewidth=0.8,
        color="tab:blue",
        alpha=0.85,
    )
    # Add dots so every bin is explicitly visible (e.g. at the bottom axis if clipped)
    plt.scatter(
        freq_mhz,
        spectrum_display,
        s=4,
        color="tab:blue",
        alpha=0.85,
        zorder=3,
        edgecolors="none",
    )
    fund_idx = metrics["fundamental_bin"]
    spur_idx = metrics["largest_spur_bin"]
    plt.plot(
        freq_mhz[fund_idx],
        spectrum_dbc[fund_idx],
        marker="o",
        markersize=10,
        markeredgecolor="tab:green",
        markerfacecolor="none",
        linestyle="none",
        label="Fundamental",
    )
    plt.plot(
        freq_mhz[spur_idx],
        spectrum_dbc[spur_idx],
        marker="o",
        markersize=10,
        markeredgecolor="tab:red",
        markerfacecolor="none",
        linestyle="none",
        label="Largest spur",
    )
    plt.title(title)
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Power / Fundamental (dBc/bin)")
    plt.grid(True, alpha=0.35)
    plt.ylim(floor_db, 10)
    plt.xlim(0, float(freq_mhz[-1]))
    plt.legend(loc="upper right")
    text = (
        f"SNDR/SNFR = {metrics['SNDR_dB']:.2f} dB\n"
        f"SFDR = {metrics['SFDR_dBc_single_bin']:.2f} dBc\n"
        f"ENOB = {metrics['ENOB_bits']:.2f} bit\n"
        f"{metrics['window']}"
    )
    plt.text(
        0.985,
        0.82,
        text,
        transform=plt.gca().transAxes,
        ha="right",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "lightgray", "alpha": 0.85},
    )
    plt.tight_layout()
    if hasattr(output_path, "parent"):
        output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()


def json_ready(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_ready(v) for v in obj]
    return obj


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FFT analysis for ADC output code.")
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=None,
        help=(
            "Input .npz or .csv file. If omitted, analyze the default "
            "nomismatch and mismatch NPZ files from model.py."
        ),
    )
    parser.add_argument("--column", default="D_raw", help="Output code column/key.")
    parser.add_argument("--fs", type=float, default=None, help="Sample rate in Hz.")
    parser.add_argument("--fin", type=float, default=None, help="Tone frequency in Hz.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "fft",
        help="Directory for PNG and JSON outputs.",
    )
    parser.add_argument(
        "--window",
        choices=["auto", "blackman_harris", "rectangular"],
        default="blackman_harris",
        help=(
            "FFT window. 'auto' uses rectangular for coherent tones and "
            "Blackman-Harris otherwise."
        ),
    )
    parser.add_argument("--signal-half-width", type=int, default=4)
    parser.add_argument("--guard-half-width", type=int, default=8)
    parser.add_argument("--spur-half-width", type=int, default=2)
    return parser.parse_args()


def analyze_one(args: argparse.Namespace, input_path: Path) -> dict[str, Any]:
    y, fs, fin = load_output_code(input_path, args.column, args.fs, args.fin)
    metrics, debug = compute_fft_metrics(
        y,
        fs=fs,
        fin=fin,
        window=args.window,
        signal_half_width=args.signal_half_width,
        guard_half_width=args.guard_half_width,
        spur_half_width=args.spur_half_width,
    )

    stem = f"{input_path.stem}_{args.column}_fft"
    png_path = args.output_dir / f"{stem}.png"
    json_path = args.output_dir / f"{stem}_metrics.json"
    plot_spectrum(
        debug,
        metrics,
        png_path,
        title=f"FFT Spectrum: {input_path.name} [{args.column}]",
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(json_ready(metrics), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Input: {input_path}")
    print(f"Column: {args.column}")
    print(f"Spectrum PNG: {png_path}")
    print(f"Metrics JSON: {json_path}")
    print(f"SNDR/SNFR: {metrics['SNDR_dB']:.3f} dB")
    print(f"SFDR single-bin: {metrics['SFDR_dBc_single_bin']:.3f} dBc")
    print(f"SFDR integrated: {metrics['SFDR_dBc_integrated']:.3f} dBc")
    print(f"ENOB: {metrics['ENOB_bits']:.3f} bits")
    print(f"Largest spur: {metrics['largest_spur_freq_Hz'] / 1e6:.6f} MHz")
    return {
        "input": str(input_path),
        "png": str(png_path),
        "json": str(json_path),
        "metrics": metrics,
    }


def main() -> None:
    args = parse_args()
    if args.input is None:
        inputs = [
            Path("outputs") / "adc_synthetic_modeling_data_nomismatch.npz",
            Path("outputs") / "adc_synthetic_modeling_data_mismatch.npz",
        ]
    else:
        inputs = [args.input]

    results = []
    for index, input_path in enumerate(inputs):
        if not input_path.exists():
            raise FileNotFoundError(
                f"{input_path} does not exist. Run model.py first or pass an input file."
            )
        if index:
            print("")
        results.append(analyze_one(args, input_path))

    if len(results) > 1:
        summary_path = args.output_dir / f"default_{args.column}_fft_compare_metrics.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(json_ready(results), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print("")
        print(f"Comparison JSON: {summary_path}")


if __name__ == "__main__":
    main()
