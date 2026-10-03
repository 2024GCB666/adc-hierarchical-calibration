"""Numerical FFT, SNDR, SFDR and ENOB evaluation, without plotting.

The command-line entry point reads a user-supplied CSV or NPZ record and
prints metrics as JSON. No default dataset or generated spectrum is bundled.
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










def main() -> None:
    parser = argparse.ArgumentParser(description="Numerical ADC FFT metrics.")
    parser.add_argument("input", type=Path, help="User-supplied CSV or NPZ file.")
    parser.add_argument("--column", default="D_raw")
    parser.add_argument("--fs", type=float)
    parser.add_argument("--fin", type=float)
    parser.add_argument("--window", choices=("auto", "rectangular", "blackman_harris"), default="auto")
    args = parser.parse_args()
    values, fs, fin = load_output_code(args.input, args.column, args.fs, args.fin)
    metrics, _ = compute_fft_metrics(values, fs=fs, fin=fin, window=args.window)
    print(json.dumps(json_ready(metrics), indent=2))


if __name__ == "__main__":
    main()
