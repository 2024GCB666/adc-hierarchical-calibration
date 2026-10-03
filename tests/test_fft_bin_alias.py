"""FFT regressions independent of the removed browser application."""
import math
import numpy as np
from fft_output_code import alias_frequency_to_rfft, compute_fft_metrics


def test_above_nyquist_frequency_folds_into_the_positive_fft_band() -> None:
    fs = 1e9
    assert math.isclose(alias_frequency_to_rfft(65000 / 65536 * fs, fs), 536 / 65536 * fs)


def test_fft_metrics_use_alias_for_above_nyquist_input_bin() -> None:
    count = 65536
    n = np.arange(count)
    fs = 1e9
    values = np.sin(2 * np.pi * 65000 / count * n) + 0.05 * np.sin(2 * np.pi * 32232 / count * n)
    metrics, _ = compute_fft_metrics(values, fs=fs, fin=65000 / count * fs, window="auto")
    assert metrics["fundamental_bin"] == 536
    assert metrics["largest_spur_bin"] == 32232
    assert metrics["fin_requested_Hz"] > fs / 2
    assert math.isclose(metrics["fin_analysis_Hz"], 536 / count * fs)


def test_fft_metrics_count_bin_one_spur_near_nyquist() -> None:
    count = 65536
    n = np.arange(count)
    values = np.sin(2 * np.pi * 32767 / count * n) + 0.05 * np.sin(2 * np.pi / count * n)
    metrics, _ = compute_fft_metrics(values, fs=1e9, fin=32767 / count * 1e9, window="auto")
    assert metrics["fundamental_bin"] == 32767
    assert metrics["largest_spur_bin"] == 1
