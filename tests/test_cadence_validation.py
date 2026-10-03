"""Tests for Cadence-exported ADC validation helpers."""

from __future__ import annotations

import numpy as np
import pytest

from calibration import CalibrationConfig
from scripts.validate_cadence_adc import (
    build_mdac_B_reference,
    build_mdac_group_streams,
)



@pytest.mark.parametrize("channel_offset", [0, 1, 2, 3])
def test_fractional_delay_reference_tracks_b_for_all_channel_rotations(
    channel_offset: int,
) -> None:
    sample_count = 512
    fs = 800.0e6
    fin = 20.0e6
    n = np.arange(sample_count)
    full_rate_code = 10.0 + 50.0 * np.sin(2.0 * np.pi * fin / fs * n)
    sar_id = (n + channel_offset) % 4
    group = build_mdac_group_streams(
        np.zeros(sample_count),
        full_rate_code,
        sar_id,
    )

    Dhat_B, _, _, mask, info = build_mdac_B_reference(
        group,
        fs=fs,
        fin=fin,
        config=CalibrationConfig(
            reference_method="fractional_delay_fir",
            fir_taps=15,
        ),
    )

    error = Dhat_B[mask] - group["U_B"][mask]
    assert np.sqrt(np.mean(error**2)) < 0.1
    expected_delay = 0.5 if channel_offset % 2 == 0 else -0.5
    assert info["fir_delay_A_samples"] == expected_delay
