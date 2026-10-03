# ADC Hierarchical Calibration

Numerical reference implementation for **Coarse/Fine-Decoupled Group-Predictive Calibration for Partially Interleaved Pipelined-SAR ADCs**.

The method aligns backend SAR fine codes within each MDAC group, predicts the target-group total code from the calibrated reference group, subtracts the target group's local coarse code, and estimates fine-domain gain, offset and a first-order timing-like correction. It supports foreground least squares (Mode A) and fixed-scale, sample-ordered NLMS (Mode B).

## Contents

| File | Role in the paper |
| --- | --- |
| `calibration.py` | Intra-group alignment, group prediction, local coarse subtraction, safe masks, LS and NLMS correction |
| `model.py` | In-memory behavioral model: redundant 2.5-effective-bit coarse stage, dual MDAC groups and four backend SAR channels |
| `fft_output_code.py` | Numerical SNDR, SFDR and ENOB calculation, including input-frequency alias handling |
| `scripts/run_foreground_validation.py` | Behavioral single-tone calibration, 20-ps skew case and input-frequency robustness |
| `scripts/scan_mdac_variance_ratio.py` | Redundant-residue power-estimation bias and the four aligned target-construction alternatives |
| `scripts/run_mode_b_timing_validation.py` | Five-seed physical cold-start validation for multitone, band-limited random and 16-QAM DMT inputs |
| `scripts/run_mode_b_tracking_validation.py` | Abrupt gain/offset/aperture-skew step and reacquisition without reinitialization |
| `scripts/validate_cadence_adc.py` | Numerical post-layout validation using a user-supplied Cadence CSV |
| `tests/` | Regression checks for calibration recovery, NLMS update order, derivative units and FFT/group alignment |

Only source code, documentation, dependency metadata and the MIT license are distributed. ADC output-code records, post-layout datasets, generated results, manuscript files, figures, plotting code and a browser UI are not included. Synthetic verification records are generated in memory when a validation script runs. The scripts print numerical summaries as JSON and do not write ADC-code or figure files.

## Install and test

Use Python 3.10 or newer:

```bash
python -m pip install -r requirements.txt
python -m pytest
```

The only dependencies are NumPy and pytest; no plotting or web framework is required.

## Reproduce the numerical experiments

Run from the repository root:

```bash
python scripts/run_foreground_validation.py --frequency-sweep
python scripts/scan_mdac_variance_ratio.py
python scripts/run_mode_b_timing_validation.py
python scripts/run_mode_b_tracking_validation.py
```

Mode A uses a known sinusoidal waveform. The default behavioral case uses 65536 samples at 1 GS/s and coherent tone bin 4001. The frequency sweep uses the paper's fixed mismatch case with zero inter-MDAC aperture skew; the separate timing check uses A/B aperture offsets of 30/50 ps (20-ps B-minus-A skew) and ch3 correction gain 1.008, matching the reported timing-mapping check.

Mode B uses three unknown-input classes and five fixed seeds, with no fitted initialization. Its correction state starts at `(g_B, o_B, k_B) = (1, 0, 0)`. The fine and slope feature scales are both the architecture-known 128 LSB. Each safe sample is corrected before its error updates the next coefficient state. The fixed predictor/differentiator lengths are 15/31 taps, corresponding to 22 A-group samples or 44 full-rate clocks of cascade delay. The default stationary-input NLMS step is 0.005; the tracking experiment uses 0.01.

Mode B requires persistently excited input below the A-stream Nyquist limit, `fs/4`. Known-tone Mode A can disambiguate the A-stream alias for tones approaching `fs/2`. The timing-like coefficient converts to an equivalent aperture-skew estimate only under the stated small-skew and supported-passband assumptions.

## Apply the algorithm to your own arrays

```python
from calibration import CalibrationConfig, calibrate_adc

corrected, coefficients, diagnostics, metrics = calibrate_adc(
    D_coarse,
    F_raw,
    sar_id=sar_id,
    fs=1e9,
    fin=known_tone_hz,
    config=CalibrationConfig(reference_method="known_tone", inter_method="two_stage"),
)
```

`D_coarse` and `F_raw` are one-dimensional arrays in common final-output LSB units, and `sar_id` identifies channels 0..3. In the verification architecture, ch0/ch2 share MDAC-A and ch1/ch3 share MDAC-B. The synthetic ideal code and injected mismatch truth are evaluation metadata, not calibration inputs. `calibrate_adc_background` exposes the fixed-scale NLMS array flow; `run_background_nlms_stream` exposes the sample-ordered adaptive update kernel.

For your own exported post-layout record:

```bash
python scripts/validate_cadence_adc.py --input /path/to/your_adc_export.csv --fs 1e9 --fin 401733398.4375 --window-samples 8192 --inject-preset sar_pair_mismatch
```

The adapter expects CSV fields `sample_idx`, `time`, `valid`, `b11` through `b0`, `flash_u`, `sar_u`, `sar_eff`, `F_raw`, `D_coarse`, `D_raw` and `AC`. It reconstructs the redundant backend SAR with weights `[128, 64, 32, 32, 16, 8, 4, 2, 1]`, checks the exported reconstruction, selects a valid record window and evaluates raw/intra-group/full-calibration stages. Controlled injection changes only the fine-code path. Foundry-derived or post-layout raw records must be supplied by the user and are not distributed here; synthetic runs do not reproduce circuit-generated settling and residue distributions.

Standalone numerical FFT analysis also requires your own record:

```bash
python fft_output_code.py /path/to/your_codes.csv --column D_raw --fs 1e9 --fin 401733398.4375
```

## License

MIT; see `LICENSE`.
