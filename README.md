# ADC Hierarchical Calibration

Reference implementation for a group-predictive, coarse/fine-decoupled hierarchical digital calibration method for partially interleaved pipelined-SAR ADCs.

The code accompanies the paper draft describing a 2.5-bit Flash/coarse front end, dual-MDAC paths, and four backend SAR channels. It focuses on algorithmic reproducibility: synthetic ADC data generation, hierarchical calibration, FFT/SNDR/SFDR evaluation, and NLMS online-update experiments.

## What is included

- `model.py`  
  Synthetic partially interleaved pipelined-SAR ADC model with controlled fine-path gain, offset, noise, and timing-like perturbations.

- `calibration.py`  
  Main hierarchical calibration flow:
  - intra-group SAR fine-code alignment,
  - A-group based B-reference construction,
  - B-local fine-code target construction,
  - inter-group gain/offset/timing-like estimation by two-stage LS, joint LS, or NLMS,
  - diagnostic metrics and output export.

- `fft_output_code.py`  
  FFT-based SNDR, SFDR, ENOB, and spectrum plotting utilities for output-code data.

- `scripts/scan_mdac_variance_ratio.py`  
  Timing-skew sweep used to show why long-term fine-code power matching can be biased by coarse/residue branch statistics.

- `app.py` and `static/`  
  Optional FastAPI browser UI for interactive simulation and calibration.

- `tests/`  
  Smoke tests for calibration recovery, fractional-delay reference behavior, NLMS convergence, and FFT alias handling.

The proprietary or paper-specific artifacts are intentionally not included: LaTeX source, Word/PDF drafts, reference papers, paper figure-generation scripts, large generated TIFF/PDF figure exports, and private post-layout raw data.

## Install

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Quick smoke test

```bash
pytest
```

The main calibration script can also run a demo dataset directly:

```bash
python calibration.py --reference-method known_tone --inter-method two_stage
```

For the NLMS adaptive-update variant:

```bash
python calibration.py --reference-method known_tone --inter-method nlms --lms-mu 0.005 --lms-epochs 1
```

## Optional web UI

```bash
uvicorn app:app --host 127.0.0.1 --port 8000
```

Then open <http://127.0.0.1:8000>.

## Notes on post-layout output-code validation

The paper distinguishes behavioral-model validation from post-layout output-code validation with controlled digital fine-path mismatch injection. This repository contains the open calibration code and behavioral/synthetic verification scripts. Private foundry or chip-layout-derived raw output-code datasets are not redistributed here.

If you have exported post-layout output-code data with columns for coarse code, fine code, SAR/channel ID, and optional ideal/reference codes, you can apply the same `calibrate_adc(...)` API in `calibration.py`.

## License

This project is released under the MIT License.
