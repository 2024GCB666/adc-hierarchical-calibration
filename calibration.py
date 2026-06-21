"""针对合成架构模型的分层ADC校准。

校准流程遵循
``ADC分层校准算法_架构建模与合成数据验证规格(2).md``：

1. 组内（Intra-group）SAR校准：ch2对齐到ch0，ch3对齐到ch1。
2. 从校准后的fine code中提取并构建A/B两组MDAC数据流。
3. 从A组数据流预测B组的理想总码字（Dhat_B）。
4. 使用最小二乘法估计B组的fine增益、偏移以及类似时序（timing-like）的失配项。
5. 将校准后的A/B数据流重新交织回全速（full rate）。

仅面向架构的数组用于参数估计。合成的 ``D_ideal`` 和 ``truth`` 元数据仅用于校准效果的验证。
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np

from fft_output_code import compute_fft_metrics
from model import (
    DatasetConfig,
    MismatchConfig,
    SignalConfig,
    TimingConfig,
    build_synthetic_adc_dataset,
    timing_case,
)


Array = np.ndarray
ReferenceMethod = Literal["linear", "fractional_delay_fir", "known_tone"]
InterMethod = Literal["two_stage", "joint", "nlms"]
FIR_RESPONSE_ERROR_TARGET = 1e-3
FIR_MAX_AUTO_TAPS = 1001


@dataclass(frozen=True)
class CalibrationConfig:
    """前台分层校准的可调配置参数。"""

    N_fine: int = 8
    reference_method: ReferenceMethod = "fractional_delay_fir"
    fir_taps: int = 15
    fine_safe_ratio: float = 0.8
    use_coarse_stable_mask: bool = False
    slope_percentile: float = 50.0
    inter_method: InterMethod = "two_stage"
    known_tone_source: Literal["A", "full"] = "A"
    lms_mu: float = 0.005
    lms_epochs: int = 1
    lms_eps: float = 1e-9
    lms_trace_block_size: int = 1024
    lms_settling_ratio: float = 1.1


def _as_float_array(values: Array) -> Array:
    return np.asarray(values, dtype=np.float64)


def _rms(values: Array) -> float:
    v = _as_float_array(values)
    return float(np.sqrt(np.mean(v * v)))


def _safe_mean(values: Array) -> float:
    return float(np.mean(_as_float_array(values)))


def _safe_std(values: Array) -> float:
    return float(np.std(_as_float_array(values)))


def _corr(x: Array, y: Array) -> float | None:
    x = _as_float_array(x)
    y = _as_float_array(y)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(mask)) < 2:
        return None
    x = x[mask] - np.mean(x[mask])
    y = y[mask] - np.mean(y[mask])
    denom = float(np.sqrt(np.dot(x, x) * np.dot(y, y)))
    if denom == 0.0 or not math.isfinite(denom):
        return None
    return float(np.dot(x, y) / denom)


def _stats(values: Array) -> dict[str, float]:
    v = _as_float_array(values)
    return {
        "mean": float(np.mean(v)),
        "std": float(np.std(v)),
        "min": float(np.min(v)),
        "max": float(np.max(v)),
    }


def _json_ready(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v) for v in obj]
    return obj


def _standardize_on_mask(values: Array, mask: Array) -> tuple[Array, float, float]:
    values = _as_float_array(values)
    mask = np.asarray(mask, dtype=bool) & np.isfinite(values)
    if int(np.sum(mask)) == 0:
        return np.zeros_like(values), 0.0, 1.0
    center = float(np.mean(values[mask]))
    scale = float(np.std(values[mask]))
    if not math.isfinite(scale) or scale == 0.0:
        scale = 1.0
    return (values - center) / scale, center, scale


def calibrate_pair_by_stats(reference: Array, target: Array) -> tuple[float, float]:
    """通过统计量校准两个通道，返回增益(g)和偏移(o)，使得 corrected_target = g * target + o。"""

    reference = _as_float_array(reference)
    target = _as_float_array(target)
    ref_std = float(np.std(reference))
    target_std = float(np.std(target))
    if target_std == 0.0 or not math.isfinite(target_std):
        raise ValueError("Cannot calibrate pair: target standard deviation is zero.")
    gain = ref_std / target_std
    offset = float(np.mean(reference) - gain * np.mean(target))
    return float(gain), float(offset)


def split_channels(F_raw: Array, sar_id: Array) -> dict[int, Array]:
    return {ch: _as_float_array(F_raw)[np.asarray(sar_id) == ch] for ch in range(4)}


def apply_intra_group_calibration(
    F_raw: Array,
    sar_id: Array,
) -> tuple[Array, dict[str, float], dict[str, Any]]:
    """使用均值和标准差统计量，将ch2对齐到ch0，将ch3对齐到ch1（组内SAR校准）。"""

    F_raw = _as_float_array(F_raw)
    sar_id = np.asarray(sar_id)
    channels = split_channels(F_raw, sar_id)
    g20, o20 = calibrate_pair_by_stats(channels[0], channels[2])
    g31, o31 = calibrate_pair_by_stats(channels[1], channels[3])

    F_corr = F_raw.copy()
    F_corr[sar_id == 2] = g20 * F_raw[sar_id == 2] + o20
    F_corr[sar_id == 3] = g31 * F_raw[sar_id == 3] + o31

    debug = {
        "before": {
            f"ch{ch}": _stats(channels[ch])
            for ch in range(4)
        },
        "after": {
            f"ch{ch}": _stats(F_corr[sar_id == ch])
            for ch in range(4)
        },
    }
    coeffs = {"g20": g20, "o20": o20, "g31": g31, "o31": o31}
    return F_corr, coeffs, debug


def build_group_streams(D_coarse: Array, F_corr: Array) -> dict[str, Array]:
    """从全速校准后的fine数据中提取并构建A/B两组 500 MS/s 的数据流。"""

    D_coarse = _as_float_array(D_coarse)
    F_corr = _as_float_array(F_corr)
    D_stage1 = D_coarse + F_corr
    return {
        "D_stage1": D_stage1,
        "D_coarse_A": D_coarse[0::2],
        "F_A": F_corr[0::2],
        "U_A": D_stage1[0::2],
        "D_coarse_B": D_coarse[1::2],
        "F_B": F_corr[1::2],
        "U_B": D_stage1[1::2],
    }


def linear_predict_B_from_A(U_A: Array) -> tuple[Array, Array, Array]:
    """使用线性插值从相邻的A组样本中预测位于中间的B组样本。"""

    U_A = _as_float_array(U_A)
    Dhat_B = np.full(U_A.size, np.nan, dtype=np.float64)
    valid = np.zeros(U_A.size, dtype=bool)
    if U_A.size >= 2:
        Dhat_B[:-1] = 0.5 * (U_A[:-1] + U_A[1:])
        valid[:-1] = True
    return Dhat_B, valid, np.asarray([0.5, 0.5], dtype=np.float64)


def _fractional_delay_fir_coefficients(
    num_taps: int,
    delay: float,
) -> tuple[Array, Array]:
    if num_taps < 3 or num_taps % 2 == 0:
        raise ValueError("num_taps must be an odd integer >= 3.")
    half = num_taps // 2
    offsets = np.arange(-half, half + 1, dtype=np.int64)
    window = np.hamming(num_taps)
    h = np.sinc(delay - offsets.astype(np.float64)) * window
    h_sum = float(np.sum(h))
    if h_sum == 0.0 or not math.isfinite(h_sum):
        raise ValueError("Fractional-delay FIR has invalid coefficient sum.")
    return offsets, h / h_sum


def _apply_fractional_delay_fir(U_A: Array, h: Array) -> tuple[Array, Array]:
    U_A = _as_float_array(U_A)
    h = _as_float_array(h)
    num_taps = int(h.size)
    if num_taps < 3 or num_taps % 2 == 0:
        raise ValueError("FIR coefficient count must be an odd integer >= 3.")
    half = num_taps // 2
    offsets = np.arange(-half, half + 1, dtype=np.int64)

    Dhat_B = np.full(U_A.size, np.nan, dtype=np.float64)
    valid = np.zeros(U_A.size, dtype=bool)
    if U_A.size >= num_taps:
        l_valid = np.arange(half, U_A.size - half, dtype=np.int64)
        acc = np.zeros(l_valid.size, dtype=np.float64)
        for coeff, offset in zip(h, offsets):
            acc += coeff * U_A[l_valid + offset]
        Dhat_B[l_valid] = acc
        valid[l_valid] = True
    return Dhat_B, valid


def _folded_full_rate_frequency(fin: float, fs: float) -> float:
    if fs <= 0.0 or not math.isfinite(fs):
        raise ValueError("fs must be a positive finite value.")
    if not math.isfinite(fin):
        raise ValueError("fin must be a finite value.")
    folded = abs(float(fin) / float(fs)) % 1.0
    if folded > 0.5:
        folded = 1.0 - folded
    return folded


def _fractional_delay_response_error(
    h: Array,
    offsets: Array,
    normalized_frequency: float,
    delay: float,
) -> float:
    f = abs(float(normalized_frequency))
    if f == 0.0:
        return 0.0
    omega = 2.0 * np.pi * f
    response = np.sum(h * np.exp(1j * omega * offsets.astype(np.float64)))
    desired = np.exp(1j * omega * float(delay))
    return float(abs(response / desired - 1.0))


def _resolve_fractional_delay_fir(
    num_taps: int,
    delay: float,
    fs: float | None,
    fin: float | None,
) -> tuple[int, Array, dict[str, Any]]:
    offsets, h = _fractional_delay_fir_coefficients(num_taps, delay)
    info: dict[str, Any] = {
        "fir_requested_taps": int(num_taps),
        "fir_effective_taps": int(num_taps),
        "fir_auto_taps_applied": False,
        "fir_response_error_target": FIR_RESPONSE_ERROR_TARGET,
        "fir_max_auto_taps": FIR_MAX_AUTO_TAPS,
    }
    if fs is None or fin is None:
        return int(num_taps), h, info

    full_rate_frequency = _folded_full_rate_frequency(float(fin), float(fs))
    a_rate_frequency = 2.0 * full_rate_frequency
    info["fir_normalized_frequency_to_A_fs"] = a_rate_frequency
    if a_rate_frequency >= 0.5:
        raise ValueError(
            "fractional_delay_fir A-only reference requires |fin| < fs/4 "
            "(A-group Nyquist). Use known_tone/full for this frequency."
        )

    requested_error = _fractional_delay_response_error(
        h,
        offsets,
        a_rate_frequency,
        delay,
    )
    info["fir_response_error_requested"] = requested_error
    info["fir_response_error_effective"] = requested_error
    if requested_error <= FIR_RESPONSE_ERROR_TARGET:
        return int(num_taps), h, info

    search_limit = max(int(num_taps), FIR_MAX_AUTO_TAPS)
    for candidate_taps in range(int(num_taps) + 2, search_limit + 1, 2):
        candidate_offsets, candidate_h = _fractional_delay_fir_coefficients(
            candidate_taps,
            delay,
        )
        candidate_error = _fractional_delay_response_error(
            candidate_h,
            candidate_offsets,
            a_rate_frequency,
            delay,
        )
        if candidate_error <= FIR_RESPONSE_ERROR_TARGET:
            info.update(
                {
                    "fir_effective_taps": int(candidate_taps),
                    "fir_auto_taps_applied": True,
                    "fir_response_error_effective": candidate_error,
                }
            )
            return int(candidate_taps), candidate_h, info

    raise ValueError(
        "fractional_delay_fir taps are too short for this high-frequency tone. "
        f"Requested taps={num_taps}, response error={requested_error:.4g}; "
        f"no odd tap count <= {search_limit} reached "
        f"{FIR_RESPONSE_ERROR_TARGET:.4g}. Use known_tone/full."
    )


def fractional_delay_predict_B_from_A(
    U_A: Array,
    num_taps: int = 15,
    delay: float = 0.5,
) -> tuple[Array, Array, Array]:
    """针对B组采样时间点的加窗Sinc分数延迟（fractional-delay）FIR预测器。"""

    _, h = _fractional_delay_fir_coefficients(num_taps, delay)
    Dhat_B, valid = _apply_fractional_delay_fir(U_A, h)
    return Dhat_B, valid, h


def fit_known_tone_reference(
    D_stage1: Array,
    U_A: Array,
    fs: float,
    fin: float,
    source: Literal["A", "full"] = "A",
) -> tuple[Array, Array, dict[str, float]]:
    """拟合已知单频正弦波，并预测B组的总码字。

    ``source="A"`` 仅拟合A组数据流，保持A组作为参考路径。
    ``source="full"`` 遵循规范中的全速（full-rate）拟合变体。
    """

    D_stage1 = _as_float_array(D_stage1)
    U_A = _as_float_array(U_A)
    omega = 2.0 * np.pi * float(fin) / float(fs)
    if source == "A":
        n_fit = np.arange(0, 2 * U_A.size, 2, dtype=np.float64)
        y_fit = U_A
    elif source == "full":
        n_fit = np.arange(D_stage1.size, dtype=np.float64)
        y_fit = D_stage1
    else:
        raise ValueError(f"Unsupported known-tone source {source!r}.")

    X = np.column_stack(
        [
            np.cos(omega * n_fit),
            np.sin(omega * n_fit),
            np.ones_like(n_fit),
        ]
    )
    theta, _, rank, svals = np.linalg.lstsq(X, y_fit, rcond=None)
    a, b, c = [float(v) for v in theta]
    n_B = np.arange(1, 2 * U_A.size + 1, 2, dtype=np.float64)
    Dhat_B = a * np.cos(omega * n_B) + b * np.sin(omega * n_B) + c
    slope_B = -a * omega * np.sin(omega * n_B) + b * omega * np.cos(omega * n_B)
    info = {
        "tone_a": a,
        "tone_b": b,
        "tone_c": c,
        "tone_rank": int(rank),
        "tone_cond": float(np.inf if svals[-1] == 0 else svals[0] / svals[-1]),
    }
    return Dhat_B, slope_B, info


def _centered_difference(values: Array) -> Array:
    values = _as_float_array(values)
    diff = np.full(values.size, np.nan, dtype=np.float64)
    if values.size >= 3:
        diff[1:-1] = values[2:] - values[:-2]
    return diff


def build_B_reference(
    group: dict[str, Array],
    fs: float | None,
    fin: float | None,
    cfg: CalibrationConfig,
) -> tuple[Array, Array, Array, Array, dict[str, Any]]:
    """构建B组的参考信号，返回预测的总码字Dhat_B、精细码字Fhat_B、归一化斜率基准slope_B以及有效掩码mask_ref。"""

    info: dict[str, Any] = {"reference_method": cfg.reference_method}
    if cfg.reference_method == "linear":
        Dhat_B, mask_ref, h = linear_predict_B_from_A(group["U_A"])
        slope_raw = _centered_difference(Dhat_B)
        info["fir_coefficients"] = h
    elif cfg.reference_method == "fractional_delay_fir":
        _, h, fir_info = _resolve_fractional_delay_fir(
            num_taps=cfg.fir_taps,
            delay=0.5,
            fs=fs,
            fin=fin,
        )
        Dhat_B, mask_ref = _apply_fractional_delay_fir(group["U_A"], h)
        info.update(fir_info)
        slope_raw = _centered_difference(Dhat_B)
        info["fir_coefficients"] = h
    elif cfg.reference_method == "known_tone":
        if fs is None or fin is None:
            raise ValueError("known_tone reference requires fs and fin.")
        Dhat_B, slope_raw, tone_info = fit_known_tone_reference(
            group["D_stage1"],
            group["U_A"],
            fs=float(fs),
            fin=float(fin),
            source=cfg.known_tone_source,
        )
        mask_ref = np.isfinite(Dhat_B) & np.isfinite(slope_raw)
        info.update(tone_info)
        info["known_tone_source"] = cfg.known_tone_source
    else:
        raise ValueError(f"Unsupported reference method {cfg.reference_method!r}.")

    Fhat_B = Dhat_B - group["D_coarse_B"]
    pre_mask = mask_ref & np.isfinite(Fhat_B) & np.isfinite(group["F_B"])
    slope_B, slope_center, slope_scale = _standardize_on_mask(slope_raw, pre_mask)
    info["slope_center"] = slope_center
    info["slope_scale"] = slope_scale
    return Dhat_B, Fhat_B, slope_B, mask_ref, info


def build_safe_mask(
    F_B: Array,
    Fhat_B: Array,
    slope_B: Array,
    C_B: Array | None,
    cfg: CalibrationConfig,
) -> tuple[Array, dict[str, int]]:
    """基于边界（edge）、fine码字范围（fine）以及coarse稳定条件（coarse）构建用于最小二乘估计的安全样本掩码（safe mask）。"""

    F_B = _as_float_array(F_B)
    Fhat_B = _as_float_array(Fhat_B)
    slope_B = _as_float_array(slope_B)
    F_fs = float(2 ** (cfg.N_fine - 1))

    mask_edge = np.isfinite(F_B) & np.isfinite(Fhat_B) & np.isfinite(slope_B)
    mask_fine = (
        (np.abs(F_B) < cfg.fine_safe_ratio * F_fs)
        & (np.abs(Fhat_B) < cfg.fine_safe_ratio * F_fs)
    )
    mask_coarse = np.ones_like(mask_edge)
    if cfg.use_coarse_stable_mask and C_B is not None:
        C_B = np.asarray(C_B)
        mask_coarse[:] = False
        if C_B.size >= 3:
            mask_coarse[1:-1] = (
                (C_B[:-2] == C_B[1:-1]) & (C_B[1:-1] == C_B[2:])
            )

    mask_safe = mask_edge & mask_fine & mask_coarse
    counts = {
        "total": int(mask_safe.size),
        "edge": int(np.sum(mask_edge)),
        "fine": int(np.sum(mask_edge & mask_fine)),
        "coarse": int(np.sum(mask_edge & mask_fine & mask_coarse)),
        "safe": int(np.sum(mask_safe)),
    }
    return mask_safe, counts


def estimate_static_gain_offset(x: Array, y: Array) -> tuple[float, float, int, float]:
    x = _as_float_array(x)
    y = _as_float_array(y)
    X = np.column_stack([x, np.ones_like(x)])
    theta, _, rank, svals = np.linalg.lstsq(X, y, rcond=None)
    cond = float(np.inf if svals[-1] == 0 else svals[0] / svals[-1])
    return float(theta[0]), float(theta[1]), int(rank), cond


def estimate_joint_ls(
    F_B: Array,
    Fhat_B: Array,
    slope_B: Array,
    mask: Array,
) -> tuple[dict[str, float], dict[str, Any], Array]:
    x = _as_float_array(F_B)[mask]
    y = _as_float_array(Fhat_B)[mask]
    s = _as_float_array(slope_B)[mask]
    X = np.column_stack([x, np.ones_like(x), s])
    theta, _, rank, svals = np.linalg.lstsq(X, y, rcond=None)
    cond = float(np.inf if svals[-1] == 0 else svals[0] / svals[-1])
    coeffs = {
        "g_B": float(theta[0]),
        "o_B": float(theta[1]),
        "k_B": float(theta[2]),
    }
    residual = Fhat_B - (
        coeffs["g_B"] * F_B + coeffs["o_B"] + coeffs["k_B"] * slope_B
    )
    diagnostics = {
        "rank_X": int(rank),
        "cond_X": cond,
        "corr_F_s": _corr(x, s),
    }
    return coeffs, diagnostics, residual


def estimate_inter_group_B_nlms(
    F_B: Array,
    Fhat_B: Array,
    slope_B: Array,
    mask_safe: Array,
    cfg: CalibrationConfig,
) -> tuple[dict[str, float], dict[str, Any], Array]:
    F_B = _as_float_array(F_B)
    Fhat_B = _as_float_array(Fhat_B)
    slope_B = _as_float_array(slope_B)
    mask_safe = np.asarray(mask_safe, dtype=bool)
    update_count = int(np.sum(mask_safe))
    if update_count < 8:
        raise ValueError("Too few safe B samples for inter-group NLMS calibration.")

    mu = float(cfg.lms_mu)
    if not math.isfinite(mu) or mu <= 0.0:
        raise ValueError("lms_mu must be a positive finite value.")
    epochs = int(cfg.lms_epochs)
    if epochs < 1:
        raise ValueError("lms_epochs must be >= 1.")
    eps = float(cfg.lms_eps)
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("lms_eps must be a positive finite value.")
    trace_block_size = int(cfg.lms_trace_block_size)
    if trace_block_size < 1:
        raise ValueError("lms_trace_block_size must be >= 1.")
    settling_ratio = float(cfg.lms_settling_ratio)
    if not math.isfinite(settling_ratio) or settling_ratio < 1.0:
        raise ValueError("lms_settling_ratio must be >= 1.0.")

    f_scale = float(np.std(F_B[mask_safe]))
    if not math.isfinite(f_scale) or f_scale == 0.0:
        f_scale = 1.0

    theta = np.asarray([f_scale, 0.0, 0.0], dtype=np.float64)
    update_indices = np.flatnonzero(mask_safe)
    block_trace: list[dict[str, Any]] = []
    epoch_rms_trace: list[float] = []
    total_update_count = 0
    last_trace_update_count = 0

    def residual_from_theta(theta_v: Array) -> Array:
        return Fhat_B - (
            theta_v[0] * (F_B / f_scale) + theta_v[1] + theta_v[2] * slope_B
        )

    def append_trace(epoch_idx: int, sample_idx: int) -> None:
        nonlocal last_trace_update_count
        residual_v = residual_from_theta(theta)
        block_trace.append(
            {
                "update_count": int(total_update_count),
                "epoch": int(epoch_idx),
                "sample_index_B": int(sample_idx),
                "rms_res_B": _rms(residual_v[mask_safe]),
                "g_B": float(theta[0] / f_scale),
                "o_B": float(theta[1]),
                "k_B": float(theta[2]),
            }
        )
        last_trace_update_count = total_update_count

    for epoch_idx in range(1, epochs + 1):
        for idx in update_indices:
            phi = np.asarray([F_B[idx] / f_scale, 1.0, slope_B[idx]], dtype=np.float64)
            if not np.all(np.isfinite(phi)) or not math.isfinite(Fhat_B[idx]):
                continue
            error = float(Fhat_B[idx] - np.dot(theta, phi))
            norm = eps + float(np.dot(phi, phi))
            theta += (mu * error / norm) * phi
            total_update_count += 1
            if total_update_count % trace_block_size == 0:
                append_trace(epoch_idx, int(idx))
        residual_epoch = residual_from_theta(theta)
        epoch_rms_trace.append(_rms(residual_epoch[mask_safe]))

    if total_update_count > 0 and last_trace_update_count != total_update_count:
        append_trace(epochs, int(update_indices[-1]))

    coeffs = {
        "g_B": float(theta[0] / f_scale),
        "o_B": float(theta[1]),
        "k_B": float(theta[2]),
    }
    residual = Fhat_B - (
        coeffs["g_B"] * F_B + coeffs["o_B"] + coeffs["k_B"] * slope_B
    )
    final_rms = _rms(residual[mask_safe])
    settling_threshold = final_rms * settling_ratio
    convergence_block_count: int | None = None
    convergence_samples: int | None = None
    if block_trace:
        rms_values = np.asarray([row["rms_res_B"] for row in block_trace], dtype=np.float64)
        for trace_idx in range(rms_values.size):
            if bool(np.all(rms_values[trace_idx:] <= settling_threshold)):
                convergence_block_count = int(trace_idx + 1)
                convergence_samples = int(block_trace[trace_idx]["update_count"])
                break

    rms_decay_db_per_sample: float | None = None
    if len(block_trace) >= 2:
        first = block_trace[0]
        last = block_trace[-1]
        first_rms = float(first["rms_res_B"])
        last_rms = float(last["rms_res_B"])
        update_span = int(last["update_count"]) - int(first["update_count"])
        if first_rms > 0.0 and last_rms > 0.0 and update_span > 0:
            rms_decay_db_per_sample = float(20.0 * math.log10(first_rms / last_rms) / update_span)

    diagnostics = {
        "adaptive_method": "nlms",
        "lms_mu": mu,
        "lms_epochs": epochs,
        "lms_eps": eps,
        "lms_trace_block_size": trace_block_size,
        "lms_settling_ratio": settling_ratio,
        "lms_safe_sample_count": update_count,
        "lms_update_count": total_update_count,
        "lms_feature_scale_F_B": f_scale,
        "lms_rms_trace": [float(row["rms_res_B"]) for row in block_trace],
        "lms_epoch_rms_trace": epoch_rms_trace,
        "lms_block_trace": block_trace,
        "convergence_reached": convergence_samples is not None,
        "convergence_block_count": convergence_block_count,
        "convergence_samples": convergence_samples,
        "settling_threshold_rms": settling_threshold,
        "rms_decay_db_per_sample": rms_decay_db_per_sample,
        "rms_res_B": final_rms,
        "mean_res_B": float(np.mean(residual[mask_safe])),
        "corr_F_s": _corr(F_B[mask_safe], slope_B[mask_safe]),
        "corr_res_F": _corr(residual[mask_safe], F_B[mask_safe]),
        "corr_res_s": _corr(residual[mask_safe], slope_B[mask_safe]),
    }
    return coeffs, diagnostics, residual


def estimate_inter_group_B(
    F_B: Array,
    Fhat_B: Array,
    slope_B: Array,
    mask_safe: Array,
    cfg: CalibrationConfig,
) -> tuple[dict[str, float], dict[str, Any], Array, Array]:
    """估计B组的fine增益(g_B)、偏移(o_B)和类似时序的系数(k_B)。"""

    F_B = _as_float_array(F_B)
    Fhat_B = _as_float_array(Fhat_B)
    slope_B = _as_float_array(slope_B)
    mask_safe = np.asarray(mask_safe, dtype=bool)
    if int(np.sum(mask_safe)) < 8:
        raise ValueError("Too few safe B samples for inter-group LS calibration.")

    abs_s = np.abs(slope_B[mask_safe])
    slope_threshold = float(np.percentile(abs_s, cfg.slope_percentile))
    mask_static = mask_safe & (np.abs(slope_B) <= slope_threshold)
    if int(np.sum(mask_static)) < 4:
        mask_static = mask_safe

    if cfg.inter_method == "nlms":
        coeffs, diagnostics, residual = estimate_inter_group_B_nlms(
            F_B,
            Fhat_B,
            slope_B,
            mask_safe,
            cfg,
        )
    elif cfg.inter_method == "joint":
        coeffs, diagnostics, residual = estimate_joint_ls(
            F_B,
            Fhat_B,
            slope_B,
            mask_safe,
        )
    elif cfg.inter_method == "two_stage":
        g_B, o_B, rank_static, cond_static = estimate_static_gain_offset(
            F_B[mask_static],
            Fhat_B[mask_static],
        )
        residual_static = Fhat_B - (g_B * F_B + o_B)
        s = slope_B[mask_safe]
        y = residual_static[mask_safe]
        denom = float(np.dot(s, s))
        k_B = 0.0 if denom == 0.0 else float(np.dot(s, y) / denom)
        coeffs = {"g_B": g_B, "o_B": o_B, "k_B": k_B}
        residual = Fhat_B - (g_B * F_B + o_B + k_B * slope_B)
        _, joint_diag, _ = estimate_joint_ls(F_B, Fhat_B, slope_B, mask_safe)
        diagnostics = {
            **joint_diag,
            "rank_static": int(rank_static),
            "cond_static": float(cond_static),
        }
    else:
        raise ValueError(f"Unsupported inter_method {cfg.inter_method!r}.")

    residual_safe = residual[mask_safe]
    diagnostics.update(
        {
            "slope_threshold": slope_threshold,
            "mask_static_count": int(np.sum(mask_static)),
            "mask_safe_count": int(np.sum(mask_safe)),
            "rms_res_B": _rms(residual_safe),
            "mean_res_B": float(np.mean(residual_safe)),
            "corr_res_F": _corr(residual[mask_safe], F_B[mask_safe]),
            "corr_res_s": _corr(residual[mask_safe], slope_B[mask_safe]),
        }
    )
    return coeffs, diagnostics, residual, mask_static


def _truth_compare(
    coeffs: dict[str, float],
    truth: dict[str, Any] | None,
) -> dict[str, Any]:
    if not truth:
        return {}

    required = {"g0", "o0", "g1", "o1", "g2", "o2", "g3", "o3"}
    if not required.issubset(truth):
        return {}
    g0 = float(truth["g0"])
    g1 = float(truth["g1"])
    g2 = float(truth["g2"])
    g3 = float(truth["g3"])
    o0 = float(truth["o0"])
    o1 = float(truth["o1"])
    o2 = float(truth["o2"])
    o3 = float(truth["o3"])
    expected = {
        "g20": g2 / g0,
        "o20": (o2 - o0) / g0,
        "g31": g3 / g1,
        "o31": (o3 - o1) / g1,
        "g_B": g1 / g0,
        "o_B": (o1 - o0) / g0,
    }
    return {
        name: {
            "true": value,
            "estimated": coeffs.get(name),
            "error": None
            if name not in coeffs
            else float(coeffs[name] - value),
        }
        for name, value in expected.items()
    }


def _finite_or_none(value: Any) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def timing_slope_basis_factor(
    reference_method: ReferenceMethod,
    fs: float | None,
    fin: float | None,
) -> float | None:
    """Return the scale from d(input_code)/dn to the timing LS slope basis."""

    if reference_method == "known_tone":
        return 1.0
    if reference_method not in ("linear", "fractional_delay_fir"):
        return None
    fs_v = _finite_or_none(fs)
    fin_v = _finite_or_none(fin)
    if fs_v is None or fin_v is None or fs_v == 0.0:
        return None
    omega = 2.0 * math.pi * fin_v / fs_v
    if abs(omega) < 1e-15:
        return 4.0
    return float(2.0 * math.sin(2.0 * omega) / omega)


def build_timing_mismatch_comparison(
    coeffs: dict[str, float],
    truth: dict[str, Any] | None,
    reference_info: dict[str, Any],
    reference_method: ReferenceMethod,
    fs: float | None,
    fin: float | None,
) -> dict[str, Any]:
    """Compare injected B-A aperture skew with the fitted timing-like term."""

    truth = truth or {}
    dt_a = _finite_or_none(truth.get("dt_mdac_A"))
    dt_b = _finite_or_none(truth.get("dt_mdac_B"))
    fs_v = _finite_or_none(fs)
    k_b = _finite_or_none(coeffs.get("k_B"))
    slope_scale = _finite_or_none(reference_info.get("slope_scale"))
    basis_factor = timing_slope_basis_factor(reference_method, fs, fin)

    actual_b_minus_a_s = None
    if dt_a is not None and dt_b is not None:
        actual_b_minus_a_s = dt_b - dt_a

    expected_k_b = None
    fitted_b_minus_a_s = None
    if (
        fs_v is not None
        and slope_scale is not None
        and basis_factor is not None
        and abs(basis_factor) > 1e-15
    ):
        if actual_b_minus_a_s is not None:
            expected_k_b = -actual_b_minus_a_s * fs_v * slope_scale / basis_factor
        if k_b is not None:
            fitted_b_minus_a_s = -k_b * basis_factor / (slope_scale * fs_v)

    error_s = None
    error_percent = None
    if actual_b_minus_a_s is not None and fitted_b_minus_a_s is not None:
        error_s = fitted_b_minus_a_s - actual_b_minus_a_s
        if actual_b_minus_a_s != 0.0:
            error_percent = error_s / actual_b_minus_a_s * 100.0

    return {
        "dt_mdac_A_ps": None if dt_a is None else dt_a * 1e12,
        "dt_mdac_B_ps": None if dt_b is None else dt_b * 1e12,
        "actual_B_minus_A_ps": None
        if actual_b_minus_a_s is None
        else actual_b_minus_a_s * 1e12,
        "fitted_B_minus_A_ps": None
        if fitted_b_minus_a_s is None
        else fitted_b_minus_a_s * 1e12,
        "B_minus_A_error_ps": None if error_s is None else error_s * 1e12,
        "B_minus_A_error_percent": error_percent,
        "expected_k_B": expected_k_b,
        "estimated_k_B": k_b,
        "slope_scale": slope_scale,
        "slope_basis_factor": basis_factor,
        "reference_method": reference_method,
        "note": (
            "B-A skew is converted with k_B = -dt_BA * fs * slope_scale / "
            "slope_basis_factor."
        ),
    }


def calibrate_adc(
    D_coarse: Array,
    F_raw: Array,
    *,
    sar_id: Array | None = None,
    C_raw: Array | None = None,
    fs: float | None = None,
    fin: float | None = None,
    D_ideal: Array | None = None,
    truth: dict[str, Any] | None = None,
    config: CalibrationConfig | None = None,
) -> tuple[Array, dict[str, float], dict[str, Any], dict[str, Any]]:
    """执行完整的分层ADC校准流程。

    返回 ``D_corr, coeffs, debug, metrics``。
    """

    cfg = config or CalibrationConfig()
    D_coarse = _as_float_array(D_coarse)
    F_raw = _as_float_array(F_raw)
    N = D_coarse.size
    if F_raw.size != N:
        raise ValueError("D_coarse and F_raw must have the same length.")
    if N % 4 != 0:
        raise ValueError("Sample count must be divisible by 4.")
    if sar_id is None:
        sar_id = np.arange(N, dtype=np.int64) % 4
    sar_id = np.asarray(sar_id)
    if sar_id.size != N:
        raise ValueError("sar_id must have the same length as D_coarse.")

    D_raw = D_coarse + F_raw
    F_intra, intra_coeffs, intra_debug = apply_intra_group_calibration(
        F_raw,
        sar_id,
    )
    group = build_group_streams(D_coarse, F_intra)
    D_after_intra = group["D_stage1"]
    Dhat_B, Fhat_B, slope_B, mask_ref, ref_info = build_B_reference(
        group,
        fs,
        fin,
        cfg,
    )

    C_B = None if C_raw is None else np.asarray(C_raw)[1::2]
    mask_safe, mask_counts = build_safe_mask(
        group["F_B"],
        Fhat_B,
        slope_B,
        C_B,
        cfg,
    )
    mask_safe &= mask_ref
    mask_counts["safe_after_reference"] = int(np.sum(mask_safe))
    inter_coeffs, inter_diag, res_B, mask_static = estimate_inter_group_B(
        group["F_B"],
        Fhat_B,
        slope_B,
        mask_safe,
        cfg,
    )

    coeffs = {**intra_coeffs, **inter_coeffs}
    slope_for_apply = np.nan_to_num(slope_B, nan=0.0, posinf=0.0, neginf=0.0)
    F_B_corr = (
        inter_coeffs["g_B"] * group["F_B"]
        + inter_coeffs["o_B"]
        + inter_coeffs["k_B"] * slope_for_apply
    )
    U_B_corr = group["D_coarse_B"] + F_B_corr
    D_corr = np.empty(N, dtype=np.float64)
    D_corr[0::2] = group["U_A"]
    D_corr[1::2] = U_B_corr

    debug = {
        "F_intra_corr": F_intra,
        "D_after_intra": D_after_intra,
        "U_A": group["U_A"],
        "U_B": group["U_B"],
        "Dhat_B": Dhat_B,
        "Fhat_B": Fhat_B,
        "s_B": slope_B,
        "mask_safe": mask_safe,
        "mask_static": mask_static,
        "res_B": res_B,
        "F_B_corr": F_B_corr,
        "reference": ref_info,
    }

    config_payload = asdict(cfg)
    for key in (
        "fir_requested_taps",
        "fir_effective_taps",
        "fir_auto_taps_applied",
        "fir_response_error_requested",
        "fir_response_error_effective",
        "fir_response_error_target",
        "fir_normalized_frequency_to_A_fs",
    ):
        if key in ref_info:
            config_payload[key] = ref_info[key]

    metrics: dict[str, Any] = {
        "config": config_payload,
        "intra": {
            "coeffs": intra_coeffs,
            "stats": intra_debug,
        },
        "inter": {
            "coeffs": inter_coeffs,
            "mask_counts": mask_counts,
            "diagnostics": inter_diag,
            "reference_tracking": {
                "rms_U_B_minus_Dhat_safe_before_lsb": _rms(
                    (group["U_B"] - Dhat_B)[mask_safe]
                ),
                "rms_U_B_corr_minus_Dhat_safe_after_lsb": _rms(
                    (U_B_corr - Dhat_B)[mask_safe]
                ),
            },
        },
        "truth_compare": _truth_compare(coeffs, truth),
        "timing_compare": build_timing_mismatch_comparison(
            coeffs,
            truth,
            ref_info,
            cfg.reference_method,
            fs,
            fin,
        ),
    }
    if D_ideal is not None:
        D_ideal = _as_float_array(D_ideal)
        metrics["error_vs_ideal"] = {
            "rms_raw_lsb": _rms(D_raw - D_ideal),
            "rms_after_intra_lsb": _rms(D_after_intra - D_ideal),
            "rms_after_full_lsb": _rms(D_corr - D_ideal),
            "mean_raw_lsb": float(np.mean(D_raw - D_ideal)),
            "mean_after_full_lsb": float(np.mean(D_corr - D_ideal)),
        }
        metrics["reference_vs_ideal"] = {
            "rms_Dhat_B_minus_D_ideal_B_on_safe_lsb": _rms(
                (Dhat_B - D_ideal[1::2])[mask_safe]
            ),
        }
    return D_corr, coeffs, debug, metrics


def calibrate_adc_background(
    D_coarse: Array,
    F_raw: Array,
    *,
    sar_id: Array | None = None,
    C_raw: Array | None = None,
    fs: float | None = None,
    fin: float | None = None,
    D_ideal: Array | None = None,
    truth: dict[str, Any] | None = None,
    config: CalibrationConfig | None = None,
) -> tuple[Array, dict[str, float], dict[str, Any], dict[str, Any]]:
    """Run the background-calibration variant without changing foreground defaults."""

    cfg = replace(config or CalibrationConfig(), inter_method="nlms")
    return calibrate_adc(
        D_coarse,
        F_raw,
        sar_id=sar_id,
        C_raw=C_raw,
        fs=fs,
        fin=fin,
        D_ideal=D_ideal,
        truth=truth,
        config=cfg,
    )


def load_npz_dataset(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = np.load(path, allow_pickle=False)
    data: dict[str, Any] = {key: payload[key] for key in payload.files}
    truth: dict[str, Any] = {}
    if "truth_json" in payload.files:
        truth = json.loads(str(payload["truth_json"]))
    return data, truth


def build_demo_dataset(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    fs = float(args.fs)
    fin = float(args.fin) if args.fin is not None else args.tone_bin / args.N * fs
    base_timing = timing_case(args.timing_case)
    timing = base_timing
    if args.dt_mdac_A_ps is not None or args.dt_mdac_B_ps is not None:
        timing = TimingConfig(
            dt_flash_common=base_timing.dt_flash_common,
            dt_mdac_A=(
                float(args.dt_mdac_A_ps) * 1e-12
                if args.dt_mdac_A_ps is not None
                else base_timing.dt_mdac_A
            ),
            dt_mdac_B=(
                float(args.dt_mdac_B_ps) * 1e-12
                if args.dt_mdac_B_ps is not None
                else base_timing.dt_mdac_B
            ),
            dt_sar=base_timing.dt_sar,
        )
    signal = SignalConfig(
        fs=fs,
        fin=fin,
        amplitude=float(args.amplitude),
        phase=float(args.phase),
        dc=0.0,
    )
    mismatch = MismatchConfig(
        g0=float(args.g0),
        o0=float(args.o0),
        g1=float(args.g1),
        o1=float(args.o1),
        g2=float(args.g2),
        o2=float(args.o2),
        g3=float(args.g3),
        o3=float(args.o3),
        noise_std=float(args.noise_std),
    )
    cfg = DatasetConfig(
        N=int(args.N),
        seed=int(args.seed),
        signal=signal,
        timing=timing,
        mismatch=mismatch,
    )
    data, truth, _ = build_synthetic_adc_dataset(cfg)
    return data, truth


def add_fft_metrics(
    metrics: dict[str, Any],
    D_raw: Array,
    D_after_intra: Array,
    D_corr: Array,
    fs: float,
    fin: float | None,
) -> None:
    fft_metrics = {}
    for name, values in (
        ("raw", D_raw),
        ("after_intra", D_after_intra),
        ("after_full", D_corr),
    ):
        fft_metrics[name], _ = compute_fft_metrics(
            _as_float_array(values),
            fs=float(fs),
            fin=None if fin is None else float(fin),
            window="auto",
        )
    metrics["fft"] = fft_metrics


def _fmt_table_value(value: Any, digits: int = 6) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(v):
        return "-"
    if v == 0.0:
        return "0"
    abs_v = abs(v)
    if 1e-3 <= abs_v < 1e5:
        return f"{v:.{digits}f}".rstrip("0").rstrip(".")
    return f"{v:.{digits}e}"


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    rendered_rows = [[_fmt_table_value(cell) for cell in row] for row in rows]
    widths = [
        max(len(headers[col]), *(len(row[col]) for row in rendered_rows))
        for col in range(len(headers))
    ]
    header_line = "| " + " | ".join(
        headers[col].ljust(widths[col]) for col in range(len(headers))
    ) + " |"
    sep_line = "| " + " | ".join("-" * widths[col] for col in range(len(headers))) + " |"
    body = [
        "| " + " | ".join(row[col].ljust(widths[col]) for col in range(len(headers))) + " |"
        for row in rendered_rows
    ]
    return "\n".join([header_line, sep_line, *body])


def _render_named_table(table: dict[str, Any]) -> str:
    return _markdown_table(table["headers"], table["rows"])


def build_comparison_tables(
    metrics: dict[str, Any],
    coeffs: dict[str, float],
    truth: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    """基于校准指标，生成用于Markdown报告的各项性能对比表格数据。"""

    stage_keys = [
        ("raw", "Raw"),
        ("after_intra", "After intra"),
        ("after_full", "After full"),
    ]
    rms_by_stage = {
        "raw": metrics.get("error_vs_ideal", {}).get("rms_raw_lsb"),
        "after_intra": metrics.get("error_vs_ideal", {}).get("rms_after_intra_lsb"),
        "after_full": metrics.get("error_vs_ideal", {}).get("rms_after_full_lsb"),
    }
    performance_rows = []
    for key, label in stage_keys:
        fft = metrics.get("fft", {}).get(key, {})
        performance_rows.append(
            [
                label,
                rms_by_stage.get(key),
                fft.get("SNDR_dB"),
                fft.get("SFDR_dBc_single_bin"),
                fft.get("ENOB_bits"),
            ]
        )

    coeff_meanings = {
        "g20": "ch2 gain correction to ch0",
        "o20": "ch2 offset correction to ch0 (LSB)",
        "g31": "ch3 gain correction to ch1",
        "o31": "ch3 offset correction to ch1 (LSB)",
        "g_B": "B gain correction to A",
        "o_B": "B offset correction to A (LSB)",
        "k_B": "B timing-like coefficient",
    }
    coeff_rows = []
    truth_compare = metrics.get("truth_compare", {})
    for name in ("g20", "o20", "g31", "o31", "g_B", "o_B", "k_B"):
        truth_row = truth_compare.get(name, {})
        true_value = truth_row.get("true")
        estimated = coeffs.get(name)
        error = None if true_value is None or estimated is None else estimated - true_value
        rel_error = None
        if true_value is not None and true_value != 0 and error is not None:
            rel_error = error / true_value * 100.0
        coeff_rows.append(
            [
                name,
                coeff_meanings[name],
                true_value,
                estimated,
                error,
                rel_error,
            ]
        )

    reference_tracking = metrics.get("inter", {}).get("reference_tracking", {})
    before = reference_tracking.get("rms_U_B_minus_Dhat_safe_before_lsb")
    after = reference_tracking.get("rms_U_B_corr_minus_Dhat_safe_after_lsb")
    reference_rows = [
        [
            "B stream vs A-predicted reference RMS on safe mask",
            before,
            after,
            None if before in (None, 0) or after is None else before / after,
        ]
    ]
    reference_vs_ideal = metrics.get("reference_vs_ideal", {}).get(
        "rms_Dhat_B_minus_D_ideal_B_on_safe_lsb"
    )
    reference_rows.append(
        [
            "A-predicted B reference vs D_ideal_B RMS on safe mask",
            reference_vs_ideal,
            "-",
            "-",
        ]
    )

    injected_rows = []
    if truth:
        for ch in range(4):
            injected_rows.append(
                [
                    f"ch{ch}",
                    truth.get(f"g{ch}"),
                    truth.get(f"o{ch}"),
                ]
            )
        injected_rows.append(["noise", "-", truth.get("noise_std")])
        injected_rows.append(["dt_mdac_A_ps", "-", float(truth.get("dt_mdac_A", 0.0)) * 1e12])
        injected_rows.append(["dt_mdac_B_ps", "-", float(truth.get("dt_mdac_B", 0.0)) * 1e12])

    ls_diag = metrics.get("inter", {}).get("diagnostics", {})
    mask_counts = metrics.get("inter", {}).get("mask_counts", {})
    ls_rows = [
        ["safe samples", mask_counts.get("safe_after_reference")],
        ["static samples", ls_diag.get("mask_static_count")],
        ["rank_X", ls_diag.get("rank_X")],
        ["cond_X", ls_diag.get("cond_X")],
        ["corr_F_s", ls_diag.get("corr_F_s")],
        ["rms_res_B_lsb", ls_diag.get("rms_res_B")],
        ["corr_res_F", ls_diag.get("corr_res_F")],
        ["corr_res_s", ls_diag.get("corr_res_s")],
    ]
    if ls_diag.get("adaptive_method") == "nlms":
        ls_rows.extend(
            [
                ["adaptive method", ls_diag.get("adaptive_method")],
                ["NLMS mu", ls_diag.get("lms_mu")],
                ["NLMS epochs", ls_diag.get("lms_epochs")],
                ["trace block size", ls_diag.get("lms_trace_block_size")],
                ["settling ratio", ls_diag.get("lms_settling_ratio")],
                ["safe samples per epoch", ls_diag.get("lms_safe_sample_count")],
                ["total coefficient updates", ls_diag.get("lms_update_count")],
                ["convergence reached", ls_diag.get("convergence_reached")],
                ["convergence block count", ls_diag.get("convergence_block_count")],
                ["convergence samples", ls_diag.get("convergence_samples")],
                ["settling threshold RMS", ls_diag.get("settling_threshold_rms")],
                ["RMS decay dB/sample", ls_diag.get("rms_decay_db_per_sample")],
            ]
        )

    return {
        "performance": {
            "title": "Before/after performance comparison",
            "headers": ["Stage", "RMS vs D_ideal (LSB)", "SNDR (dB)", "SFDR (dBc)", "ENOB (bit)"],
            "rows": performance_rows,
        },
        "coefficients": {
            "title": "Estimated correction coefficients vs injected truth",
            "headers": ["Coeff", "Meaning", "True", "Estimated", "Error", "Rel err (%)"],
            "rows": coeff_rows,
        },
        "reference_tracking": {
            "title": "B-group reference tracking",
            "headers": ["Metric", "Before", "After", "Improvement x"],
            "rows": reference_rows,
        },
        "injected_mismatch": {
            "title": "Raw injected mismatch parameters",
            "headers": ["Item", "Injected gain", "Injected offset / value"],
            "rows": injected_rows,
        },
        "ls_diagnostics": {
            "title": "Inter-group calibration diagnostics",
            "headers": ["Metric", "Value"],
            "rows": ls_rows,
        },
    }


def build_markdown_report(
    stem: str,
    coeffs: dict[str, float],
    metrics: dict[str, Any],
    files: dict[str, str],
) -> str:
    lines = [
        f"# ADC Calibration Report: {stem}",
        "",
        "The coefficient truth values are correction coefficients derived from the injected channel mismatch.",
        "For example, g20_true = g2 / g0 and g_B_true = g1 / g0.",
        "",
        "## Output Files",
        "",
        _markdown_table(["Artifact", "Path"], [[name, path] for name, path in files.items()]),
        "",
    ]
    for key in (
        "performance",
        "coefficients",
        "reference_tracking",
        "injected_mismatch",
        "ls_diagnostics",
        "timing_mismatch",
    ):
        table = metrics.get("tables", {}).get(key)
        if not table:
            continue
        lines.extend(
            [
                f"## {table['title']}",
                "",
                _render_named_table(table),
                "",
            ]
        )
    lines.extend(
        [
            "## Coefficients",
            "",
            _markdown_table(
                ["Coeff", "Value"],
                [[name, coeffs[name]] for name in ("g20", "o20", "g31", "o31", "g_B", "o_B", "k_B")],
            ),
            "",
        ]
    )
    return "\n".join(lines)


def save_calibration_outputs(
    output_dir: Path,
    stem: str,
    D_raw: Array,
    D_corr: Array,
    coeffs: dict[str, float],
    debug: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / f"{stem}_calibration.npz"
    json_path = output_dir / f"{stem}_calibration_summary.json"
    report_path = output_dir / f"{stem}_calibration_report.md"
    np.savez_compressed(
        npz_path,
        D_raw=_as_float_array(D_raw),
        D_after_intra=_as_float_array(debug["D_after_intra"]),
        D_corr=_as_float_array(D_corr),
        F_intra_corr=_as_float_array(debug["F_intra_corr"]),
        Dhat_B=_as_float_array(debug["Dhat_B"]),
        Fhat_B=_as_float_array(debug["Fhat_B"]),
        s_B=_as_float_array(debug["s_B"]),
        mask_safe=np.asarray(debug["mask_safe"], dtype=bool),
        res_B=_as_float_array(debug["res_B"]),
    )
    json_path.write_text(
        json.dumps(
            _json_ready({"coeffs": coeffs, "metrics": metrics}),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    files = {
        "npz": str(npz_path),
        "summary_json": str(json_path),
        "report_md": str(report_path),
    }
    report_path.write_text(
        build_markdown_report(stem, coeffs, metrics, files),
        encoding="utf-8",
    )
    return files


def print_report_tables(metrics: dict[str, Any]) -> None:
    for key in ("performance", "coefficients", "reference_tracking", "ls_diagnostics"):
        table = metrics.get("tables", {}).get(key)
        if not table:
            continue
        print("")
        print(table["title"])
        print(_render_named_table(table))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run layered ADC calibration on synthetic or saved data.",
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=None,
        help="Optional .npz dataset from model.py. If omitted, a demo dataset is generated.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "calibration")
    parser.add_argument(
        "--output-stem",
        default=None,
        help="Optional output filename stem. Defaults to the input/demo name.",
    )
    parser.add_argument("--N", type=int, default=2**16)
    parser.add_argument("--fs", type=float, default=1.0e9)
    parser.add_argument(
        "--fin",
        type=float,
        default=None,
        help="Demo tone in Hz. Defaults to tone_bin / N * fs.",
    )
    parser.add_argument("--tone-bin", type=int, default=4001)
    parser.add_argument("--amplitude", type=float, default=511.0)
    parser.add_argument("--phase", type=float, default=0.31)
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument(
        "--timing-case",
        choices=("case_0", "case_1", "case_2"),
        default="case_0",
    )
    parser.add_argument(
        "--dt-mdac-A-ps",
        type=float,
        default=None,
        help="Override demo MDAC-A aperture skew in ps.",
    )
    parser.add_argument(
        "--dt-mdac-B-ps",
        type=float,
        default=None,
        help="Override demo MDAC-B aperture skew in ps.",
    )
    parser.add_argument(
        "--reference-method",
        choices=("linear", "fractional_delay_fir", "known_tone"),
        default="known_tone",
    )
    parser.add_argument("--fir-taps", type=int, default=15)
    parser.add_argument("--fine-safe-ratio", type=float, default=0.8)
    parser.add_argument("--use-coarse-stable-mask", action="store_true")
    parser.add_argument(
        "--inter-method",
        choices=("two_stage", "joint", "nlms"),
        default="two_stage",
    )
    parser.add_argument(
        "--known-tone-source",
        choices=("A", "full"),
        default="A",
    )
    parser.add_argument("--lms-mu", type=float, default=0.005)
    parser.add_argument("--lms-epochs", type=int, default=1)
    parser.add_argument("--lms-eps", type=float, default=1e-9)
    parser.add_argument("--lms-trace-block-size", type=int, default=1024)
    parser.add_argument("--lms-settling-ratio", type=float, default=1.1)
    parser.add_argument("--no-fft", action="store_true")
    parser.add_argument("--noise-std", type=float, default=0.08)
    parser.add_argument("--g0", type=float, default=1.0)
    parser.add_argument("--o0", type=float, default=0.0)
    parser.add_argument("--g1", type=float, default=1.0)
    parser.add_argument("--o1", type=float, default=0.0)
    parser.add_argument("--g2", type=float, default=1.012)
    parser.add_argument("--o2", type=float, default=-1.5)
    parser.add_argument("--g3", type=float, default=0.988)
    parser.add_argument("--o3", type=float, default=1.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.input is None:
        data, truth = build_demo_dataset(args)
        stem = f"demo_{args.timing_case}_{args.reference_method}"
    else:
        data, truth = load_npz_dataset(args.input)
        stem = args.input.stem
    if args.output_stem:
        stem = args.output_stem

    cfg = CalibrationConfig(
        N_fine=8,
        reference_method=args.reference_method,
        fir_taps=args.fir_taps,
        fine_safe_ratio=args.fine_safe_ratio,
        use_coarse_stable_mask=bool(args.use_coarse_stable_mask),
        inter_method=args.inter_method,
        known_tone_source=args.known_tone_source,
        lms_mu=args.lms_mu,
        lms_epochs=args.lms_epochs,
        lms_eps=args.lms_eps,
        lms_trace_block_size=args.lms_trace_block_size,
        lms_settling_ratio=args.lms_settling_ratio,
    )
    calibrator = calibrate_adc_background if cfg.inter_method == "nlms" else calibrate_adc
    D_corr, coeffs, debug, metrics = calibrator(
        data["D_coarse"],
        data["F_raw"],
        sar_id=data.get("sar_id"),
        C_raw=data.get("C_raw"),
        fs=float(data["fs"]) if "fs" in data else None,
        fin=float(data["fin"]) if "fin" in data else None,
        D_ideal=data.get("D_ideal"),
        truth=truth,
        config=cfg,
    )
    D_raw = _as_float_array(data["D_coarse"]) + _as_float_array(data["F_raw"])
    if not args.no_fft and "fs" in data:
        add_fft_metrics(
            metrics,
            D_raw,
            debug["D_after_intra"],
            D_corr,
            fs=float(data["fs"]),
            fin=float(data["fin"]) if "fin" in data else None,
        )
    metrics["tables"] = build_comparison_tables(metrics, coeffs, truth)

    files = save_calibration_outputs(
        args.output_dir,
        stem,
        D_raw,
        D_corr,
        coeffs,
        debug,
        metrics,
    )

    print("Layered ADC calibration complete.")
    print(f"Reference method: {cfg.reference_method}")
    print(f"Output NPZ: {files['npz']}")
    print(f"Summary JSON: {files['summary_json']}")
    print(f"Markdown report: {files['report_md']}")
    print("Coefficients:")
    for name in ("g20", "o20", "g31", "o31", "g_B", "o_B", "k_B"):
        print(f"  {name}: {coeffs[name]:.8g}")
    if "error_vs_ideal" in metrics:
        e = metrics["error_vs_ideal"]
        print("RMS error vs D_ideal:")
        print(f"  raw:         {e['rms_raw_lsb']:.6g} LSB")
        print(f"  after intra: {e['rms_after_intra_lsb']:.6g} LSB")
        print(f"  after full:  {e['rms_after_full_lsb']:.6g} LSB")
    if "fft" in metrics:
        print("FFT SNDR / SFDR:")
        for key in ("raw", "after_intra", "after_full"):
            m = metrics["fft"][key]
            print(
                f"  {key:12s} SNDR={m['SNDR_dB']:.3f} dB, "
                f"SFDR={m['SFDR_dBc_single_bin']:.3f} dBc"
            )
    print_report_tables(metrics)


if __name__ == "__main__":
    main()
