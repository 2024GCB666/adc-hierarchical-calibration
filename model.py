"""ADC architecture model and synthetic data generator.

This module implements the modeling-only part described in
``ADC分层校准算法_架构建模与合成数据验证规格(2).md``.

It intentionally does not implement calibration.  The output data preserves
the ADC structure:

    common 2.5-bit Flash/coarse path
      -> dual-MDAC 2-way interleaving
      -> four SAR backend channels
      -> D_raw = D_coarse + F_raw

Run:

    python model.py

The script writes a compressed NPZ, a CSV table, and a JSON summary under
``outputs/`` by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class SignalConfig:
    """输入模拟信号配置，默认生成单频正弦波: x[n] = A*sin(2*pi*fin/fs*n + phi) + dc"""
    fs: float = 1.0e9  # 总采样率 1 GS/s
    fin: float = 499e6  # 默认输入频率
    amplitude: float = 511.0  # 输入幅度，单位为最终输出 LSB (满量程)
    phase: float = 0.31
    dc: float = 0.0


@dataclass(frozen=True)
class TimingConfig:
    """采样时间配置，用于验证 Flash 与 MDAC 采样时刻不一致引起的误差 (Flash-MDAC timing skew)"""
    dt_flash_common: float = 0.0  # 前级 Flash 的等效采样时刻偏差
    dt_mdac_A: float = 0.0  # MDAC-A 组的采样时刻偏差
    dt_mdac_B: float = 0.0  # MDAC-B 组的采样时刻偏差
    # Metadata only.  Backend SAR samples an already-settled MDAC residue in
    # this model, so per-SAR aperture skew does not alter x_mdac or F_ideal.
    # 后级 SAR 对 MDAC residue 的采样 skew，通常仅作为 metadata 记录，因为在 behavioral model 中 residue 已经 settled
    dt_sar: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class MdacConfig:
    """MDAC 配置"""
    residue_gain: float = 4.0  # MDAC residue 放大倍数


@dataclass(frozen=True)
class MismatchConfig:
    """失配注入配置，包含独立的各通道增益和偏置失配。
    注：失配注入使用的是逆模型，这样后续校准器可以恢复这些系数。
    """
    g0: float = 1.0      # ch0 增益失配
    o0: float = 0.0      # ch0 偏置失配
    g1: float = 1.0      # ch1 增益失配
    o1: float = 0.0      # ch1 偏置失配
    g2: float = 1.012    # ch2 增益失配
    o2: float = -1.50    # ch2 偏置失配
    g3: float = 0.988    # ch3 增益失配
    o3: float = 1.20     # ch3 偏置失配
    noise_std: float = 0.08  # 添加的随机噪声标准差


@dataclass(frozen=True)
class CoarseModel:
    """前级 2.5-bit Coarse 判决器的架构模型配置参数"""
    codes: Array            # 判决码字列表 (例如: -3, -2, -1, 0, +1, +2, +3)
    thresholds: Array       # 对应的比较器判决阈值 (单位: 最终输出 LSB)
    dac_levels: Array       # 反馈的 DAC 电平 (单位: 最终输出 LSB)
    digital_weights: Array  # 用于最终数字重构的 coarse 权重贡献 (单位: 最终输出 LSB)
    name: str = "2p5b_7level_redundant"


@dataclass(frozen=True)
class DatasetConfig:
    """数据集生成配置，包含样本数 N、后级精度 N_fine 和各种配置参数"""
    N: int = 2**16
    N_fine: int = 8  # 后级 SAR 的精度，如 8-bit fine code
    seed: int = 20260614
    signal: SignalConfig = field(default_factory=SignalConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)
    mdac: MdacConfig = field(default_factory=MdacConfig)
    mismatch: MismatchConfig = field(default_factory=MismatchConfig)
    threshold_offsets: tuple[float, ...] | None = None  # Flash 比较器的 offset


def default_2p5b_redundant_model(
    N_fine: int = 8,
    threshold_offsets: tuple[float, ...] | None = None,
) -> CoarseModel:
    """返回默认的 7-level 冗余 2.5-bit Flash/MDAC 模型。
    
    即 C_raw ∈ {-3, -2, -1, 0, +1, +2, +3}。
    基于 N_fine (默认为8)，计算 F_FS = 128 (±128 的 fine scale)。
    DAC levels 间隔 W_DAC = 128。
    """

    fine_fs = 2 ** (N_fine - 1)  # 后级 fine 满量程 F_FS，例如 128
    codes = np.arange(-3, 4, dtype=np.int16)
    levels = (codes.astype(np.float64) * fine_fs).astype(np.float64)  # DAC levels: -384, -256, ..., +384
    thresholds = 0.5 * (levels[:-1] + levels[1:])  # 理想判决阈值为相邻 DAC level 中点
    if threshold_offsets is not None:
        offsets = np.asarray(threshold_offsets, dtype=np.float64)
        if offsets.shape != thresholds.shape:
            raise ValueError(
                "threshold_offsets must have one value per threshold "
                f"({thresholds.size} values for this model)."
            )
        thresholds = thresholds + offsets
    return CoarseModel(
        codes=codes,
        thresholds=thresholds.astype(np.float64),
        dac_levels=levels,
        digital_weights=levels.copy(),
    )


def generate_input_waveform(t: Array, cfg: SignalConfig) -> Array:
    """Generate the analog input in final-output-LSB units."""

    return cfg.amplitude * np.sin(2.0 * np.pi * cfg.fin * t + cfg.phase) + cfg.dc


def input_slope_per_sample(n: Array, cfg: SignalConfig) -> Array:
    """Derivative of the input code with respect to global sample index."""

    omega = 2.0 * np.pi * cfg.fin / cfg.fs
    return cfg.amplitude * omega * np.cos(omega * n + cfg.phase)


def flash_2p5b_quantize(x_flash: Array, model: CoarseModel) -> Array:
    """Table-driven 2.5-bit Flash quantizer.
    根据输入的采样值 (Flash 采样时刻)，使用阈值表查出 C_raw 判决码。
    """

    idx = np.searchsorted(model.thresholds, x_flash, side="right")
    return model.codes[idx].astype(np.int16)


def _levels_from_code(codes: Array, table_codes: Array, table_levels: Array) -> Array:
    """Map Flash codes to table levels without baking in a fixed formula."""

    idx = np.searchsorted(table_codes, codes)
    if np.any(idx < 0) or np.any(idx >= table_codes.size):
        raise ValueError("Code outside coarse model table.")
    if np.any(table_codes[idx] != codes):
        raise ValueError("Code outside coarse model table.")
    return table_levels[idx]


def dac_level_from_code(C_raw: Array, model: CoarseModel) -> Array:
    """根据 C_raw 查表获取 MDAC DAC feedback level / DAC 电平 D_DAC (input-referred final LSB)"""
    return _levels_from_code(C_raw, model.codes, model.dac_levels)


def coarse_to_final_lsb(C_raw: Array, model: CoarseModel) -> Array:
    """根据 C_raw 查表获取用于数字重构的 coarse contribution D_coarse (最终输出 LSB)"""
    return _levels_from_code(C_raw, model.codes, model.digital_weights)


def _normalized_on_mask(values: Array, mask: Array) -> tuple[Array, float, float]:
    """Normalize values on a mask and return a full-length normalized array."""

    center = float(np.mean(values[mask]))
    scale = float(np.std(values[mask]))
    if not np.isfinite(scale) or scale == 0.0:
        scale = 1.0
    normalized = (values - center) / scale
    return normalized, center, scale


def inject_fine_path_mismatch(
    F_ideal: Array,
    sar_id: Array,
    mdac_id: Array,
    slope_norm: Array,
    cfg: MismatchConfig,
) -> Array:
    """Inject fine-only mismatch using the inverse correction model.
    根据规格约束，fine-path mismatch 只能作用在 fine code (F_ideal) 上，
    不能直接作用在整个总码字 (D_raw) 上。
    """

    F_float = F_ideal.astype(np.float64).copy()

    # Independent SAR mismatch for all channels.
    ch0 = sar_id == 0
    ch1 = sar_id == 1
    ch2 = sar_id == 2
    ch3 = sar_id == 3

    F_float[ch0] = (F_float[ch0] - cfg.o0) / cfg.g0
    F_float[ch1] = (F_float[ch1] - cfg.o1) / cfg.g1
    F_float[ch2] = (F_float[ch2] - cfg.o2) / cfg.g2
    F_float[ch3] = (F_float[ch3] - cfg.o3) / cfg.g3

    return F_float


def quantize_and_clip_fine_code(
    F_float: Array,
    rng: np.random.Generator,
    N_fine: int,
    noise_std: float,
) -> tuple[Array, Array, Array]:
    """Add backend noise, round, and clip to signed 8-bit style range.
    后级 SAR 输出 8-bit fine code，建模时应将其转换为 signed fine code。
    中心接近 0，范围例如 [-128, +127]。
    """

    F_min = -(2 ** (N_fine - 1))
    F_max = 2 ** (N_fine - 1) - 1
    noise = rng.normal(loc=0.0, scale=noise_std, size=F_float.shape)
    F_rounded = np.rint(F_float + noise)
    clipped = (F_rounded < F_min) | (F_rounded > F_max)
    F_raw = np.clip(F_rounded, F_min, F_max).astype(np.int16)
    return F_raw, noise, clipped


def build_synthetic_adc_dataset(
    cfg: DatasetConfig | None = None,
    coarse_model: CoarseModel | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build modeling-only synthetic ADC data.

    Returns:
        data: complete ADC architecture arrays.
        truth: injected truth/configuration metadata.
        metrics: modeling integrity statistics.
    """

    cfg = cfg or DatasetConfig()
    if cfg.N <= 0:
        raise ValueError("N must be positive.")
    if cfg.N % 4 != 0:
        raise ValueError("N must be divisible by 4 to keep SAR channels balanced.")

    model = coarse_model or default_2p5b_redundant_model(
        cfg.N_fine,
        cfg.threshold_offsets,
    )
    rng = np.random.default_rng(cfg.seed)

    n = np.arange(cfg.N, dtype=np.int64)
    # 通道编号定义: 4-way SAR TI (ch0: n=4k, ch1: n=4k+1, ...)
    sar_id = (n % 4).astype(np.int8)
    # 双 MDAC 编号定义: 2-way MDAC TI (MDAC-A: n=2l, MDAC-B: n=2l+1)
    mdac_id = (n % 2).astype(np.int8)
    Ts = 1.0 / cfg.signal.fs

    t_nominal = n * Ts
    # Mode 2: Flash-MDAC timing skew mode (Flash 判决等效采样时刻 t_flash)
    t_flash = t_nominal + cfg.timing.dt_flash_common
    dt_mdac_group = np.zeros(cfg.N, dtype=np.float64)
    dt_mdac_group[mdac_id == 0] = cfg.timing.dt_mdac_A
    dt_mdac_group[mdac_id == 1] = cfg.timing.dt_mdac_B
    dt_sar_table = np.asarray(cfg.timing.dt_sar, dtype=np.float64)
    if dt_sar_table.shape != (4,):
        raise ValueError("TimingConfig.dt_sar must contain four SAR channel skews.")
    dt_sar = dt_sar_table[sar_id]
    t_mdac_group = t_nominal + dt_mdac_group
    # Effective residue sampling time.  Only the MDAC A/B aperture skew changes
    # the input instant that forms the residue.  The backend SAR sees the MDAC
    # residue after it has settled, so dt_sar is not applied to t_mdac.
    # MDAC residue 的等效采样时刻 t_mdac
    t_mdac = t_mdac_group.copy()

    # 获得对应的理想模拟波形
    x_analog = generate_input_waveform(t_nominal, cfg.signal)
    x_flash = generate_input_waveform(t_flash, cfg.signal)
    x_mdac = generate_input_waveform(t_mdac, cfg.signal)

    # In this behavioral model the ideal full-rate code is the input sampled
    # at the MDAC residue instant, so D_coarse + F_ideal reconstructs it.
    D_ideal = x_mdac.copy()

    # 共享的前级 2.5-bit Flash / coarse decision path (由 t_flash 产生)
    C_raw = flash_2p5b_quantize(x_flash, model)
    D_DAC = dac_level_from_code(C_raw, model)
    D_coarse = coarse_to_final_lsb(C_raw, model)

    if cfg.mdac.residue_gain == 0.0:
        raise ValueError("MDAC residue gain must be nonzero.")
    # 2.5-bit MDAC residue form, matching Vout = 4*Vin - D*Vref.
    # D_DAC is already the input-referred DAC level D*Vref/4, so the
    # output-side DAC term is residue_gain * D_DAC.
    # 构建 MDAC 的模拟残差 V_residue = G_MDAC * (x_mdac - D_DAC)
    V_residue = cfg.mdac.residue_gain * (x_mdac - D_DAC)
    F_residue_input_referred = V_residue / cfg.mdac.residue_gain
    # The backend SAR quantizes the settled MDAC residue.  F_ideal is the same
    # residue converted back to final-output-LSB units so digital reconstruction
    # remains D_raw = D_coarse + F_raw.  The D_DAC-D_coarse term keeps the model
    # correct if analog DAC levels and digital weights later diverge.
    # F_ideal[n] = x_mdac[n] - D_coarse[n] 理想无失配时的细化代码
    F_ideal = F_residue_input_referred + (D_DAC - D_coarse)
    F_ideal_q, _, F_ideal_clip = quantize_and_clip_fine_code(
        F_ideal,
        rng=np.random.default_rng(cfg.seed + 1),
        N_fine=cfg.N_fine,
        noise_std=0.0,
    )
    D_no_mismatch = D_coarse.astype(np.int32) + F_ideal_q.astype(np.int32)

    slope = input_slope_per_sample(n, cfg.signal)
    slope_norm, slope_mean_B, slope_std_B = _normalized_on_mask(slope, mdac_id == 1)

    # 注入失配和噪声 (失配只会注入到 fine-path 上，不影响总码字上的 coarse 结构)
    F_mismatch_float = inject_fine_path_mismatch(
        F_ideal=F_ideal,
        sar_id=sar_id,
        mdac_id=mdac_id,
        slope_norm=slope_norm,
        cfg=cfg.mismatch,
    )
    F_raw, fine_noise, fine_clipped = quantize_and_clip_fine_code(
        F_mismatch_float,
        rng=rng,
        N_fine=cfg.N_fine,
        noise_std=cfg.mismatch.noise_std,
    )
    
    # 最终粗细化结合公式: D_raw = D_coarse + F_raw
    D_raw = D_coarse.astype(np.int32) + F_raw.astype(np.int32)

    data: dict[str, Any] = {
        "n": n,
        "fs": float(cfg.signal.fs),
        "fin": float(cfg.signal.fin),
        "sar_id": sar_id,
        "mdac_id": mdac_id,
        "t_nominal": t_nominal,
        "t_flash": t_flash,
        "t_mdac_group": t_mdac_group,
        "t_mdac": t_mdac,
        "dt_mdac_group": dt_mdac_group,
        "dt_sar": dt_sar,
        "x_analog": x_analog,
        "x_flash": x_flash,
        "x_mdac": x_mdac,
        "D_ideal": D_ideal,
        "C_raw": C_raw,
        "D_DAC": D_DAC,
        "D_coarse": D_coarse,
        "mdac_residue_gain": np.full(cfg.N, cfg.mdac.residue_gain, dtype=np.float64),
        "V_residue": V_residue,
        "F_residue_input_referred": F_residue_input_referred,
        "F_ideal": F_ideal,
        "F_ideal_q": F_ideal_q,
        "D_no_mismatch": D_no_mismatch,
        "slope_per_sample": slope,
        "slope_norm": slope_norm,
        "F_mismatch_float": F_mismatch_float,
        "fine_noise": fine_noise,
        "F_raw": F_raw,
        "D_raw": D_raw,
        "F_ideal_clipped": F_ideal_clip,
        "F_raw_clipped": fine_clipped,
    }

    truth: dict[str, Any] = {
        **asdict(cfg.mismatch),
        **asdict(cfg.timing),
        "N": cfg.N,
        "N_fine": cfg.N_fine,
        "seed": cfg.seed,
        "coarse_model": model.name,
        "signal": asdict(cfg.signal),
        "mdac": asdict(cfg.mdac),
        "slope_basis": "normalized d(input_code)/dn; normalized on B group",
        "slope_mean_B": slope_mean_B,
        "slope_std_B": slope_std_B,
        "threshold_table": model.thresholds.tolist(),
        "code_table": model.codes.tolist(),
        "dt_sar_by_channel": {
            f"ch{idx}": float(value)
            for idx, value in enumerate(cfg.timing.dt_sar)
        },
        "dac_level_table": {
            str(int(code)): float(level)
            for code, level in zip(model.codes, model.dac_levels)
        },
        "digital_weight_table": {
            str(int(code)): float(level)
            for code, level in zip(model.codes, model.digital_weights)
        },
    }

    metrics = compute_model_metrics(data, truth, cfg.N_fine)
    validate_architecture(data)
    return data, truth, metrics


def validate_architecture(data: dict[str, Any]) -> None:
    """Fail fast if the generated data violates the requested structure."""

    n = data["n"]
    if not np.array_equal(data["sar_id"], n % 4):
        raise AssertionError("sar_id must equal n % 4.")
    if not np.array_equal(data["mdac_id"], n % 2):
        raise AssertionError("mdac_id must equal n % 2.")
    if not np.array_equal(
        data["D_raw"],
        data["D_coarse"].astype(np.int32) + data["F_raw"].astype(np.int32),
    ):
        raise AssertionError("D_raw must equal D_coarse + F_raw.")


def _rms(x: Array) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def _safe_float(x: Any) -> float | None:
    value = float(x)
    if math.isfinite(value):
        return value
    return None


def _channel_stats(values: Array, sar_id: Array) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for ch in range(4):
        v = np.asarray(values[sar_id == ch], dtype=np.float64)
        stats[f"ch{ch}"] = {
            "mean": _safe_float(np.mean(v)),
            "std": _safe_float(np.std(v)),
            "min": _safe_float(np.min(v)),
            "max": _safe_float(np.max(v)),
        }
    return stats


def _channel_ratio(mask: Array, sar_id: Array) -> dict[str, float]:
    return {
        f"ch{ch}": float(np.mean(mask[sar_id == ch]))
        for ch in range(4)
    }


def compute_model_metrics(
    data: dict[str, Any],
    truth: dict[str, Any],
    N_fine: int,
) -> dict[str, Any]:
    """Return modeling integrity metrics only, not calibration results."""

    sar_id = data["sar_id"]
    C_raw = data["C_raw"]
    D_coarse = data["D_coarse"]
    F_raw = data["F_raw"]
    F_min = -(2 ** (N_fine - 1))
    F_max = 2 ** (N_fine - 1) - 1

    unique_C, count_C = np.unique(C_raw, return_counts=True)
    unique_D, count_D = np.unique(D_coarse, return_counts=True)

    ideal_err = data["D_no_mismatch"].astype(np.float64) - data["D_ideal"]
    raw_err = data["D_raw"].astype(np.float64) - data["D_ideal"]

    return {
        "model": {
            "N": int(data["n"].size),
            "fs": float(data["fs"]),
            "fin": float(data["fin"]),
            "coarse_states": unique_C.astype(int).tolist(),
            "coarse_state_counts": {
                str(int(code)): int(count)
                for code, count in zip(unique_C, count_C)
            },
            "coarse_levels": unique_D.astype(float).tolist(),
            "coarse_level_counts": {
                str(float(level)): int(count)
                for level, count in zip(unique_D, count_D)
            },
            "fine_range": {"min": int(F_min), "max": int(F_max)},
            "fine_clip_ratio": float(np.mean(data["F_raw_clipped"])),
            "fine_clip_ratio_by_channel": _channel_ratio(
                data["F_raw_clipped"],
                sar_id,
            ),
            "ideal_fine_clip_ratio": float(np.mean(data["F_ideal_clipped"])),
            "ideal_fine_clip_ratio_by_channel": _channel_ratio(
                data["F_ideal_clipped"],
                sar_id,
            ),
            "F_raw_by_channel": _channel_stats(F_raw, sar_id),
            "F_ideal_by_channel": _channel_stats(data["F_ideal"], sar_id),
            "D_raw_equals_D_coarse_plus_F_raw": bool(
                np.array_equal(
                    data["D_raw"],
                    data["D_coarse"].astype(np.int32)
                    + data["F_raw"].astype(np.int32),
                )
            ),
            "sar_id_is_n_mod_4": bool(np.array_equal(sar_id, data["n"] % 4)),
            "mdac_id_is_n_mod_2": bool(
                np.array_equal(data["mdac_id"], data["n"] % 2)
            ),
        },
        "error_vs_ideal": {
            "rms_no_mismatch_lsb": _rms(ideal_err),
            "rms_raw_lsb": _rms(raw_err),
            "mean_raw_error_lsb": _safe_float(np.mean(raw_err)),
            "min_raw_error_lsb": _safe_float(np.min(raw_err)),
            "max_raw_error_lsb": _safe_float(np.max(raw_err)),
        },
        "truth": {
            key: value
            for key, value in truth.items()
            if key
            in {
                "g0",
                "o0",
                "g1",
                "o1",
                "g2",
                "o2",
                "g3",
                "o3",
                "noise_std",
                "dt_flash_common",
                "dt_mdac_A",
                "dt_mdac_B",
                "dt_sar",
                "mdac",
            }
        },
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


def write_dataset_csv(data: dict[str, Any], csv_path: Path) -> None:
    """Write the full row-wise dataset to CSV."""

    columns = [
        "n",
        "sar_id",
        "mdac_id",
        "t_nominal",
        "t_flash",
        "t_mdac_group",
        "t_mdac",
        "dt_mdac_group",
        "dt_sar",
        "x_analog",
        "x_flash",
        "x_mdac",
        "D_ideal",
        "C_raw",
        "D_DAC",
        "D_coarse",
        "mdac_residue_gain",
        "V_residue",
        "F_residue_input_referred",
        "F_ideal",
        "F_ideal_q",
        "F_raw",
        "D_raw",
        "D_no_mismatch",
        "slope_norm",
        "F_raw_clipped",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        N = int(data["n"].size)
        for i in range(N):
            row = []
            for col in columns:
                value = data[col][i] if isinstance(data[col], np.ndarray) else data[col]
                if isinstance(value, (float, np.floating)):
                    row.append(f"{float(value):.12g}")
                elif isinstance(value, (bool, np.bool_)):
                    row.append(int(value))
                else:
                    row.append(value)
            writer.writerow(row)


def save_outputs(
    data: dict[str, Any],
    truth: dict[str, Any],
    metrics: dict[str, Any],
    output_dir: Path,
    stem: str = "adc_synthetic_modeling_data",
    write_csv: bool = True,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / f"{stem}.npz"
    csv_path = output_dir / f"{stem}.csv"
    json_path = output_dir / f"{stem}_summary.json"

    array_payload = {
        key: value for key, value in data.items() if isinstance(value, np.ndarray)
    }
    np.savez_compressed(
        npz_path,
        **array_payload,
        fs=np.asarray(data["fs"]),
        fin=np.asarray(data["fin"]),
        truth_json=np.asarray(json.dumps(_json_ready(truth), ensure_ascii=False)),
        metrics_json=np.asarray(json.dumps(_json_ready(metrics), ensure_ascii=False)),
    )

    files = {"npz": str(npz_path)}
    if write_csv:
        write_dataset_csv(data, csv_path)
        files["csv"] = str(csv_path)

    summary = {
        "description": "Modeling-only synthetic ADC data; no calibration applied.",
        "files": files,
        "data_columns": {
            "n": "global sample index",
            "sar_id": "four-way SAR channel id, n % 4",
            "mdac_id": "dual-MDAC id, n % 2",
            "C_raw": "2.5-bit Flash raw state",
            "D_DAC": "MDAC DAC feedback level in final-output LSB",
            "D_coarse": "coarse contribution in final-output LSB",
            "mdac_residue_gain": "MDAC residue gain; default is 4",
            "V_residue": "MDAC output residue, residue_gain*(x_mdac - D_DAC)",
            "F_residue_input_referred": "V_residue divided by residue_gain",
            "F_ideal": "ideal input-referred fine residue before mismatch",
            "F_raw": "signed 8-bit SAR fine code after mismatch/noise/quantization",
            "D_raw": "raw reconstructed code, D_coarse + F_raw",
            "D_ideal": "ideal full-rate code at the MDAC sampling instant",
            "t_mdac_group": "MDAC A/B group sampling time that forms the residue",
            "dt_mdac_group": "A/B MDAC aperture skew applied to this sample",
            "dt_sar": "SAR timing metadata only; not applied to residue sampling",
        },
        "truth": truth,
        "metrics": metrics,
    }
    json_path.write_text(
        json.dumps(_json_ready(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    files["summary_json"] = str(json_path)
    return files


def timing_case(case_name: str) -> TimingConfig:
    """Convenience timing presets for Flash-MDAC relative skew studies."""

    cases = {
        "case_0": TimingConfig(0.0, 0.0, 0.0, (0.0, 0.0, 0.0, 0.0)),
        "case_1": TimingConfig(
            dt_flash_common=0.0,
            dt_mdac_A=30.0e-12,
            dt_mdac_B=30.0e-12,
            dt_sar=(0.0, 0.0, 0.0, 0.0),
        ),
        "case_2": TimingConfig(
            dt_flash_common=0.0,
            dt_mdac_A=30.0e-12,
            dt_mdac_B=50.0e-12,
            dt_sar=(0.0, 0.0, 0.0, 0.0),
        ),
    }
    if case_name not in cases:
        raise ValueError(f"Unknown timing case {case_name!r}.")
    return cases[case_name]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate modeling-only synthetic data for the ADC architecture.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--prefix",
        default="adc_synthetic_modeling_data",
        help=(
            "Fixed output prefix. The script writes "
            "<prefix>_nomismatch.* and <prefix>_mismatch.*"
        ),
    )
    parser.add_argument("--N", type=int, default=2**16)
    parser.add_argument("--fs", type=float, default=1.0e9)
    parser.add_argument("--fin", type=float, default=499e6)
    parser.add_argument("--amplitude", type=float, default=511.0)
    parser.add_argument("--phase", type=float, default=0.31)
    parser.add_argument("--dc", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--N-fine", type=int, default=8)
    parser.add_argument(
        "--timing-case",
        choices=("case_0", "case_1", "case_2"),
        default="case_2",
        help="Timing case used for the mismatch dataset.",
    )
    parser.add_argument("--noise-std", type=float, default=0.08)
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Only write NPZ and JSON summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    signal = SignalConfig(
        fs=args.fs,
        fin=args.fin,
        amplitude=args.amplitude,
        phase=args.phase,
        dc=args.dc,
    )
    ideal_mismatch = MismatchConfig(
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
    common = dict(
        N=args.N,
        N_fine=args.N_fine,
        seed=args.seed,
        signal=signal,
        mdac=MdacConfig(residue_gain=4.0),
    )
    jobs = [
        (
            "nomismatch",
            DatasetConfig(
                **common,
                timing=timing_case("case_0"),
                mismatch=ideal_mismatch,
            ),
        ),
        (
            "mismatch",
            DatasetConfig(
                **common,
                timing=timing_case(args.timing_case),
                mismatch=MismatchConfig(noise_std=args.noise_std),
            ),
        ),
    ]

    print("Synthetic ADC modeling data generated.")
    for suffix, cfg in jobs:
        data, truth, metrics = build_synthetic_adc_dataset(cfg)
        files = save_outputs(
            data,
            truth,
            metrics,
            output_dir=args.output_dir,
            stem=f"{args.prefix}_{suffix}",
            write_csv=not args.no_csv,
        )
        print(f"\n[{suffix}]")
        for name, path in files.items():
            print(f"{name}: {path}")
        print(
            "Checks: "
            f"sar_id=n%4 {metrics['model']['sar_id_is_n_mod_4']}, "
            f"mdac_id=n%2 {metrics['model']['mdac_id_is_n_mod_2']}, "
            "D_raw=D_coarse+F_raw "
            f"{metrics['model']['D_raw_equals_D_coarse_plus_F_raw']}"
        )
        print(
            "Fine clip ratio: "
            f"{metrics['model']['fine_clip_ratio']:.6g}; "
            "raw RMS error vs D_ideal: "
            f"{metrics['error_vs_ideal']['rms_raw_lsb']:.6g} LSB"
        )


if __name__ == "__main__":
    main()
