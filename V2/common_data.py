"""V2 共享数据工具 — 增加分位数裁剪归一化"""
from pathlib import Path
import struct
import zlib

import numpy as np


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    v1_root = Path(__file__).resolve().parent
    project_root = Path(__file__).resolve().parents[2]
    for root in (Path.cwd(), v1_root, project_root):
        candidate = (root / path).resolve()
        if candidate.exists():
            return candidate
    if path.name:
        for root in (project_root, v1_root.parent, v1_root):
            matches = list(root.rglob(path.name))
            if matches:
                return matches[0]
    return path


# ── 归一化器 ──────────────────────────────────────────────

class MinMaxScaler1D:
    """原始 MinMax（V1 兼容）"""
    def fit(self, x):
        x = np.asarray(x, dtype=np.float64)
        self.min_ = float(np.nanmin(x))
        self.max_ = float(np.nanmax(x))
        self.scale_ = self.max_ - self.min_
        if self.scale_ == 0:
            self.scale_ = 1.0
        return self

    def transform(self, x):
        return (np.asarray(x, dtype=np.float64) - self.min_) / self.scale_

    def inverse_transform(self, x):
        return np.asarray(x, dtype=np.float64) * self.scale_ + self.min_


class QuantileClipMinMaxScaler:
    """分位数裁剪 + MinMax：先 clip 到 [lower_q, upper_q] 分位数，再 MinMax

    海杂波 dB 域常有极端尖峰（海尖峰效应），全局 min/max 会被拉宽，
    导致有效动态范围被压缩到极窄区间。裁剪后归一化更稳健。
    """
    def __init__(self, lower_quantile=0.01, upper_quantile=0.99):
        self.lower_q = lower_quantile
        self.upper_q = upper_quantile

    def fit(self, x):
        x = np.asarray(x, dtype=np.float64)
        self.clip_min_ = float(np.quantile(x, self.lower_q))
        self.clip_max_ = float(np.quantile(x, self.upper_q))
        x_clipped = np.clip(x, self.clip_min_, self.clip_max_)
        self.min_ = float(np.nanmin(x_clipped))
        self.max_ = float(np.nanmax(x_clipped))
        self.scale_ = self.max_ - self.min_
        if self.scale_ == 0:
            self.scale_ = 1.0
        return self

    def transform(self, x):
        x = np.asarray(x, dtype=np.float64)
        x = np.clip(x, self.clip_min_, self.clip_max_)
        return (x - self.min_) / self.scale_

    def inverse_transform(self, x):
        return np.asarray(x, dtype=np.float64) * self.scale_ + self.min_


class StandardScalerLastDim:
    """沿最后一维标准化（复数 IQ 用）"""
    def fit(self, x):
        x = np.asarray(x, dtype=np.float64)
        flat = x.reshape(-1, x.shape[-1])
        self.mean_ = flat.mean(axis=0, keepdims=True)
        self.std_ = flat.std(axis=0, keepdims=True)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, x):
        x = np.asarray(x, dtype=np.float64)
        return (x - self.mean_) / self.std_

    def inverse_transform(self, x):
        x = np.asarray(x, dtype=np.float64)
        return x * self.std_ + self.mean_


# ── MAT 文件读取（与 V1 相同）──────────────────────────────

def _read_tag(buf, off):
    raw = struct.unpack_from("<I", buf, off)[0]
    small_size = raw >> 16
    small_type = raw & 0xFFFF
    if small_size:
        return small_type, small_size, off + 4, off + 8
    size = struct.unpack_from("<I", buf, off + 4)[0]
    payload_off = off + 8
    next_off = payload_off + size + ((8 - (size % 8)) % 8)
    return raw, size, payload_off, next_off


def _parse_v5_matrix(buf):
    off = 0
    _, _, _, off = _read_tag(buf, off)
    _, size, payload_off, off = _read_tag(buf, off)
    dims = struct.unpack_from("<" + "i" * (size // 4), buf, payload_off)
    _, size, payload_off, off = _read_tag(buf, off)
    name = buf[payload_off:payload_off + size].decode("utf-8", errors="replace")
    _, size, payload_off, off = _read_tag(buf, off)
    real = np.frombuffer(buf, dtype="<f8", count=size // 8, offset=payload_off).copy()
    imag = None
    if off + 8 <= len(buf):
        _, size, payload_off, _ = _read_tag(buf, off)
        if size == real.size * 8:
            imag = np.frombuffer(buf, dtype="<f8", count=size // 8, offset=payload_off).copy()
    arr = real if imag is None else real + 1j * imag
    arr = arr.reshape(tuple(dims), order="F")
    return name, arr


def _load_mat_v5(path):
    data = Path(path).read_bytes()
    off = 128
    arrays = {}
    while off + 8 <= len(data):
        typ, size, payload_off, next_off = _read_tag(data, off)
        if typ == 0 or size == 0:
            break
        payload = data[payload_off:payload_off + size]
        if typ == 15:
            payload = zlib.decompress(payload)
            typ2, size2, payload_off2, _ = _read_tag(payload, 0)
            if typ2 == 14:
                name, arr = _parse_v5_matrix(payload[payload_off2:payload_off2 + size2])
                arrays[name] = arr
        elif typ == 14:
            name, arr = _parse_v5_matrix(payload)
            arrays[name] = arr
        off = next_off
    return arrays


def load_mat_variable(file_path, preferred_key="clutter_pc"):
    file_path = resolve_path(file_path)
    arrays = None
    try:
        import scipy.io
        raw = scipy.io.loadmat(str(file_path))
        arrays = {k: v for k, v in raw.items() if not k.startswith("__")}
    except Exception:
        arrays = _load_mat_v5(file_path)

    if preferred_key and preferred_key in arrays:
        arr = arrays[preferred_key]
    else:
        candidates = [(k, v) for k, v in arrays.items() if isinstance(v, np.ndarray)]
        complex_candidates = [(k, v) for k, v in candidates if np.iscomplexobj(v)]
        if complex_candidates:
            arr = complex_candidates[0][1]
        elif candidates:
            arr = candidates[0][1]
        else:
            raise KeyError(f"No ndarray variable found in {file_path}")

    arr = np.asarray(arr).squeeze()
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got shape {arr.shape}")
    return arr


def load_complex_profiles(args):
    key = getattr(args, "mat_key", "clutter_pc")
    raw = load_mat_variable(args.file_path, key)

    profile_start = getattr(args, "profile_start", getattr(args, "pulse_start", 0))
    profile_end = getattr(args, "profile_end", getattr(args, "pulse_end", None))
    range_cell_limit = getattr(args, "range_cell_limit", None)

    if profile_end is None or profile_end > raw.shape[0]:
        profile_end = raw.shape[0]
    if range_cell_limit is None or range_cell_limit > raw.shape[1]:
        range_cell_limit = raw.shape[1]

    profiles = raw[profile_start:profile_end, :range_cell_limit]
    if profiles.size == 0:
        raise ValueError(
            f"Empty data after slicing: profiles {profile_start}:{profile_end}, "
            f"range_cell_limit={range_cell_limit}, raw_shape={raw.shape}"
        )
    return profiles


def split_profile_indices(n_profiles, train_ratio=0.7, val_ratio=0.15, seed=2026):
    if n_profiles < 3:
        raise ValueError("At least 3 profiles are required for train/val/test splitting.")

    rng = np.random.default_rng(seed)
    indices = np.arange(n_profiles)
    rng.shuffle(indices)

    n_train = max(1, int(round(n_profiles * train_ratio)))
    n_val = max(1, int(round(n_profiles * val_ratio)))
    if n_train + n_val >= n_profiles:
        n_train = max(1, n_profiles - 2)
        n_val = 1

    train_idx = np.sort(indices[:n_train])
    val_idx = np.sort(indices[n_train:n_train + n_val])
    test_idx = np.sort(indices[n_train + n_val:])
    return train_idx, val_idx, test_idx
