"""V2 Baseline 对照 — 不跑 baseline 就无法判断模型是否真的学到了东西

4 个 baseline：
1. Last-Value：用输入最后一个值作为所有预测值
2. Mean：用输入均值作为所有预测值
3. Linear：线性回归外推
4. Repeat：重复输入最后 pred_len 个值

如果神经网络不能显著优于这些 baseline，说明预测任务本身就不成立。
"""
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
from common_data import QuantileClipMinMaxScaler, load_complex_profiles, split_profile_indices


def make_windows(data_2d, seq_len, pred_len):
    """从 2D 数据中提取所有 (input, target) 窗口"""
    n_profiles, n_range = data_2d.shape
    xs, ys = [], []
    for p in range(n_profiles):
        for start in range(n_range - seq_len - pred_len + 1):
            xs.append(data_2d[p, start:start + seq_len])
            ys.append(data_2d[p, start + seq_len:start + seq_len + pred_len])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def baseline_last_value(x, pred_len):
    """用输入最后一个值作为所有预测值"""
    last = x[:, -1:]
    return np.repeat(last, pred_len, axis=1)


def baseline_mean(x, pred_len):
    """用输入均值作为所有预测值"""
    mean = x.mean(axis=1, keepdims=True)
    return np.repeat(mean, pred_len, axis=1)


def baseline_repeat(x, pred_len):
    """重复输入最后 pred_len 个值"""
    return x[:, -pred_len:]


def baseline_linear(x, pred_len):
    """线性回归外推"""
    # x: [N, seq_len] 或 [N, seq_len, 1]
    if x.ndim == 3:
        x = x[:, :, 0]
    seq_len = x.shape[1]
    t = np.arange(seq_len, dtype=np.float64)
    t_mean = t.mean()
    t_std = t.std() + 1e-8
    t_norm = (t - t_mean) / t_std  # [seq_len]

    # 向量化最小二乘: slope = sum((x_i - mean) * t_i) / sum(t_i^2)
    x_d = x.astype(np.float64)  # [N, seq_len]
    x_mean = x_d.mean(axis=1, keepdims=True)  # [N, 1]
    slope = ((x_d - x_mean) * t_norm[None, :]).sum(axis=1, keepdims=True) / (t_norm ** 2).sum()  # [N, 1]

    # 外推
    t_future = np.arange(seq_len, seq_len + pred_len, dtype=np.float64)
    t_future_norm = (t_future - t_mean) / t_std  # [pred_len]
    pred = x_mean + slope * t_future_norm[None, :]  # [N, pred_len]
    return pred.astype(np.float32)


def evaluate_baseline(name, pred_fn, x_test, y_test, scaler, pred_len):
    pred = pred_fn(x_test, pred_len)
    # 反归一化到 dB
    pred_db = scaler.inverse_transform(pred.reshape(-1, 1)).reshape(pred.shape)
    target_db = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(y_test.shape)

    mae = np.abs(pred_db - target_db).mean()
    mse = ((pred_db - target_db) ** 2).mean()

    # std_ratio
    pred_std = pred_db.reshape(pred_db.shape[0], -1).std(axis=1)
    target_std = target_db.reshape(target_db.shape[0], -1).std(axis=1)
    std_ratio = (pred_std / (target_std + 1e-6)).mean()

    print(f"  {name:20s} | MAE={mae:.4f} dB | MSE={mse:.6f} | std_ratio={std_ratio:.3f}")
    return {"name": name, "mae_db": float(mae), "mse": float(mse), "std_ratio": float(std_ratio)}


if __name__ == "__main__":
    # ── 加载数据 ──
    class DataArgs:
        file_path = '../../数据集/tets/echo.mat'
        mat_key = 'clutter_pc'
        profile_start = 0
        profile_end = 40
        range_cell_limit = 11530
        split_seed = 2026
        train_ratio = 0.7
        val_ratio = 0.15
        amp_eps = 1e-12

    args = DataArgs()
    profiles = load_complex_profiles(args)
    amp_db = 20.0 * np.log10(np.abs(profiles) + 1e-12)

    train_idx, val_idx, test_idx = split_profile_indices(
        amp_db.shape[0], 0.7, 0.15, 2026
    )

    scaler = QuantileClipMinMaxScaler(0.01, 0.99).fit(amp_db[train_idx].reshape(-1, 1))
    data_scaled = scaler.transform(amp_db).astype(np.float32)

    seq_len = 64
    pred_len = 8

    print(f"Building windows: seq_len={seq_len}, pred_len={pred_len}")
    x_test, y_test = make_windows(data_scaled[test_idx], seq_len, pred_len)
    print(f"Test windows: {x_test.shape[0]}")

    # ── 跑 baseline ──
    print(f"\n{'='*60}")
    print(f"Baseline Results (seq={seq_len}, pred={pred_len})")
    print(f"{'='*60}")

    results = []
    results.append(evaluate_baseline("Last-Value", baseline_last_value, x_test, y_test, scaler, pred_len))
    results.append(evaluate_baseline("Mean", baseline_mean, x_test, y_test, scaler, pred_len))
    results.append(evaluate_baseline("Repeat", baseline_repeat, x_test, y_test, scaler, pred_len))
    results.append(evaluate_baseline("Linear", baseline_linear, x_test, y_test, scaler, pred_len))

    # 保存
    os.makedirs("./checkpoints", exist_ok=True)
    with open("./checkpoints/baseline_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to ./checkpoints/baseline_results.json")
    print(f"\n★ 如果神经网络不能显著优于 Last-Value 和 Linear，说明预测任务本身就不成立。")
