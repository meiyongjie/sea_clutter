"""V2 预测工具 — 简化损失 + 单步优先训练

V1 的问题：dynamic_loss_weight 让模型过度关注不可预测的海尖峰，
std_loss 和 grad_loss 相互冲突，导致模型在"平滑 vs 尖锐"之间摇摆。

V2 策略：
  - 基础损失：MSE（简单稳定）
  - 可选梯度损失：帮助匹配局部斜率，权重宜小（0.1~0.3）
  - 去掉 dynamic weighting 和 std loss
  - 训练先用 rollout_steps=1 单步，确认非直线后再加 rollout
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── 模型推理接口 ──────────────────────────────────────────

def model_forecast(model, context, args):
    """单次前向，取最后 pred_len 步作为预测"""
    output = model(context)
    return output[:, -args.pred_len:, :]


def rollout_forecast(model, context, args, rollout_steps=None):
    """闭环递归预测（仅用于评估，训练阶段不用）"""
    steps = max(1, int(getattr(args, "rollout_steps", 1))) if rollout_steps is None else max(1, int(rollout_steps))
    preds = []
    current = context
    for _ in range(steps):
        pred = model_forecast(model, current, args)
        preds.append(pred)
        keep = max(args.seq_len - pred.shape[1], 0)
        if keep > 0:
            current = torch.cat([current[:, -keep:, :], pred], dim=1)
        else:
            current = pred[:, -args.seq_len:, :]
    return torch.cat(preds, dim=1)


def forecast_target(batch_y, args, rollout_steps=None):
    steps = max(1, int(getattr(args, "rollout_steps", 1))) if rollout_steps is None else max(1, int(rollout_steps))
    horizon = args.pred_len * steps
    if batch_y.shape[1] < horizon:
        raise ValueError(f"Need target length {horizon}, got {batch_y.shape[1]}.")
    return batch_y[:, :horizon, :]


# ── 损失函数 ──────────────────────────────────────────────

class AmplitudeForecastLoss(nn.Module):
    """V2 简化损失：MSE + 可选梯度损失

    去掉了 V1 的 dynamic_loss_weight 和 std_loss_weight。
    这两个在 V1 中导致模型过度关注离群点（海尖峰），
    而海尖峰恰恰是最不可预测的部分。
    """
    def __init__(self, args):
        super().__init__()
        self.grad_weight = float(getattr(args, "grad_loss_weight", 0.0))

    def forward(self, pred, target):
        # 基础 MSE
        loss = F.mse_loss(pred, target)

        # 可选：梯度损失 — 鼓励预测的局部斜率与真实一致
        if self.grad_weight > 0.0 and pred.shape[1] > 1:
            pred_grad = pred[:, 1:, :] - pred[:, :-1, :]
            target_grad = target[:, 1:, :] - target[:, :-1, :]
            loss = loss + self.grad_weight * F.mse_loss(pred_grad, target_grad)

        return loss


# ── 评估指标 ──────────────────────────────────────────────

def amplitude_metrics(pred, target, scaler):
    """计算 dB 域 MAE 和 std_ratio（检测直线问题的关键指标）"""
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    pred_db = scaler.inverse_transform(pred_np.reshape(-1, 1)).reshape(pred_np.shape)
    target_db = scaler.inverse_transform(target_np.reshape(-1, 1)).reshape(target_np.shape)

    abs_err = np.abs(pred_db - target_db)
    pred_flat = pred_db.reshape(pred_db.shape[0], -1)
    target_flat = target_db.reshape(target_db.shape[0], -1)
    pred_std = pred_flat.std(axis=1)
    target_std = target_flat.std(axis=1)
    std_ratio = pred_std / (target_std + 1e-6)

    return {
        "mae_db_sum": float(abs_err.sum()),
        "mae_db_count": int(abs_err.size),
        "std_ratio_sum": float(std_ratio.sum()),
        "std_ratio_count": int(std_ratio.size),
        "mse_sum": float(F.mse_loss(pred, target, reduction="sum").item()),
        "mse_count": int(target.numel()),
    }


def merge_metric_sums(items):
    merged = {
        "mae_db_sum": 0.0, "mae_db_count": 0,
        "std_ratio_sum": 0.0, "std_ratio_count": 0,
        "mse_sum": 0.0, "mse_count": 0,
    }
    for item in items:
        for key in merged:
            merged[key] += item[key]
    return {
        "mae_db": float(merged["mae_db_sum"] / max(merged["mae_db_count"], 1)),
        "std_ratio": float(merged["std_ratio_sum"] / max(merged["std_ratio_count"], 1)),
        "mse": float(merged["mse_sum"] / max(merged["mse_count"], 1)),
    }
