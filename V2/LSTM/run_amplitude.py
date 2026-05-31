"""V2 LSTM 训练 — 单步训练 + Teacher Forcing + 早停

核心改动：
1. 训练用 forward_train (teacher forcing)，推理用 forward (自回归)
2. rollout_steps=1，不搞闭环，先让单步预测出非直线
3. 简化损失：MSE + 轻量梯度损失
4. 早停：val mae_db 连续 patience 轮不降则停
"""
import json
import os
from pathlib import Path
import sys

import torch
import torch.optim as optim

sys.path.append(str(Path(__file__).resolve().parents[1]))
from forecast_utils import (
    AmplitudeForecastLoss,
    amplitude_metrics,
    forecast_target,
    merge_metric_sums,
    rollout_forecast,
)

from config_lstm import args
from dataset_amplitude import get_data
from lstm import SeaClutterLSTM


def evaluate(args, model, loader, scaler, rollout_steps=None):
    model.eval()
    metric_sums = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(args.device)
            batch_y = batch_y.to(args.device)
            output = rollout_forecast(model, batch_x, args, rollout_steps)
            target = forecast_target(batch_y, args, rollout_steps)
            metric_sums.append(amplitude_metrics(output, target, scaler))
    return merge_metric_sums(metric_sums)


def train(args, train_loader, val_loader, scaler):
    model = SeaClutterLSTM(args).to(args.device)
    criterion = AmplitudeForecastLoss(args)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.save_dir, exist_ok=True)
    best_val = float("inf")
    patience_counter = 0
    patience = getattr(args, "patience", 15)
    history = []

    tf_ratio = getattr(args, "teacher_force_ratio", 0.5)

    print(f"Training LSTM (Encoder-Decoder) on {args.device}")
    print(f"seq_len={args.seq_len}, pred_len={args.pred_len}, rollout_steps={args.rollout_steps}")
    print(f"teacher_force_ratio={tf_ratio}, hidden_size={args.hidden_size}")

    for epoch in range(1, args.epochs + 1):
        # ── 训练 ──
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(args.device)
            batch_y = batch_y.to(args.device)

            optimizer.zero_grad()
            # ★ Teacher Forcing 训练
            output = model.forward_train(batch_x, batch_y, tf_ratio=tf_ratio)
            target = batch_y[:, :args.pred_len, :]
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        train_loss /= max(len(train_loader), 1)

        # ── 验证（纯自回归，无 teacher forcing） ──
        val_metrics = evaluate(args, model, val_loader, scaler, rollout_steps=1)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_mae_db": val_metrics["mae_db"],
            "val_std_ratio": val_metrics["std_ratio"],
        })

        # ── 保存最优 ──
        if val_metrics["mae_db"] < best_val:
            best_val = val_metrics["mae_db"]
            torch.save(model.state_dict(), args.model_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"Epoch {epoch:03d}/{args.epochs} | train={train_loss:.6f} | "
                f"val_mae_db={val_metrics['mae_db']:.4f} | "
                f"val_std_ratio={val_metrics['std_ratio']:.3f} | "
                f"best_val={best_val:.4f}"
            )

        # ── 早停 ──
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch} (patience={patience})")
            break

    return model, history


if __name__ == "__main__":
    train_loader, val_loader, test_loader, scaler, _, split = get_data(args)
    model, history = train(args, train_loader, val_loader, scaler)

    # 加载最优模型测试
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))

    # 单步测试
    test_direct = evaluate(args, model, test_loader, scaler, rollout_steps=1)
    # 4步 rollout 测试（仅评估，不参与训练）
    test_rollout = evaluate(args, model, test_loader, scaler, rollout_steps=4)

    result = {
        "model": "LSTM_EncoderDecoder_v2",
        "seq_len": args.seq_len,
        "pred_len": args.pred_len,
        "rollout_steps": args.rollout_steps,
        "hidden_size": args.hidden_size,
        "teacher_force_ratio": args.teacher_force_ratio,
        "grad_loss_weight": args.grad_loss_weight,
        "test_direct": test_direct,
        "test_rollout_4step": test_rollout,
        "history": history,
    }
    result_path = os.path.join(args.save_dir, "lstm_amplitude_v2_results.json")
    os.makedirs(args.save_dir, exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Test Direct:  MAE={test_direct['mae_db']:.4f} dB  std_ratio={test_direct['std_ratio']:.3f}")
    print(f"Test Rollout: MAE={test_rollout['mae_db']:.4f} dB  std_ratio={test_rollout['std_ratio']:.3f}")
    print(f"Saved to {result_path}")
