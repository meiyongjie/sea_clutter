"""V2 冒烟测试 — 快速验证三个模型能跑通 + 输出非直线

只跑 5 epoch，小数据量，确认：
1. 数据加载正常
2. 模型前向/反向正常
3. 输出 std_ratio > 0（非直线）
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common_data import QuantileClipMinMaxScaler, load_complex_profiles, split_profile_indices
from forecast_utils import AmplitudeForecastLoss, amplitude_metrics, merge_metric_sums


def quick_data(seq_len=64, pred_len=8, n_profiles=10, n_range=2000):
    """用少量数据快速测试"""
    class Args:
        file_path = '../../数据集/tets/echo.mat'
        mat_key = 'clutter_pc'
        profile_start = 0
        profile_end = 40
        range_cell_limit = 11530
        split_seed = 2026
        train_ratio = 0.7
        val_ratio = 0.15
        amp_eps = 1e-12

    profiles = load_complex_profiles(Args())
    # 只取前 n_range 个距离单元加速
    profiles = profiles[:, :n_range]
    amp_db = 20.0 * np.log10(np.abs(profiles) + 1e-12)

    train_idx, val_idx, test_idx = split_profile_indices(amp_db.shape[0], 0.7, 0.15, 2026)
    scaler = QuantileClipMinMaxScaler(0.01, 0.99).fit(amp_db[train_idx].reshape(-1, 1))
    data_scaled = scaler.transform(amp_db).astype(np.float32)

    from torch.utils.data import DataLoader, TensorDataset

    def make_loader(idx):
        xs, ys = [], []
        d = data_scaled[idx]
        for p in range(d.shape[0]):
            for s in range(d.shape[1] - seq_len - pred_len + 1):
                xs.append(d[p, s:s+seq_len])
                ys.append(d[p, s+seq_len:s+seq_len+pred_len])
        if not xs:
            return None
        x_t = torch.from_numpy(np.array(xs)[:500][:, :, np.newaxis])  # 最多500样本
        y_t = torch.from_numpy(np.array(ys)[:500][:, :, np.newaxis])
        return DataLoader(TensorDataset(x_t, y_t), batch_size=64, shuffle=True, drop_last=True)

    return make_loader(train_idx), make_loader(val_idx), scaler


def smoke_test_rnn(name, model_cls, config_cls, train_loader, val_loader, scaler):
    """测试 LSTM / GRU"""
    args = config_cls()
    args.epochs = 5
    args.device = torch.device('cpu')

    model = model_cls(args)
    criterion = AmplitudeForecastLoss(args)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    tf_ratio = getattr(args, 'teacher_force_ratio', 0.5)
    print(f"\n{'='*50}")
    print(f"Smoke Test: {name}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(1, 6):
        model.train()
        total_loss = 0
        for bx, by in train_loader:
            optimizer.zero_grad()
            output = model.forward_train(bx, by, tf_ratio=tf_ratio)
            target = by[:, :args.pred_len, :]
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        # 验证
        model.eval()
        metric_sums = []
        with torch.no_grad():
            for bx, by in val_loader:
                # 自回归推理
                output = model(bx)
                target = by[:, :args.pred_len, :]
                metric_sums.append(amplitude_metrics(output, target, scaler))

        val_m = merge_metric_sums(metric_sums)
        avg_loss = total_loss / max(len(train_loader), 1)
        print(f"  Epoch {epoch}/5 | loss={avg_loss:.6f} | mae_db={val_m['mae_db']:.4f} | std_ratio={val_m['std_ratio']:.3f}")

    return val_m


def smoke_test_patchtst(train_loader, val_loader, scaler):
    """测试 PatchTST"""
    sys.path.insert(0, str(Path(__file__).resolve().parent / "PatchTST"))
    from PatchTST import Model
    from config_amplitude import Configs

    args = Configs()
    args.epochs = 5
    args.device = torch.device('cpu')
    # PatchTST 的 seq_len=128，需要重建数据
    # 为了冒烟，临时改 seq_len
    args.seq_len = 64  # 冒烟用64，与数据一致
    args.patch_len = 4
    args.stride = 2

    model = Model(args)
    criterion = AmplitudeForecastLoss(args)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    print(f"\n{'='*50}")
    print(f"Smoke Test: PatchTST V2")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(1, 6):
        model.train()
        total_loss = 0
        for bx, by in train_loader:
            optimizer.zero_grad()
            output = model(bx)
            pred = output[:, -args.pred_len:, :]
            target = by[:, :args.pred_len, :]
            loss = criterion(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        metric_sums = []
        with torch.no_grad():
            for bx, by in val_loader:
                output = model(bx)
                pred = output[:, -args.pred_len:, :]
                target = by[:, :args.pred_len, :]
                metric_sums.append(amplitude_metrics(pred, target, scaler))

        val_m = merge_metric_sums(metric_sums)
        avg_loss = total_loss / max(len(train_loader), 1)
        print(f"  Epoch {epoch}/5 | loss={avg_loss:.6f} | mae_db={val_m['mae_db']:.4f} | std_ratio={val_m['std_ratio']:.3f}")

    return val_m


if __name__ == "__main__":
    print("Loading data (subset)...")
    train_loader, val_loader, scaler = quick_data()

    if train_loader is None or val_loader is None:
        print("ERROR: Could not create data loaders")
        sys.exit(1)

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # ── LSTM ──
    sys.path.insert(0, str(Path(__file__).resolve().parent / "LSTM"))
    from lstm import SeaClutterLSTM
    from config_lstm import Configs as LSTMConfig
    lstm_result = smoke_test_rnn("LSTM Encoder-Decoder V2", SeaClutterLSTM, LSTMConfig, train_loader, val_loader, scaler)

    # ── GRU ──
    sys.path.insert(0, str(Path(__file__).resolve().parent / "GRU"))
    from SCPNet import SCPNet
    from config_SCPNet import Configs as GRUConfig
    gru_result = smoke_test_rnn("GRU+CrossAttention V2", SCPNet, GRUConfig, train_loader, val_loader, scaler)

    # ── PatchTST ──
    patchtst_result = smoke_test_patchtst(train_loader, val_loader, scaler)

    # ── 总结 ──
    print(f"\n{'='*60}")
    print(f"SMOKE TEST SUMMARY (5 epochs, subset data)")
    print(f"{'='*60}")
    print(f"  Baseline Mean:  MAE=4.36 dB  std_ratio=0.000")
    print(f"  Baseline Linear: MAE=4.37 dB  std_ratio=0.020")
    print(f"  LSTM V2:        MAE={lstm_result['mae_db']:.4f} dB  std_ratio={lstm_result['std_ratio']:.3f}")
    print(f"  GRU V2:         MAE={gru_result['mae_db']:.4f} dB  std_ratio={gru_result['std_ratio']:.3f}")
    print(f"  PatchTST V2:    MAE={patchtst_result['mae_db']:.4f} dB  std_ratio={patchtst_result['std_ratio']:.3f}")
    print()

    # 判定
    all_pass = True
    for name, result in [("LSTM", lstm_result), ("GRU", gru_result), ("PatchTST", patchtst_result)]:
        mae_ok = result['mae_db'] < 4.36  # 优于 Mean baseline
        std_ok = result['std_ratio'] > 0.02  # 优于 Linear baseline
        status = "PASS" if (mae_ok or std_ok) else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  {name}: {status}  (mae_ok={mae_ok}, std_ok={std_ok})")

    if all_pass:
        print("\n✅ All models pass smoke test! Ready for full training.")
    else:
        print("\n⚠️  Some models fail. Need further debugging.")
