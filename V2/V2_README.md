# V2 海杂波距离维预测 — 修复直线问题

## V1 的问题 → V2 的修复

| 问题 | V1 做法 | V2 修复 |
|------|---------|---------|
| **LSTM 输出直线** | `fc(last_hidden) → 8点并行` | Encoder-Decoder 自回归 + Teacher Forcing |
| **GRU 输出直线** | 解码器输入全零 | 种子投影 + Cross-Attention + 残差跳连 |
| **PatchTST 注意力退化** | 16个patch, d_model=64 | 63个patch, d_model=128, 3层 |
| **归一化被尖峰破坏** | 全局 MinMax | 分位数裁剪(1%~99%) MinMax |
| **损失函数冲突** | dynamic+std+grad 三重拉扯 | MSE + 可选梯度(0.2) |
| **rollout 训练过早** | rollout_steps=4 | rollout_steps=1，先单步 |
| **无 baseline 对照** | 无 | 4个 baseline (Last/Mean/Repeat/Linear) |
| **无早停** | 跑满 epochs | patience=15 早停 |

## 目录结构

```
V2/
├── common_data.py          # 共享：数据加载 + QuantileClipMinMaxScaler
├── forecast_utils.py       # 共享：简化损失 + rollout + 指标
├── baseline.py             # Baseline 对照
├── requirements.txt
├── LSTM/
│   ├── lstm.py             # Encoder-Decoder LSTM + Teacher Forcing
│   ├── config_lstm.py      # seq=64, pred=8, rollout=1, hidden=128
│   ├── dataset_amplitude.py
│   └── run_amplitude.py    # 训练脚本
├── GRU/
│   ├── SCPNet.py           # Cross-Attention + 种子 + 残差
│   ├── config_SCPNet.py    # seq=64, pred=8, rollout=1, hidden=128
│   ├── dataset_amplitude.py
│   └── run_amplitude.py
└── PatchTST/
    ├── PatchTST.py         # 模型入口（同V1结构）
    ├── config_amplitude.py # seq=128, pred=8, rollout=1, patch=4, stride=2, d=128
    ├── dataset_amplitude.py
    ├── run_amplitude.py
    └── layers/             # 从V1拷贝
```

## 运行顺序

### 第1步：先跑 Baseline（必须！）
```bash
cd V2
python baseline.py
```
记录 4 个 baseline 的 MAE(dB) 和 std_ratio。**如果神经网络不能显著优于 Linear baseline，说明问题不在模型。**

### 第2步：跑 LSTM（最简单，先验证架构修复有效）
```bash
cd V2/LSTM
python run_amplitude.py
```
关键指标：
- `val_std_ratio`：应 > 0.3（V1 约 0.0~0.1 即直线），若 > 0.5 说明架构修复成功
- `val_mae_db`：应低于 baseline 的 Last-Value

### 第3步：跑 GRU+Attention
```bash
cd V2/GRU
python run_amplitude.py
```

### 第4步：跑 PatchTST
```bash
cd V2/PatchTST
python run_amplitude.py
```

### 第5步：对比
所有模型跑完后，对比：
- 各模型 vs baseline 的 MAE(dB)
- 各模型的 std_ratio（>0.5 为非直线，>0.7 为良好）
- 单步预测 vs 4步 rollout 的退化程度

## 如果单步预测仍然是直线

检查以下可能性：
1. **数据本身**：画一条测试剖面的 dB 幅度图，看相邻 8 个点之间是否有变化
2. **归一化**：检查归一化后数据的范围是否在 [0, 1] 内且分布合理
3. **学习率**：尝试 lr=0.01（LSTM/GRU）或 lr=0.001（PatchTST）
4. **梯度损失**：尝试 grad_loss_weight=0.5（更强地约束斜率）

## 如果单步预测非直线但 rollout 仍是直线

这是正常的分布漂移问题，后续可加：
1. Scheduled Sampling（训练时逐步减少 teacher forcing 比例）
2. rollout_steps=2 的渐进训练
3.噪声注入（训练时给输入加小噪声模拟预测误差）
