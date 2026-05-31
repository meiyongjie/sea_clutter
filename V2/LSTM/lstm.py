"""V2 LSTM — Encoder-Decoder 自回归结构

V1 的问题：
  fc(last_hidden) → 8 个点并行输出
  → 没有时序结构，8 个点自然塌缩为均值（直线）

V2 修复：
  Encoder LSTM 提取输入序列特征
  Decoder LSTM 逐步自回归解码，每步以上一步输出为输入
  → 每步条件不同 → 输出自然有起伏
  → 训练时用 Teacher Forcing 加速收敛
"""
import torch
import torch.nn as nn


class SeaClutterLSTM(nn.Module):
    def __init__(self, configs):
        super(SeaClutterLSTM, self).__init__()

        self.pred_len = configs.pred_len
        self.hidden_size = configs.hidden_size
        self.num_layers = configs.num_layers

        # ── Encoder ──
        self.encoder = nn.LSTM(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=configs.dropout if self.num_layers > 1 else 0.0,
        )

        # ── Decoder（自回归） ──
        self.decoder = nn.LSTM(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=configs.dropout if self.num_layers > 1 else 0.0,
        )

        # ── 输出映射 ──
        self.output_fc = nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        """推理模式：自回归解码，用自身预测作为下一步输入

        Args:
            x: [batch, seq_len, 1]
        Returns:
            [batch, pred_len, 1]
        """
        # 编码
        _, (h, c) = self.encoder(x)

        # 逐步解码
        dec_input = x[:, -1:, :]  # 用输入最后一个值作为种子
        outputs = []
        for _ in range(self.pred_len):
            dec_out, (h, c) = self.decoder(dec_input, (h, c))
            step_pred = self.output_fc(dec_out)  # [batch, 1, 1]
            outputs.append(step_pred)
            dec_input = step_pred.detach()  # 截断梯度，下一步用预测值

        return torch.cat(outputs, dim=1)  # [batch, pred_len, 1]

    def forward_train(self, x, target, tf_ratio=0.5):
        """训练模式：Teacher Forcing

        以 tf_ratio 概率用真实值作为下一步输入，
        否则用模型自身预测。这比纯自回归训练更稳定，
        同时比纯 teacher forcing 更适应推理时的分布。

        Args:
            x:      [batch, seq_len, 1]  输入序列
            target: [batch, pred_len, 1] 目标序列
            tf_ratio: 使用真实值的概率
        Returns:
            [batch, pred_len, 1]
        """
        _, (h, c) = self.encoder(x)

        dec_input = x[:, -1:, :]  # 种子：输入最后一个值
        outputs = []
        for t in range(self.pred_len):
            dec_out, (h, c) = self.decoder(dec_input, (h, c))
            step_pred = self.output_fc(dec_out)
            outputs.append(step_pred)

            # Teacher Forcing：随机选择用真实值还是预测值
            if target is not None and torch.rand(1).item() < tf_ratio:
                dec_input = target[:, t:t + 1, :]
            else:
                dec_input = step_pred.detach()

        return torch.cat(outputs, dim=1)  # [batch, pred_len, 1]
