"""V2 SCPNet — 修复直线问题的 GRU+Attention

V1 的问题：
  解码器输入全零 → 无起点信息 → 输出自然塌缩为均值
  Self-Attention Q=K=V 全来自同一个全零驱动的解码器 → 无区分度

V2 修复：
  1. 解码器输入：用编码器最后一步的输出投影作为种子（而非全零）
  2. 残差跳连：编码器特征直接跳连到输出，保证至少能回传输入末值
  3. Cross-Attention：Q 来自解码器，K/V 来自编码器 → 有信息来源
  4. 加大 hidden_size 到 128
"""
import torch
import torch.nn as nn


class SCPNet(nn.Module):
    """Sea Clutter Prediction Network V2"""

    def __init__(self, configs):
        super(SCPNet, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.hidden_size = configs.hidden_size
        self.num_layers = configs.num_layers
        self.n_heads = configs.n_heads

        # ── Encoder ──
        self.encoder = nn.GRU(
            input_size=configs.enc_in,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )

        # ── 种子投影：把编码器最后隐状态投影为解码器第一步输入 ──
        self.seed_fc = nn.Linear(self.hidden_size, configs.enc_in)

        # ── Decoder ──
        self.decoder = nn.GRU(
            input_size=configs.enc_in,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )

        # ── Cross-Attention：Q=解码器, K/V=编码器 ──
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.n_heads,
            batch_first=True,
        )

        # ── 输出：解码器特征 + 注意力特征 + 残差跳连 ──
        # 拼接 [dec_out, attn_out, enc_last] → 3 * hidden_size
        self.linear = nn.Linear(self.hidden_size * 3, configs.enc_in)

    def forward(self, x):
        """推理模式：自回归解码

        Args:
            x: [batch, seq_len, 1]
        Returns:
            [batch, pred_len, 1]
        """
        batch_size = x.size(0)

        # ── Encode ──
        enc_out, enc_hidden = self.encoder(x)
        # enc_out: [batch, seq_len, hidden_size]
        # enc_hidden: [num_layers, batch, hidden_size]

        # ── 种子：用编码器最后隐状态生成解码器第一步输入 ──
        seed = torch.sigmoid(self.seed_fc(enc_hidden[-1]))  # [batch, 1]

        # ── 自回归 Decode ──
        dec_input = seed.unsqueeze(1)  # [batch, 1, 1]
        outputs = []
        h = enc_hidden  # 用编码器状态初始化解码器

        for t in range(self.pred_len):
            dec_out, h = self.decoder(dec_input, h)
            # dec_out: [batch, 1, hidden_size]

            # Cross-Attention：解码器问编码器
            attn_out, _ = self.cross_attn(
                dec_out, enc_out, enc_out
            )  # [batch, 1, hidden_size]

            # 拼接 + 输出
            enc_last = enc_hidden[-1].unsqueeze(1).expand(-1, 1, -1)  # [batch, 1, hidden_size]
            fused = torch.cat([dec_out, attn_out, enc_last], dim=-1)  # [batch, 1, hidden*3]
            step_pred = self.linear(fused)  # [batch, 1, 1]
            outputs.append(step_pred)

            dec_input = step_pred.detach()  # 下一步用预测值

        return torch.cat(outputs, dim=1)  # [batch, pred_len, 1]

    def forward_train(self, x, target, tf_ratio=0.5):
        """训练模式：Teacher Forcing + Cross-Attention"""
        batch_size = x.size(0)

        # ── Encode ──
        enc_out, enc_hidden = self.encoder(x)

        # ── 种子 ──
        seed = torch.sigmoid(self.seed_fc(enc_hidden[-1]))
        dec_input = seed.unsqueeze(1)

        outputs = []
        h = enc_hidden

        for t in range(self.pred_len):
            dec_out, h = self.decoder(dec_input, h)
            attn_out, _ = self.cross_attn(dec_out, enc_out, enc_out)
            enc_last = enc_hidden[-1].unsqueeze(1).expand(-1, 1, -1)
            fused = torch.cat([dec_out, attn_out, enc_last], dim=-1)
            step_pred = self.linear(fused)
            outputs.append(step_pred)

            # Teacher Forcing
            if target is not None and torch.rand(1).item() < tf_ratio:
                dec_input = target[:, t:t + 1, :]
            else:
                dec_input = step_pred.detach()

        return torch.cat(outputs, dim=1)
