"""V2 PatchTST 配置 — 加大上下文 + 更多 patch + 更大 d_model

V1 的问题：
  seq_len=64, patch_len=8, stride=4 → 只有 16 个 patch
  d_model=64, n_heads=4 → 每个 head 仅 16 维
  → 注意力机制几乎退化成全连接，无法捕捉长程依赖

V2 修复：
  seq_len=128 → 更多上下文信息
  patch_len=4, stride=2 → 63 个 patch（注意力有足够对象）
  d_model=128 → 每个 head 32 维，分辨率提升
"""
import os
import torch


class Configs:
    def __init__(self):
        # ── 数据 ──
        self.file_path = '../../数据集/tets/echo.mat'
        self.mat_key = 'clutter_pc'
        self.profile_start = 0
        self.profile_end = 40
        self.range_cell_limit = 11530
        self.task_mode = 'forecast'
        self.split_seed = 2026
        self.train_ratio = 0.7
        self.val_ratio = 0.15
        self.amp_eps = 1e-12

        # ── 预测窗口 ──
        self.seq_len = 128          # ★ V1=64, 加大上下文
        self.pred_len = 8
        self.rollout_steps = 1      # ★ 单步训练

        # ── PatchTST 模型 ──
        self.enc_in = 1
        self.d_model = 128          # ★ V1=64
        self.n_heads = 4            # 每个 head 32 维
        self.e_layers = 3           # ★ V1=2, 加深
        self.d_ff = 256             # ★ V1=128
        self.dropout = 0.2          # ★ V1=0.05, 加正则防过拟合
        self.fc_dropout = 0.1
        self.head_dropout = 0.1
        self.individual = False     # 共享头
        self.patch_len = 4          # ★ V1=8, 更细粒度
        self.stride = 2             # ★ V1=4, 更密采样
        self.padding_patch = 'end'

        # RevIN
        self.revin = True
        self.affine = True
        self.subtract_last = False

        # 分解（关闭，单通道不需要）
        self.decomposition = False
        self.kernel_size = 25

        # ── 训练 ──
        self.do_train = True
        self.save_dir = './checkpoints'
        self.model_name = 'patchtst_amplitude_v2.pth'
        self.model_path = os.path.join(self.save_dir, self.model_name)
        self.batch_size = 128
        self.learning_rate = 0.0005  # ★ 比 RNN 小，Transformer 需要更保守的 lr
        self.epochs = 80
        self.grad_clip = 1.0
        self.grad_loss_weight = 0.2
        self.patience = 15
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


args = Configs()
