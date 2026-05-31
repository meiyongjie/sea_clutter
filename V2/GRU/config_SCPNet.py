"""V2 GRU 配置 — 单步训练 + 加大容量"""
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
        self.seq_len = 64
        self.pred_len = 8
        # ★ V2：rollout_steps=1，先单步
        self.rollout_steps = 1

        # ── GRU + Attention ──
        self.enc_in = 1
        self.hidden_size = 128       # V1=64, 加大
        self.num_layers = 2
        self.n_heads = 4
        self.teacher_force_ratio = 0.5

        # ── 训练 ──
        self.do_train = True
        self.save_dir = './checkpoints'
        self.model_name = 'gru_attention_amplitude_v2.pth'
        self.model_path = os.path.join(self.save_dir, self.model_name)
        self.batch_size = 128
        self.learning_rate = 0.001
        self.epochs = 80
        self.grad_clip = 1.0
        self.grad_loss_weight = 0.2
        self.patience = 15
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


args = Configs()
