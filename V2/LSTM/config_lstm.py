"""V2 LSTM 配置 — 单步训练 + 教师强制"""
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
        # ★ V2 关键改动：rollout_steps=1，先单步训练
        self.rollout_steps = 1

        # ── LSTM ──
        self.enc_in = 1
        self.hidden_size = 128       # V1=64, 加大容量
        self.num_layers = 2
        self.dropout = 0.1
        self.teacher_force_ratio = 0.5  # Teacher Forcing 概率

        # ── 训练 ──
        self.do_train = True
        self.save_dir = './checkpoints'
        self.model_name = 'lstm_amplitude_v2.pth'
        self.model_path = os.path.join(self.save_dir, self.model_name)
        self.batch_size = 128
        self.learning_rate = 0.001
        self.epochs = 80            # V1=30, 加长训练
        self.grad_clip = 1.0
        self.grad_loss_weight = 0.2  # 轻量梯度损失
        self.patience = 15           # 早停耐心
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


args = Configs()
