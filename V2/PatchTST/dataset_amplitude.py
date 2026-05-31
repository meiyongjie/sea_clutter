"""V2 PatchTST 数据集 — 分位数裁剪归一化 + seq_len=128"""
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.append(str(Path(__file__).resolve().parents[1]))
from common_data import QuantileClipMinMaxScaler, load_complex_profiles, split_profile_indices


class RangeAmplitudeDataset(Dataset):
    def __init__(self, data_2d, args):
        self.data = np.asarray(data_2d, dtype=np.float32)
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.rollout_steps = max(1, int(getattr(args, "rollout_steps", 1)))
        self.future_len = self.pred_len * self.rollout_steps
        self.task_mode = getattr(args, "task_mode", "forecast")

        self.n_profiles, self.n_range = self.data.shape
        if self.task_mode == "forecast":
            self.samples_per_profile = self.n_range - self.seq_len - self.future_len + 1
        else:
            raise ValueError(f"Unsupported task_mode: {self.task_mode}")

        if self.samples_per_profile <= 0:
            raise ValueError(
                f"Range length {self.n_range} too short for seq={self.seq_len}, future={self.future_len}."
            )

    def __len__(self):
        return self.n_profiles * self.samples_per_profile

    def __getitem__(self, index):
        profile_idx = index // self.samples_per_profile
        range_start = index % self.samples_per_profile
        x0 = range_start
        x1 = x0 + self.seq_len
        y1 = x1 + self.future_len
        seq_x = self.data[profile_idx, x0:x1]
        seq_y = self.data[profile_idx, x1:y1]
        return torch.from_numpy(seq_x[:, None]), torch.from_numpy(seq_y[:, None])


def get_data(args):
    profiles = load_complex_profiles(args)
    print(f"Loaded complex profiles: {profiles.shape} [profiles, range]")

    amp_db = 20.0 * np.log10(np.abs(profiles) + getattr(args, "amp_eps", 1e-12))

    train_idx, val_idx, test_idx = split_profile_indices(
        amp_db.shape[0],
        getattr(args, "train_ratio", 0.7),
        getattr(args, "val_ratio", 0.15),
        getattr(args, "split_seed", 2026),
    )

    scaler = QuantileClipMinMaxScaler(
        lower_quantile=0.01, upper_quantile=0.99
    ).fit(amp_db[train_idx].reshape(-1, 1))
    data_scaled = scaler.transform(amp_db).astype(np.float32)

    train_set = RangeAmplitudeDataset(data_scaled[train_idx], args)
    val_set = RangeAmplitudeDataset(data_scaled[val_idx], args)
    test_set = RangeAmplitudeDataset(data_scaled[test_idx], args)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    print(f"Split profiles: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    print(f"Windows per profile: {train_set.samples_per_profile}")
    print(f"Future length per window: {train_set.future_len}")
    return train_loader, val_loader, test_loader, scaler, data_scaled, (train_idx, val_idx, test_idx)
