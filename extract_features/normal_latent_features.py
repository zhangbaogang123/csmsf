# -*- coding: utf-8 -*-
import os
import numpy as np
from tqdm import tqdm

subjects = [f"subj0{i}" for i in range(1, 9)]
input_root = "/your_data_dir/data/processed_data"
output_root = "/your_data_dir/data/processed_data"
latent_shape = (4, 64, 64)

# === 1. 收集所有训练 latent 计算全局 min/max ===
all_flat_latents = []

print("正在收集所有训练 latent 数据以计算全局 min/max...")

for subject in subjects:
    train_dir = os.path.join(input_root, subject, "train_latents_blurred")
    latent_files = [f for f in os.listdir(train_dir) if f.endswith(".npy")]

    for fname in tqdm(latent_files, desc=f"Loading {subject}"):
        latent = np.load(os.path.join(train_dir, fname))  # shape: (4, 64, 64)
        all_flat_latents.append(latent.flatten())         # shape: (16384,)

all_flat_latents = np.stack(all_flat_latents, axis=0)  # shape: (N_all, 16384)
min_vals = all_flat_latents.min(axis=0, keepdims=True)  # shape: (1, 16384)
max_vals = all_flat_latents.max(axis=0, keepdims=True)
range_vals = max_vals - min_vals
range_vals[range_vals == 0] = 1.0  # 防止除以 0

print("全局 min/max 计算完成")

# === 2. 保存 min/max 用于测试/反归一化 ===
np.save(os.path.join(output_root, "min_vals.npy"), min_vals)
np.save(os.path.join(output_root, "max_vals.npy"), max_vals)

# === 3. 对训练和测试 latent 统一归一化并保存 ===
for subject in subjects:
    for split in ["train_latents_blurred", "test_latents_blurred"]:
        input_dir = os.path.join(input_root, subject, split)
        output_dir = os.path.join(output_root, subject, split+"_minmax")
        os.makedirs(output_dir, exist_ok=True)

        latent_files = [f for f in os.listdir(input_dir) if f.endswith(".npy")]

        for fname in tqdm(latent_files, desc=f"Normalizing {subject}/{split}"):
            path = os.path.join(input_dir, fname)
            latent = np.load(path).flatten()
            latent_norm = (latent - min_vals) / range_vals
            latent_norm = latent_norm.reshape(latent_shape)
            np.save(os.path.join(output_dir, fname), latent_norm)

print(" 全部训练集和测试集归一化完成，结果保存在:")
print(f" {output_root}")

