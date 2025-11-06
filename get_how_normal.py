# -*- coding: utf-8 -*-
import numpy as np
import glob
import os

# ---------- 1. 加载原始数据 ----------
raw_path = "/root/data-tmp/new_data/processed_data/subj01/fsaverage_not_mean_1000/nsd_test_fmriavg_nsdgeneral_sub1.npy"
raw = np.load(raw_path).astype(np.float32)
print("Raw shape:", raw.shape)
print("Raw mean/std/min/max:", raw.mean(), raw.std(), raw.min(), raw.max())

# ---------- 2. 加载标准化后的前1000条 ----------
norm_dir = "/root/data-tmp/data/processed_data/subj01/fsaverage_not_mean_1000_new/test/"
norm_files = sorted(glob.glob(os.path.join(norm_dir, "sub1_fmri_test_*.npy")))

# 合并前1000条（如果确实有1000个文件）
norm = np.stack([np.load(f) for f in norm_files], axis=0).astype(np.float32)
print("Norm shape:", norm.shape)
print("Norm mean/std/min/max:", norm.mean(), norm.std(), norm.min(), norm.max())

# ---------- 3. 尝试识别标准化方式 ----------
# （1）Z-score 标准化（均值0、方差1）
if np.allclose(norm.mean(), 0, atol=1e-2) and np.allclose(norm.std(), 1, atol=1e-2):
    print("Likely Z-score standardization: (x - mean) / std")

# （2）Min–Max 归一化 [0,1]
elif np.all((0 <= norm) & (norm <= 1)):
    print("Likely Min–Max normalization: (x - min) / (max - min)")

# （3）[-1,1] 归一化
elif np.all((-1 <= norm) & (norm <= 1)):
    print("Likely [-1,1] normalization: 2*(x - min)/(max-min) - 1")

else:
    print("Not a standard normalization (possibly per-vertex or per-sample scaling)")

# ---------- 4. 验证是否为每条样本单独标准化 ----------
per_sample_means = np.mean(norm, axis=(1,2))
per_sample_stds  = np.std(norm, axis=(1,2))
print("Mean of per-sample means:", per_sample_means.mean())
print("Mean of per-sample stds:", per_sample_stds.mean())
