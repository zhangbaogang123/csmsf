# -*- coding: utf-8 -*-

import numpy as np
import os
import matplotlib.pyplot as plt



# coding: utf-8
import os
import numpy as np

# -------------------- 配置 --------------------
BASE = "/your_data_dir/data/processed_data"
TEMPLATE = BASE + "/subj{sid:02d}/fsaverage_not_mean_1000"
SAVE_BASE = BASE + "/subj{sid:02d}/fsaverage_not_mean_1000_new"
STATS_PATH = "/metadata/fmri_data_stats.npz"  #
SUBS = [1,2,3,4,5,6,7,8]
ATOL = 1e-7
EPS = 1e-8   # 防止除0

def load_subject_arrays(sid):
    train_path = TEMPLATE.format(sid=sid) + f"/nsd_train_fmriavg_nsdgeneral_sub{sid}.npy"
    test_path  = TEMPLATE.format(sid=sid) + f"/nsd_test_fmriavg_nsdgeneral_sub{sid}.npy"
    train = np.load(train_path).astype(np.float32)
    test  = np.load(test_path).astype(np.float32)
    print(f"[LOAD] sub{sid}: train{train.shape}, test{test.shape}")
    return train, test

def compute_or_load_stats():
    # 如果已有统计文件，直接加载；否则用全部训练集合并计算并保存
    if os.path.exists(STATS_PATH):
        print(f"[STATS] load existing stats: {STATS_PATH}")
        data = np.load(STATS_PATH)
        return (data["xyz_mean"], data["xyz_std"], data["mean_signal"], data["std_signal"])

    print("[STATS] computing global stats from all subjects' TRAIN...")
    all_list = []
    for sid in SUBS:
        train, _ = load_subject_arrays(sid)
        all_list.append(train)      # 只用 train 计算统计量
        del train
    all_fmri = np.concatenate(all_list, axis=0)  # (N_all_train, V, 4)
    del all_list

    xyz = all_fmri[:, :, :3].astype(np.float32)
    signal = all_fmri[:, :, 3].astype(np.float32)

    xyz_mean = xyz.mean(axis=(0,1), keepdims=True)
    xyz_std  = xyz.std(axis=(0,1), keepdims=True)
    signal_mean = signal.mean(axis=(0,1), keepdims=True)
    signal_std  = signal.std(axis=(0,1), keepdims=True)

    # 防止出现 0 方差
    xyz_std  = np.where(np.abs(xyz_std)  < EPS, 1.0, xyz_std).astype(np.float32)
    signal_std = np.where(np.abs(signal_std) < EPS, 1.0, signal_std).astype(np.float32)

    os.makedirs(os.path.dirname(STATS_PATH), exist_ok=True)
    np.savez(STATS_PATH,
             xyz_mean=xyz_mean.astype(np.float32),
             xyz_std=xyz_std.astype(np.float32),
             mean_signal=signal_mean.astype(np.float32),
             std_signal=signal_std.astype(np.float32))
    print(f"[STATS] saved -> {STATS_PATH}")
    del all_fmri, xyz, signal
    return xyz_mean, xyz_std, signal_mean, signal_std

def standardize_fmri_data(fmri_data, xyz_mean, xyz_std, signal_mean, signal_std):
    # fmri_data: (N, V, 4) ; 前3列(x,y,z) 后1列(beta)
    xyz = fmri_data[:, :, :3].astype(np.float32)
    sig = fmri_data[:, :, 3].astype(np.float32)

    xyz_norm = (xyz - xyz_mean) / (xyz_std + EPS)
    sig_norm = (sig - signal_mean) / (signal_std + EPS)

    return np.concatenate([xyz_norm, sig_norm[..., None]], axis=-1).astype(np.float32)

def save_per_sample(arr, out_dir, prefix):
    os.makedirs(out_dir, exist_ok=True)
    for i, sample in enumerate(arr):
        fn = f"{prefix}_{i}.npy"
        np.save(os.path.join(out_dir, fn), sample)
        if (i % 500) == 0:
            print(f"[SAVE] {prefix} idx={i}/{len(arr)}")

# -------------------- 主流程 --------------------
if __name__ == "__main__":
    # 1) 统计量
    xyz_mean, xyz_std, mean_signal, std_signal = compute_or_load_stats()
    print("[STATS] xyz_mean/std:", xyz_mean.shape, xyz_std.shape, " signal_mean/std:", mean_signal.shape, std_signal.shape)

    # 2) 逐受试者标准化并保存
    for sid in SUBS:
        train, test = load_subject_arrays(sid)
        train_n = standardize_fmri_data(train, xyz_mean, xyz_std, mean_signal, std_signal)
        test_n  = standardize_fmri_data(test,  xyz_mean, xyz_std, mean_signal, std_signal)

        save_train_dir = SAVE_BASE.format(sid=sid) + "/train"
        save_test_dir  = SAVE_BASE.format(sid=sid) + "/test"
        os.makedirs(save_train_dir, exist_ok=True)
        os.makedirs(save_test_dir,  exist_ok=True)

        # 逐样本保存
        save_per_sample(train_n, save_train_dir, prefix=f"sub{sid}_fmri_train")
        save_per_sample(test_n,  save_test_dir,  prefix=f"sub{sid}_fmri_test")


        # np.save(os.path.join(SAVE_BASE.format(sid=sid), f"sub{sid}_train_standardized.npy"), train_n)
        # np.save(os.path.join(SAVE_BASE.format(sid=sid), f"sub{sid}_test_standardized.npy"),  test_n)

        print(f"[DONE] sub{sid}: train{train_n.shape}, test{test_n.shape}")











