# -*- coding: utf-8 -*-
from torch.utils.data import Dataset



import os
import glob
import json
import functools
import numpy as np
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

# ========== 配置区域 ==========
SUBJECTS = list(range(1, 9))  # 1..8
SPLITS = ["train", "test"]    # 两个子集都处理
# 三组 (n_windows, window_size)
CONFIGS = [(1024, 64), (512, 128), (256, 256)]

# 窗口分组文件所在根目录，里边按 group_{n_windows}_{window_size}/windows_*.npy


# 输入：逐样本的标准化 fMRI
#   /root/data-tmp/new_data/processed_data/subjXX/fsaverage_not_mean_1000_new/{train|test}/*.npy
IN_ROOT = Path("/root/data-tmp/new_data/processed_data")

# 输出：拼接后的窗口特征
#   /root/data-tmp/new_data/processed_data/subjXX/fsaverage_not_mean_1000_new_{nwin}_{wsize}/{train|test}/*.npy
OUT_SUFFIX_TEMPLATE = "fsaverage_not_mean_1000_new_{nwin}_{wsize}"

ROOT = Path(__file__).parent           # 当前脚本所在目录
META = ROOT / "metadata"

# 顶点索引（含重复）的保存路径（全局）
L_IDX_PATH = META / "l_voxel_indices.npy"
R_IDX_PATH = META / "r_voxel_indices.npy"
# GROUPS_ROOT = Path("/root/data-tmp/data/groups")
# 进程并发数
MAX_WORKERS = max(1, os.cpu_count() or 8)


# ========== 核心函数 ==========
def build_g2row_maps(l_voxel_indices: np.ndarray, r_voxel_indices: np.ndarray):
    """
    为左右半球分别构建：全局顶点编号 -> 在拼接后的 fmri_roi 中的行号列表（保留重复）
    左半球行号 [0, len(l)-1]，右半球行号 [len(l), len(l)+len(r)-1]
    """
    g2row_l = defaultdict(list)
    for i, v in enumerate(l_voxel_indices):
        g2row_l[int(v)].append(i)

    g2row_r = defaultdict(list)
    offset = len(l_voxel_indices)
    for i, v in enumerate(r_voxel_indices):
        g2row_r[int(v)].append(offset + i)

    return g2row_l, g2row_r


def map_window_indices(windows: np.ndarray, g2row: dict) -> np.ndarray:
    """将窗口中的全局顶点编号映射为 fmri_roi 的行号（保留首次出现）"""
    mapped = np.zeros_like(windows)
    get = g2row.get
    for i in range(windows.shape[0]):
        for j in range(windows.shape[1]):
            v = int(windows[i, j])
            rows = get(v)
            if not rows:
                raise ValueError(f"[ERROR] vertex {v} not found in ROI indices")
            mapped[i, j] = rows[0]
    return mapped


def extract_fmri_features_from_windows(
    fmri_roi: np.ndarray,
    windows_l: np.ndarray,
    windows_r: np.ndarray,
    g2row_l: dict,
    g2row_r: dict,
) -> np.ndarray:
    """
    fmri_roi: (N_l + N_r, 4)  -> 每行: (x, y, z, beta)
    windows_*: (n_win_half, window_size) -> 顶点的全局编号
    g2row_*: 全局顶点编号 -> 行号列表（保留重复；我们取首次出现行号）
    返回: (n_windows_total, window_size, 5) -> (x,y,z,beta,hemi)
    """
    win_l_idx = map_window_indices(windows_l, g2row_l)  # (n_win_l, win_size)
    win_r_idx = map_window_indices(windows_r, g2row_r)  # (n_win_r, win_size)

    features_l = fmri_roi[win_l_idx]                    # (n_win_l, win_size, 4)
    features_r = fmri_roi[win_r_idx]                    # (n_win_r, win_size, 4)

    hemi_l = np.zeros((features_l.shape[0], features_l.shape[1], 1), dtype=np.float32)
    hemi_r = np.ones((features_r.shape[0], features_r.shape[1], 1), dtype=np.float32)

    features_l = np.concatenate([features_l, hemi_l], axis=-1)  # (n_win_l, win_size, 5)
    features_r = np.concatenate([features_r, hemi_r], axis=-1)  # (n_win_r, win_size, 5)

    return np.concatenate([features_l, features_r], axis=0)     # (n_windows, win_size, 5)


def process_single_file(
    path: str,
    windows_l: np.ndarray,
    windows_r: np.ndarray,
    g2row_l: dict,
    g2row_r: dict,
    save_path: Path,
):
    """处理单个样本 npy 文件 -> 保存窗口特征"""
    filename = os.path.basename(path)
    fmri_data = np.load(path)  # (V, 4)
    out = extract_fmri_features_from_windows(
        fmri_roi=fmri_data,
        windows_l=windows_l,
        windows_r=windows_r,
        g2row_l=g2row_l,
        g2row_r=g2row_r,
    )
    save_path.mkdir(parents=True, exist_ok=True)
    np.save((save_path / filename).as_posix(), out)
    return filename


def process_subject_split_with_config(
    sid: int,
    split: str,
    n_windows: int,
    window_size: int,
    windows_l: np.ndarray,
    windows_r: np.ndarray,
    g2row_l: dict,
    g2row_r: dict,
):
    """
    处理指定 subject 的 train/test，在给定 (n_windows, window_size) 配置下的全部样本。
    """
    subj_dir = IN_ROOT / f"subj{sid:02d}" / "fsaverage_not_mean_1000_new" / split
    npy_files = sorted(glob.glob((subj_dir / "*.npy").as_posix()))
    if not npy_files:
        print(f"[WARN] No files found: {subj_dir}")
        return

    out_suffix = OUT_SUFFIX_TEMPLATE.format(nwin=n_windows, wsize=window_size)
    save_path = IN_ROOT / f"subj{sid:02d}" / out_suffix / split
    print(f"[RUN] sub{sid:02d} {split} | nwin={n_windows}, wsize={window_size} | {len(npy_files)} files")

    work = functools.partial(
        process_single_file,
        windows_l=windows_l,
        windows_r=windows_r,
        g2row_l=g2row_l,
        g2row_r=g2row_r,
        save_path=save_path,
    )

    # 并发处理
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = [ex.submit(work, f) for f in npy_files]
        done = 0
        for fut in as_completed(futs):
            _ = fut.result()
            done += 1
            if done % 200 == 0 or done == len(npy_files):
                print(f"[PROG] sub{sid:02d} {split} {n_windows}x{window_size}: {done}/{len(npy_files)}")


def main():
    # 读入左右半球索引（含重复）
    if not L_IDX_PATH.exists() or not R_IDX_PATH.exists():
        raise FileNotFoundError("l_voxel_indices.npy / r_voxel_indices.npy not found.")
    l_voxel_indices = np.load(L_IDX_PATH.as_posix())
    r_voxel_indices = np.load(R_IDX_PATH.as_posix())
    g2row_l, g2row_r = build_g2row_maps(l_voxel_indices, r_voxel_indices)

    # 三个配置逐个处理（避免一次性加载过多 windows 占内存）
    for (n_windows, window_size) in CONFIGS:
        group_path = META / f"group_{n_windows}_{window_size}"
        windows_l_path = group_path / "windows_l.npy"
        windows_r_path = group_path / "windows_r.npy"
        if not windows_l_path.exists() or not windows_r_path.exists():
            raise FileNotFoundError(f"Missing windows files under: {group_path}")

        windows_l = np.load(windows_l_path.as_posix())
        windows_r = np.load(windows_r_path.as_posix())
        # 基本校验
        assert windows_l.ndim == 2 and windows_r.ndim == 2, "windows must be 2D arrays"
        assert windows_l.shape[1] == window_size and windows_r.shape[1] == window_size, \
            f"window_size mismatch for {n_windows}x{window_size}"

        for sid in SUBJECTS:
            for split in SPLITS:
                process_subject_split_with_config(
                    sid=sid,
                    split=split,
                    n_windows=n_windows,
                    window_size=window_size,
                    windows_l=windows_l,
                    windows_r=windows_r,
                    g2row_l=g2row_l,
                    g2row_r=g2row_r,
                )

    print("[DONE] All subjects, splits and configs processed.")


if __name__ == "__main__":
    main()




# import numpy as np
# import os
# import concurrent.futures
# import torch
# from collections import defaultdict
# from concurrent.futures import ProcessPoolExecutor
# import functools
# import glob
#
#
# def extract_fmri_features_from_windows(fmri_roi, windows_l, windows_r, l_voxel_indices, r_voxel_indices):
#     """
#     支持同一侧大脑 ROI 内重复顶点的窗口特征提取。
#
#     参数:
#         fmri_roi           : shape (N_l + N_r, 4)，拼接后的 fMRI 数据（保留重复行）
#         windows_size       : 每个窗口顶点数量
#         n_windows          : 总窗口数量（左右脑各一半）
#
#     返回:
#         features_all       : shape (n_windows, windows_size, 4)
#     """
#     # === Step 1: 加载数据 ===
#     # windows_l = np.load(f"windows_l_{n_windows}_{windows_size}.npy")  # 全局编号
#     # windows_r = np.load(f"windows_r_{n_windows}_{windows_size}.npy")
#     # l_voxel_indices = np.load("l_voxel_indices.npy")  # 含重复
#     # r_voxel_indices = np.load("r_voxel_indices.npy")
#
#     # === Step 2: 构造全局编号 → 多个行号（保留重复）===
#     g2row_l = defaultdict(list)
#     for i, v in enumerate(l_voxel_indices):
#         g2row_l[v].append(i)
#
#     g2row_r = defaultdict(list)
#     for i, v in enumerate(r_voxel_indices):
#         g2row_r[v].append(i + len(l_voxel_indices))  # 注意偏移！
#
#     # total_rows = sum(len(rows) for rows in g2row_l.values())
#     # print("g2row_l 中总行数：", total_rows)
#     # print("l_voxel_indices 总行数：", len(l_voxel_indices))
#     # === Step 3: 将窗口中全局编号映射为行号（按首次出现）===
#     def map_window_indices(windows, g2row):
#         mapped = np.zeros_like(windows)
#         for i in range(windows.shape[0]):
#             for j in range(windows.shape[1]):
#                 v = windows[i, j]
#                 rows = g2row.get(v)
#                 if not rows:
#                     raise ValueError(f"顶点 {v} 在 ROI 中找不到")
#                 mapped[i, j] = rows[0]  # 取第一次出现的行号
#         return mapped
#
#     windows_l_roi = map_window_indices(windows_l, g2row_l)
#     windows_r_roi = map_window_indices(windows_r, g2row_r)
#
#     # === Step 4: 提取特征 ===
#     features_l = fmri_roi[windows_l_roi]  # shape (n_win_l, win_size, 4)
#     features_r = fmri_roi[windows_r_roi]
#
#     # === Step 5: 添加 hemisphere 编码 ===
#     hemi_l = np.zeros((features_l.shape[0], features_l.shape[1], 1), dtype=np.float32)
#     hemi_r = np.ones((features_r.shape[0], features_r.shape[1], 1), dtype=np.float32)
#
#     features_l = np.concatenate([features_l, hemi_l], axis=-1)  # shape (n_win_l, win_size, 5)
#     features_r = np.concatenate([features_r, hemi_r], axis=-1)
#
#     # === Step 6: 拼接输出 ===
#     features_all = np.concatenate([features_l, features_r], axis=0)  # shape (n_windows, win_size, 5)
#     return features_all
#
#
# window_size = 64
# n_windows = 1024
# group_path = f"/root/data-tmp/data/groups/group_{n_windows}_{window_size}/"
# windows_l = np.load(group_path + "windows_l.npy")  # 全局编号
# windows_r = np.load(group_path + "windows_r.npy")
# l_voxel_indices = np.load("l_voxel_indices.npy")  # 含重复
# r_voxel_indices = np.load("r_voxel_indices.npy")
#
#
# # sub = 1
#
#
# def process_single_file(path, windows_l, windows_r, l_voxel_indices, r_voxel_indices, save_path):
#     print(path)
#     filename = os.path.basename(path)
#     fmri_data = np.load(path)
#     fmri_data = extract_fmri_features_from_windows(
#         fmri_data,
#         windows_l=windows_l,
#         windows_r=windows_r,
#         l_voxel_indices=l_voxel_indices,
#         r_voxel_indices=r_voxel_indices,
#     )
#     np.save(os.path.join(save_path, filename), fmri_data)
#
#
# for sub in range(8):
#     sub = sub + 1
#     folder_path = f"/root/data-tmp/new_data/processed_data/subj0{sub}/fsaverage_not_mean_1000_new/test"
#
#     # 获取所有 .npy 文件的完整路径列表
#     npy_files = sorted(glob.glob(os.path.join(folder_path, "*.npy")))
#     save_path = f"/root/data-tmp/new_data/processed_data/subj0{sub}/fsaverage_not_mean_1000_new_{n_windows}_{window_size}/test/"
#     os.makedirs(save_path, exist_ok=True)
#     with ProcessPoolExecutor(max_workers=128) as executor:
#         executor.map(
#             functools.partial(
#                 process_single_file,
#                 windows_l=windows_l,
#                 windows_r=windows_r,
#                 l_voxel_indices=l_voxel_indices,
#                 r_voxel_indices=r_voxel_indices,
#                 save_path=save_path,
#             ),
#             npy_files,
#         )

