# -*- coding: utf-8 -*-
import os
import numpy as np
from PIL import Image

SUBJECTS = list(range(1, 9))
BASE_FMT = "/your_data-dir/data/processed_data/subj{sid:02d}/fsaverage_not_mean_1000"

OUT_DIR_TRAIN = "train_images"
OUT_DIR_TEST = "test_images"
EXT = ".png"
OVERWRITE = False


def ensure_uint8_rgb(img_np: np.ndarray) -> np.ndarray:
    """Convert any input array to HxWx3 uint8 RGB."""
    arr = img_np

    # Convert CxHxW → HxWxC
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[1] > 8 and arr.shape[2] > 8:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]

    # Normalize dtype and value range
    if arr.dtype in (np.float32, np.float64):
        m = float(arr.max()) if arr.size else 1.0
        if m <= 1.0 + 1e-6:
            arr = (arr * 255.0).clip(0, 255)
        else:
            arr = arr.clip(0, 255)
        arr = arr.astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = arr.clip(0, 255).astype(np.uint8)

    # Convert grayscale or handle channels
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3:
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        elif arr.shape[-1] == 3:
            pass
        elif arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        else:
            raise ValueError(f"Unsupported channel count: {arr.shape}")
    else:
        raise ValueError(f"Unsupported ndim: {arr.ndim}")
    return arr


def dump_split(npy_path: str, out_dir: str, overwrite: bool = False) -> int:
    """Convert one npy file to individual PNG images."""
    if not os.path.isfile(npy_path):
        print(f"[WARN] File not found: {npy_path}")
        return 0

    os.makedirs(out_dir, exist_ok=True)
    arr = np.load(npy_path, mmap_mode="r")
    assert arr.ndim in (3, 4), f"Expected shape (N,H,W[,C]) or (N,C,H,W), got {arr.shape}"
    N = arr.shape[0]
    print(f"[INFO] {npy_path} -> N={N}, output: {out_dir}")

    saved = 0
    for i in range(N):
        fname = f"{i}{EXT}"
        fpath = os.path.join(out_dir, fname)
        if (not overwrite) and os.path.exists(fpath):
            continue
        try:
            img = ensure_uint8_rgb(arr[i])
            Image.fromarray(img, mode="RGB").save(fpath)
            saved += 1
            if saved % 1000 == 0:
                print(f"[PROGRESS] saved {saved}/{N} (latest: {fname})")
        except Exception as e:
            print(f"[ERROR] image {i} failed: {e}")
    print(f"[DONE] saved {saved}/{N} images -> {out_dir}")
    return saved


if __name__ == "__main__":
    total = 0
    for sub in SUBJECTS:
        base = BASE_FMT.format(sid=sub)
        train_path = os.path.join(base, f"nsd_train_stim_sub{sub}.npy")
        test_path = os.path.join(base, f"nsd_test_stim_sub{sub}.npy")
        out_train = os.path.join(base, OUT_DIR_TRAIN)
        out_test = os.path.join(base, OUT_DIR_TEST)

        total += dump_split(train_path, out_train, overwrite=OVERWRITE)
        total += dump_split(test_path, out_test, overwrite=OVERWRITE)

    print(f"\n[SUMMARY] total images saved: {total}")
