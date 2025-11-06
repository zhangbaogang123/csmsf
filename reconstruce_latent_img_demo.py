# -*- coding: utf-8 -*-
import os
from diffusers.models import AutoencoderKL
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from pl_model_latent import FineTunedModel
from pl_model_clip import Pct
from torch.cuda.amp import autocast
sub =7
device_num = 0

img_model_dir="clip_img_ckpt_dir/xx.ckpt"
latent_model_dir="vae latent ckpt dir/xx.ckpt"
save_num="v9121"
use_flattent=False
def custom_sort_fmri(file_path):
    parts = file_path.split('/')
    file_name = parts[-1]  # 获取文件名，例如 'sub2_fmri_test_0.npy'
    sub_num = int(file_name.split('_')[0][3:])  # 提取 'sub' 后面的数字
    test_num = int(file_name.split('_')[-1].split('.')[0])  # 提取 'test' 后面的数字
    return (sub_num, test_num)


multi_scale_size = [(256,256),(512,128),(1024, 64)]
checkpoint_path = img_model_dir
pct_model = Pct.load_from_checkpoint(
    checkpoint_path=checkpoint_path,
    tf_drop=0.2, lr=2e-4, weight_decay=0.05, multi_scale_size=multi_scale_size, use_sub_info=True, type="img",map_location=lambda storage, loc: storage.cuda(0))
model = FineTunedModel.load_from_checkpoint(checkpoint_path=latent_model_dir, pct=pct_model, tf_drop=0.3, lr=1e-4,
                                            weight_decay=0.5, multi_scale_size=multi_scale_size,map_location=lambda storage, loc: storage.cuda(0),
                                            use_sub_info=True)
model.eval()
device = f'cuda:{device_num}'
model.to(device)
fmri_dir = "your_data_dir/processed_data/subj{:02d}/fsaverage_not_mean_1000_new/test/".format(
    sub)
fmri_file_list = os.listdir(fmri_dir)
fmri_file_list = [os.path.join(fmri_dir, file_name) for file_name in fmri_file_list]
fmri_file_list = sorted(fmri_file_list, key=custom_sort_fmri)
latent_dir = "/your_data_dir/processed_data/subj{:02d}/test_latents_blurred/".format(
    sub)
latent_file_list = os.listdir(latent_dir)
latent_file_list = [os.path.join(latent_dir, file_name) for file_name in latent_file_list]
latent_file_list = sorted(latent_file_list, key=custom_sort_fmri)
autoenc = AutoencoderKL(
    down_block_types=['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
    up_block_types=['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'],
    block_out_channels=[128, 256, 512, 512],
    layers_per_block=2,
    sample_size=256
)
autoenc.to(f'cuda:{device_num}')
autoenc.load_state_dict(torch.load('your_model_dir/models/sd_image_var_autoenc.pth'))
autoenc.requires_grad_(False)
autoenc.eval()
min_vals = torch.tensor(np.load("min_vals.npy"), dtype=torch.float32).to(device)
max_vals = torch.tensor(np.load("max_vals.npy"), dtype=torch.float32).to(device)
range_vals = max_vals - min_vals
range_vals[range_vals == 0] = 1.0  # 防止除以 0

# reshape 为 (1, 4, 64, 64)，便于广播
min_vals = min_vals.view(1, 4, 64, 64)
range_vals = range_vals.view(1, 4, 64, 64)
out_file = "your_out_dir/results/latent/sub_0{}/{}".format(sub,save_num)
os.makedirs(out_file, exist_ok=True)
def farthest_point_sampling_indices(num_points, n_centers, initial_index=0):
    selected = [initial_index]
    distances = np.full((num_points,), np.inf)
    for _ in range(1, n_centers):
        last_center = selected[-1]
        distances = np.minimum(distances, np.abs(np.arange(num_points) - last_center))
        next_center = np.argmax(distances)
        selected.append(next_center)
    return np.array(sorted(selected))


def generate_windows_around_centers(center_indices, window_size, total_points):
    half_size = window_size // 2
    windows = []
    for c in center_indices:
        start = max(0, c - half_size)
        end = min(total_points, c + half_size)
        if end - start < window_size:
            if start == 0:
                end = min(total_points, window_size)
            else:
                start = max(0, total_points - window_size)
        window = np.arange(start, end)
        if len(window) == window_size:
            windows.append(window)
    return np.stack(windows)  # shape (n_windows, window_size)

def extract_fmri_features_simple(fmri_data, windows):
    return fmri_data[windows]  # shape: (n_windows, window_size, 4)
split="test"

if use_flattent:

    rng = np.random.RandomState(seed=42)
    perm = rng.permutation(22492)  # 打乱后的顶点顺序
    inv_perm = np.argsort(perm)
    windows_by_scale = {}
    for (n, w) in multi_scale_size:
        centers = farthest_point_sampling_indices(num_points=22492, n_centers=n, initial_index=0)
        shuffled_centers = perm[centers]
        windows = generate_windows_around_centers(shuffled_centers, window_size=w, total_points=22492)
        windows_by_scale[f"{n}_{w}"] = windows
else:
    fmri_datas_dict = {f"{n}_{w}": [] for (n, w) in multi_scale_size}
    for (n_win, win_size) in multi_scale_size:
        key = f"{n_win}_{win_size}"
        scale_tag = f"fsaverage_not_mean_1000_new_{n_win}_{win_size}"
        fmri_dir = f"your_data_dir/processed_data/subj{sub:02d}/{scale_tag}/{split}/"
        fmri_files = [os.path.join(fmri_dir, f) for f in os.listdir(fmri_dir)]
        fmri_datas_dict[key].extend(fmri_files)
    for k in fmri_datas_dict:
        fmri_datas_dict[k] = sorted(fmri_datas_dict[k], key=custom_sort_fmri)

index = 0
for i in range(982):
    fmri_multi_scale = {}
    path = ""
    print("--------------------------------------------------------------------------")
    if use_flattent:
        print("use_flattent")
        fmri_dir = f"/your_data_dir/processed_data/subj0{sub}/fsaverage_not_mean_1000_new/{split}/sub{sub}_fmri_test_{i}.npy"
        fmri = np.load(fmri_dir)
        fmri = fmri[perm]
        fmri_multi_scale = {}
        print(fmri_dir)
        for (n, w) in multi_scale_size:
            key = f"{n}_{w}"
            windows = windows_by_scale[key]
            features = extract_fmri_features_simple(fmri, windows)
            fmri_multi_scale[key] = torch.from_numpy(features).float().to(device).unsqueeze(0)
    else:
        for key in fmri_datas_dict:
            fmri_path = fmri_datas_dict[key][i]
            print(fmri_path)
            path = fmri_path
            with open(fmri_path, "rb") as f:
                fmri_multi_scale[key] = torch.from_numpy(np.load(f, allow_pickle=False)).float().to(device).unsqueeze(0)

    with torch.no_grad():
        sub_tensor = torch.tensor(sub).to(device).unsqueeze(0)
        with autocast(dtype=torch.bfloat16):
            result = model(fmri_multi_scale,sub_tensor)

        restored = result * range_vals + min_vals
        reconst = autoenc.decode(restored / 0.18215).sample / 2 + 0.5
        if reconst.ndim == 4:
            reconst = reconst[0]
        to_pil_image = transforms.ToPILImage()
        img = to_pil_image(reconst)
        save_path = out_file + "/{}.png".format(i)
        img.save(save_path)
        index = index + 1

