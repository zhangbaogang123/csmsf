# -*- coding: utf-8 -*-
import sys
sys.path.append('/root/brain-diffuser/versatile_diffusion')
sys.path.append('/root/brain-diffuser/gnn')
import os
import argparse
# import os
import os.path as osp
import PIL
from PIL import Image

from pathlib import Path
import numpy as np
import numpy.random as npr

import torch
import torchvision.transforms as tvtrans
from lib.cfg_helper import model_cfg_bank
from lib.model_zoo import get_model
from lib.model_zoo.ddim_vd import DDIMSampler_VD
from lib.experiments.sd_default import color_adjust, auto_merge_imlist
from pl_model_tranfomer_con_loss import Pct as pct_img
from pl_model_text import FineTunedModel as pct_text
from torch.cuda.amp import autocast
# 不使用vdvae图片时，Strength=0.9，ddim_steps=60
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub", help="Subject Number", default=1)
parser.add_argument("-diff_str", "--diff_str", help="Diffusion Strength", default=0.75)
parser.add_argument("-mix_str", "--mix_str", help="Mixing Strength", default=0.5)  # text
parser.add_argument("-device", "--device", help="Mdevice_num", default=0)
parser.add_argument("--use_prompt", action="store_true", help="Use prompt guidance")
parser.add_argument("--use_latent", action="store_true", help="Use latent guidance")
parser.add_argument("-img_model_dir", "--img_model_dir", help="img_model_dir", default=0)
parser.add_argument("-text_model_dir", "--text_model_dir", help="text_model_dir", default=0)
args = parser.parse_args()
device = int(args.device)
use_prompt = args.use_prompt
use_latent = args.use_latent
use_flattent=False
img_model_dir=args.img_model_dir
text_model_dir=args.text_model_dir
# use_prompt = True
# use_latent = True
print("use_prompt:", use_prompt)
print("use_latent:", use_latent)
os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(device)
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# img_model_dir="/root/data-tmp/brain-diffuser/gnn/transfomer/2025-06-11_01-11-52/epoch=348-val_loss=0.73.ckpt"
# text_model_dir="/root/data-tmp/brain-diffuser/gnn/transfomer/2025-06-11_19-41-51/epoch=66-val_loss=0.35.ckpt"

latent_num="v9121"
text_num="9120"
img_num="9091"
multi_scale_size = [(256,256),(512,128),(1024, 64)]
print(sys.path)
sub = int(args.sub)
device_num = device

print("sub:", sub)
print("device:", device)
assert sub in [1, 2, 5, 7]
strength = float(args.diff_str)
mixing = float(args.mix_str)


def regularize_image(x):
    BICUBIC = PIL.Image.Resampling.BICUBIC
    if isinstance(x, str):
        x = Image.open(x).resize([512, 512], resample=BICUBIC)
        x = tvtrans.ToTensor()(x)
    elif isinstance(x, PIL.Image.Image):
        x = x.resize([512, 512], resample=BICUBIC)
        x = tvtrans.ToTensor()(x)
    elif isinstance(x, np.ndarray):
        x = PIL.Image.fromarray(x).resize([512, 512], resample=BICUBIC)
        x = tvtrans.ToTensor()(x)
    elif isinstance(x, torch.Tensor):
        pass
    else:
        assert False, 'Unknown image type'

    assert (x.shape[1] == 512) & (x.shape[2] == 512), \
        'Wrong image size'
    return x


cfgm_name = 'vd_noema'
sampler = DDIMSampler_VD
pth = '/root/data-tmp/brain-diffuser/versatile_diffusion/pretrained/vd-four-flow-v1-0-fp16-deprecated.pth'
cfgm = model_cfg_bank()(cfgm_name)
net = get_model()(cfgm)
sd = torch.load(pth, map_location='cpu')
net.load_state_dict(sd, strict=False)
print("net.clip.cuda(device_num)", device_num)
net.clip.cuda(device_num)
net.autokl.cuda(device_num)
net.cuda(device_num)
sampler = sampler(net)

batch_size = 1
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
def custom_sort_fmri(file_path):
    parts = file_path.split('/')
    file_name = parts[-1]  # 获取文件名，例如 'sub2_fmri_test_0.npy'
    sub_num = int(file_name.split('_')[0][3:])  # 提取 'sub' 后面的数字
    test_num = int(file_name.split('_')[-1].split('.')[0])  # 提取 'test' 后面的数字
    return (sub_num, test_num)
split = "test"
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
        fmri_dir = f"/root/data-tmp/data/processed_data/subj{sub:02d}/{scale_tag}/{split}/"
        fmri_files = [os.path.join(fmri_dir, f) for f in os.listdir(fmri_dir)]
        fmri_datas_dict[key].extend(fmri_files)
    for k in fmri_datas_dict:
        fmri_datas_dict[k] = sorted(fmri_datas_dict[k], key=custom_sort_fmri)
img_model = pct_img.load_from_checkpoint(
    checkpoint_path=img_model_dir,
    tf_drop=0.2, lr=2e-4, weight_decay=0.05, multi_scale_size=multi_scale_size, use_sub_info=True, type="img",map_location=torch.device("cuda:0"))
text_model = pct_text.load_from_checkpoint(
    checkpoint_path=text_model_dir,
    pct=img_model,
    tf_drop=0.3, lr=2e-4, weight_decay=0.1, multi_scale_size=multi_scale_size,
    use_sub_info=True,map_location=torch.device("cuda:0"))
img_model.to(f'cuda:{device_num}')
img_model.eval()
text_model.to(f'cuda:{device_num}')
text_model.eval()
device = f'cuda:{device_num}'
pred_vision = None
pred_text = None
with torch.no_grad():
    # inputs = fmri_tensor.to(device)
    # inputs = inputs.permute(0, 2, 1).float()
    for i in range(982):
        fmri_multi_scale = {}
        path = ""
        print("--------------------------------------------------------------------------")
        if use_flattent:
            print("use_flattent")
            fmri_dir = f"/root/data-tmp/data/processed_data/subj0{sub}/fsaverage_not_mean_1000_new/{split}/sub{sub}_fmri_test_{i}.npy"
            fmri = np.load(fmri_dir)  # (22492, 4)
            fmri = fmri[perm]
            fmri_multi_scale = {}

            for (n, w) in multi_scale_size:
                key = f"{n}_{w}"
                windows = windows_by_scale[key]
                features = extract_fmri_features_simple(fmri, windows)  # (n, w, 4)
                fmri_multi_scale[key] = torch.from_numpy(features).float().to(device).unsqueeze(0)
        else:
            for key in fmri_datas_dict:
                fmri_path = fmri_datas_dict[key][i]
                print(fmri_path)
                path = fmri_path
                with open(fmri_path, "rb") as f:
                    fmri_multi_scale[key] = torch.from_numpy(np.load(f, allow_pickle=False)).float().to(device).unsqueeze(0)
        sub_tensor = torch.tensor(sub).to(device).unsqueeze(0)
        with autocast(dtype=torch.bfloat16):
            _, img_result = img_model(fmri_multi_scale, sub_tensor)
            _, text_result = text_model(fmri_multi_scale, sub_tensor)
            if pred_vision == None:
                pred_vision = img_result
                pred_text = text_result
            else:
                pred_vision = torch.cat((pred_vision, img_result), dim=0)
                pred_text = torch.cat((pred_text, text_result), dim=0)

pred_vision = pred_vision.to(torch.float32).cpu().numpy()
pred_text = pred_text.to(torch.float32).cpu().numpy()
pred_img_clip = np.zeros_like(pred_vision)
pred_text_clip = np.zeros_like(pred_text)
print("start load train_clip")

train_text_clip_dir = "/root/data-tmp/data/extracted_features/subj{:02d}/fsaverage_not_mean_1000/train_text/".format(
    sub)
train_text_clip_file_list = os.listdir(train_text_clip_dir)
train_text_clip_file_list = [os.path.join(train_text_clip_dir, file_name) for file_name in train_text_clip_file_list]
train_text_clips = []
for file_name in train_text_clip_file_list:
    train_text_clip = np.load(file_name)
    train_text_clips.append(train_text_clip)
train_text_clip = np.stack(train_text_clips, axis=0)
# train_text_clip = np.load(
#     '/usr/data/brain-diffuser/data/extracted_features/subj{:02d}/fsaverage_not_mean_1000/nsd_cliptext_train.npy'.format(
#         sub),
#     mmap_mode='r')

train_img_clip_dir = "/root/data-tmp/data/extracted_features/subj{:02d}/fsaverage_not_mean_1000/train/".format(
    sub)
train_img_clip_file_list = os.listdir(train_img_clip_dir)
train_img_clip_file_list = [os.path.join(train_img_clip_dir, file_name) for file_name in train_img_clip_file_list]
train_clips = []
for file_name in train_img_clip_file_list:
    train_clip = np.load(file_name)
    train_clips.append(train_clip)
train_img_clip = np.stack(train_clips, axis=0)
print("end load train_clip")

for i in range(77):
    print(i)
    pred_test_latent = pred_text[:, i]
    pred_test_latent_std = np.std(pred_test_latent, axis=0)
    epsilon = 1e-6
    pred_test_latent_std += epsilon
    std_norm_test_latent = (pred_test_latent - np.mean(pred_test_latent, axis=0)) / pred_test_latent_std
    pred_text_clip[:, i] = std_norm_test_latent * np.std(train_text_clip[:, i], axis=0) + np.mean(train_text_clip[:, i],
                                                                                                  axis=0)
    t_clip = pred_text[:, i]
for i in range(257):
    print(i)
    pred_test_latent = pred_vision[:, i]
    pred_test_latent_mean = np.mean(pred_test_latent, axis=0)
    pred_test_latent_std = np.std(pred_test_latent, axis=0)
    epsilon = 1e-6
    pred_test_latent_std += epsilon
    std_norm_test_latent = (pred_test_latent - pred_test_latent_mean) / pred_test_latent_std
    pred_img_clip[:, i] = std_norm_test_latent * np.std(train_img_clip[:, i], axis=0) + np.mean(train_img_clip[:, i],
                                                                                                axis=0)
    t_clip = pred_vision[:, i]
pred_vision = torch.from_numpy(pred_img_clip).to(device).half()
pred_text = torch.from_numpy(pred_text_clip).to(device).half()
n_samples = 1
ddim_steps = 50
ddim_eta = 0
scale = 7.5
xtype = 'image'
ctype = 'prompt'
net.autokl.half()

torch.manual_seed(0)
from tqdm import tqdm

if not use_latent:
    strength = 0.9
    ddim_steps = 60
random_array = np.random.randint(0, 256, size=(512, 512, 3), dtype=np.uint8)

# 转换成 PIL.Image.Image 类型
if not use_latent:
    zim = Image.fromarray(random_array)
# zim.show()
for im_id in tqdm(range(len(pred_vision)), desc="Processing Batches"):
    print("pre:", i)
    if use_latent:
        zim = Image.open('/root/data-tmp/brain-diffuser/results/latent/sub_{:02d}/{}/{}.png'.format(sub,latent_num, im_id))
    zim = regularize_image(zim)
    zin = zim * 2 - 1
    zin = zin.unsqueeze(0).cuda(device_num).half()

    init_latent = net.autokl_encode(zin)

    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, verbose=False)
    # strength=0.75
    assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
    t_enc = int(strength * ddim_steps)
    device = f'cuda:{device_num}'
    z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]).to(device))
    # z_enc,_ = sampler.encode(init_latent.cuda(1).half(), c.cuda(1).half(), torch.tensor([t_enc]).to(sampler.model.model.diffusion_model.device))

    dummy = ''
    utx = net.clip_encode_text(dummy)
    utx = utx.cuda(device_num).half()

    dummy = torch.zeros((1, 3, 224, 224)).cuda(device_num)
    uim = net.clip_encode_vision(dummy)
    uim = uim.cuda(device_num).half()

    z_enc = z_enc.cuda(device_num)

    h, w = 512, 512
    shape = [n_samples, 4, h // 8, w // 8]

    cim = pred_vision[im_id].unsqueeze(0)
    # cim = clip_datas[im_id].unsqueeze(0).half()
    # ctx = pred_text[im_id].unsqueeze(0)
    # text_dir = "/data/zbg/brain-diffuser/data/extracted_features/subj{:02d}/fsaverage_not_mean_1000/test_text_2/test_sub_{}_{}.npy".format(
    #     1, 1, im_id)
    # print(text_dir)
    # ctx = np.load(text_dir)
    ctx = pred_text[im_id].unsqueeze(0)
    if not use_prompt:
        ctx = utx
    # 增加一个维度（例如，将形状从 [3, 4, 5] 转为 [1, 3, 4, 5]）
    # ctx = ctx.unsqueeze(0).cuda(device_num).half()
    # c[:,0] = u[:,0]
    # z_enc = z_enc.cuda(1).half()

    sampler.model.model.diffusion_model.device = f'cuda:{device_num}'
    sampler.model.model.diffusion_model.half().cuda(device_num)
    # mixing = 0.4

    z = sampler.decode_dc(
        x_latent=z_enc,
        first_conditioning=[uim, cim],
        second_conditioning=[utx, ctx],
        t_start=t_enc,
        unconditional_guidance_scale=scale,
        xtype='image',
        first_ctype='vision',
        second_ctype='prompt',
        mixed_ratio=(1 - mixing), )

    z = z.cuda(device_num).half()
    x = net.autokl_decode(z)
    color_adj = 'None'
    # color_adj_to = cin[0]
    color_adj_flag = (color_adj != 'none') and (color_adj != 'None') and (color_adj is not None)
    color_adj_simple = (color_adj == 'Simple') or color_adj == 'simple'
    color_adj_keep_ratio = 0.5

    if color_adj_flag and (ctype == 'vision'):
        x_adj = []
        for xi in x:
            color_adj_f = color_adjust(ref_from=(xi + 1) / 2, ref_to=color_adj_to)
            xi_adj = color_adj_f((xi + 1) / 2, keep=color_adj_keep_ratio, simple=color_adj_simple)
            x_adj.append(xi_adj)
        x = x_adj
    else:
        x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
        x = [tvtrans.ToPILImage()(xi) for xi in x]
    save_path = "/root/data-tmp/brain-diffuser/results/versatile_diffusion/subj{:02d}/8_sub_{}_latent_{}_mix_0.5_text_{}/".format(
        sub,img_num,latent_num,text_num)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_name = "{}.png".format(im_id)
    file_path = os.path.join(save_path, file_name)
    x[0].save(file_path)
    print("idx:", im_id)
