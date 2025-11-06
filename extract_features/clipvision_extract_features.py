# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path
ROOT = Path(__file__).parent           # 当前脚本所在目录
META = ROOT / "metadata"
VD_PATH=ROOT / "versatile_diffusion"
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 维持你原来的 sys.path 写法
sys.path.append('versatile_diffusion')

from PIL import Image
import numpy as np
import torch
from lib.cfg_helper import model_cfg_bank
from lib.model_zoo import get_model
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

import argparse

parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub", help="Subject Number (ignored in 8-sub mode)", default=1)
args = parser.parse_args()

cfgm_name = 'vd_noema'

pth = 'versatile_diffusion/pretrained/vd-four-flow-v1-0-fp16-deprecated.pth'
cfgm = model_cfg_bank()(cfgm_name)
net = get_model()(cfgm)
sd = torch.load(pth, map_location='cpu')
net.load_state_dict(sd, strict=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net.clip = net.clip.to(device)   # 按你原来的做法只把 clip 放到 device


class batch_generator_external_images(Dataset):

    def __init__(self, data_path):
        self.data_path = data_path
        self.im = np.load(data_path).astype(np.uint8)

    def __getitem__(self, idx):
        img = Image.fromarray(self.im[idx])
        img = T.functional.resize(img, (512, 512))
        img = T.functional.to_tensor(img).float()
        img = img * 2 - 1
        return img

    def __len__(self):
        return len(self.im)


batch_size = 1

for sub in range(1, 9):
    # --- 构造数据集/loader ---
    train_path = '/your_data-dir/data/processed_data/subj{:02d}/fsaverage_not_mean_1000/nsd_train_stim_sub{}.npy'.format(sub, sub)
    test_path  = '/your_data-dir/data/processed_data/subj{:02d}/fsaverage_not_mean_1000/nsd_test_stim_sub{}.npy'.format(sub, sub)

    train_images = batch_generator_external_images(data_path=train_path)
    test_images  = batch_generator_external_images(data_path=test_path)

    trainloader = DataLoader(train_images, batch_size, shuffle=False)
    testloader  = DataLoader(test_images,  batch_size, shuffle=False)

    num_embed, num_features, num_test, num_train = 257, 768, len(test_images), len(train_images)
    train_clip = np.zeros((num_train, num_embed, num_features))
    test_clip  = np.zeros((num_test,  num_embed, num_features))

    with torch.no_grad():
        # --- test ---
        for i, cin in enumerate(testloader):
            print(f"[sub {sub:02d}] test batch {i}")
            c = net.clip_encode_vision(cin)
            test_clip[i] = c[0].cpu().numpy()
            save_dir = '/your_data-dir/data/extracted_features/subj{:02d}/fsaverage_not_mean_1000/test/'.format(sub)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path = save_dir + "test_clip_{:02d}_{}.npy".format(sub, i)
            np.save(save_path, c[0].cpu().numpy())

        # --- train ---
        for i, cin in enumerate(trainloader):
            print(f"[sub {sub:02d}] train batch {i}")
            c = net.clip_encode_vision(cin)
            train_clip[i] = c[0].cpu().numpy()
            save_dir = '/your_data-dir/data/extracted_features/subj{:02d}/fsaverage_not_mean_1000/train/'.format(sub)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path = save_dir + "train_clip_{:02d}_{}.npy".format(sub, i)
            np.save(save_path, c[0].cpu().numpy())
