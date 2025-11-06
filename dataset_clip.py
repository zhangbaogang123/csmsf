# -*- coding: utf-8 -*-
from torch.utils.data import Dataset
import numpy as np
import os
import concurrent.futures
import torch
from collections import defaultdict
import re
import time
from functools import lru_cache


@lru_cache(maxsize=1)  # 可根据可用内存调节
def cached_load_npy(path):
    with open(path, "rb") as f:
        return np.load(f, allow_pickle=False)


def safe_load_npy(path):
    with open(path, "rb") as f:
        return np.load(f, allow_pickle=False)


class FmriDataSet(Dataset):
    def __init__(self, istrain=True, train_subs=[1], test_subs=[1],
                 multi_scale_size=[(256, 256), (512, 128), (1024, 64)]):
        print(f"Main process PID: {os.getpid()}")
        self.device = "cpu"
        self.istrain = istrain
        self.img_datas_list = []
        self.train_subs = train_subs
        self.test_subs = test_subs
        self.multi_scale_size = multi_scale_size
        subjects = self.train_subs if istrain else self.test_subs
        split = "train" if istrain else "test"
        self.fmri_datas_dict = {f"{n}_{w}": [] for (n, w) in self.multi_scale_size}

        for sub in subjects:
            # fmri_dir = "/root/data-tmp/data/processed_data/subj{:02d}/fsaverage_not_mean_1000_new_512_128/train/".format(
            #     sub)
            for (n_win, win_size) in self.multi_scale_size:
                key = f"{n_win}_{win_size}"
                scale_tag = f"fsaverage_not_mean_1000_new_{n_win}_{win_size}"
                fmri_dir = f"/root/data-tmp/data/processed_data/subj{sub:02d}/{scale_tag}/{split}/"
                fmri_files = [os.path.join(fmri_dir, f) for f in os.listdir(fmri_dir)]
                self.fmri_datas_dict[key].extend(fmri_files)

            img_dir = f"/root/data-tmp/data/extracted_features/subj{sub:02d}/fsaverage_not_mean_1000/{split}/"

            img_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir)]

            self.img_datas_list.extend(img_files)

        def custom_sort_fmri(file_path):
            parts = file_path.split('/')
            file_name = parts[-1]  # 获取文件名，例如 'sub2_fmri_test_0.npy'
            sub_num = int(file_name.split('_')[0][3:])  # 提取 'sub' 后面的数字
            test_num = int(file_name.split('_')[-1].split('.')[0])  # 提取 'test' 后面的数字
            return (sub_num, test_num)

        def custom_sort_img(file_path):
            parts = file_path.split('/')
            file_name = parts[-1]
            parts = file_name.split('_')
            first_num = int(parts[2])  # 提取第一个数字
            second_num = int(parts[3].split('.')[0])  # 提取第二个数字
            return (first_num, second_num)

        for k in self.fmri_datas_dict:
            self.fmri_datas_dict[k] = sorted(self.fmri_datas_dict[k], key=custom_sort_fmri)

        self.img_datas_list = sorted(self.img_datas_list, key=custom_sort_img)

        self.sample_len = len(self.img_datas_list)

    def __getitem__(self, item):
        # t0 = time.time()
        fmri_multi_scale = {}
        path = ""
        for key in self.fmri_datas_dict:
            fmri_path = self.fmri_datas_dict[key][item]
            path = fmri_path
            with open(fmri_path, "rb") as f:
                fmri_multi_scale[key] = torch.from_numpy(np.load(f, allow_pickle=False)).float()

        img = torch.from_numpy(cached_load_npy(self.img_datas_list[item])).float()
        match = re.search(r"subj(\d+)", path)
        subject_id = 1
        if match:
            subject_id = match.group(1)
            subject_id = int(subject_id)
        else:
            print("not match")
        match = re.search(r'_(\d+)\.npy$', path)
        if match:
            number = int(match.group(1))
            # print(number)  # 输出：103
        else:
            print("not match")

        return fmri_multi_scale, img, subject_id

    def __len__(self):
        return self.sample_len
