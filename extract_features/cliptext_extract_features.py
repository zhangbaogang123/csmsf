import sys

sys.path.append('versatile_diffusion')
import os
import numpy as np

import torch
from lib.cfg_helper import model_cfg_bank
from lib.model_zoo import get_model
# from torch.utils.data import DataLoader, Dataset
#
# from lib.model_zoo.vd import VD
# from lib.cfg_holder import cfg_unique_holder as cfguh
# from lib.cfg_helper import get_command_line_args, cfg_initiates, load_cfg_yaml
# import matplotlib.pyplot as plt
# import torchvision.transforms as T



cfgm_name = 'vd_noema'
pth = 'versatile_diffusion/pretrained/vd-four-flow-v1-0-fp16-deprecated.pth'
cfgm = model_cfg_bank()(cfgm_name)
net = get_model()(cfgm)
sd = torch.load(pth, map_location='cpu')
net.load_state_dict(sd, strict=False)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net.clip = net.clip.to(device)
for sub in range(8):
    sub = sub + 1
    train_caps = np.load(
        '/your_data-dir/data/processed_data/subj{:02d}/fsaverage_not_mean_1000/nsd_train_cap_sub{}.npy'.format(
            sub, sub))
    test_caps = np.load(
        '/your_data-dir/data/processed_data/subj{:02d}/fsaverage_not_mean_1000/nsd_test_cap_sub{}.npy'.format(
            sub,
            sub))

    num_embed, num_features, num_test, num_train = 77, 768, len(test_caps), len(train_caps)

    train_clip = np.zeros((num_train, num_embed, num_features))
    test_clip = np.zeros((num_test, num_embed, num_features))
    save_train_path = "/your_data-dir/data/extracted_features/subj{:02d}/fsaverage_not_mean_1000/train_text_2/".format(
        sub)
    save_test_path = "/your_data-dir/data/extracted_features/subj{:02d}/fsaverage_not_mean_1000/test_text_2/".format(
        sub)
    os.makedirs(save_train_path, exist_ok=True)
    os.makedirs(save_test_path, exist_ok=True)
    with torch.no_grad():
        for i, annots in enumerate(test_caps):
            cin = list(annots[annots != ''])
            print(i)
            c = net.clip_encode_text(cin)
            file_name = "test_sub_{}_{}.npy".format(sub, i)
            np.save(save_test_path + file_name, c)
            test_clip[i] = c.to('cpu').numpy().mean(0)



        for i, annots in enumerate(train_caps):
            cin = list(annots[annots != ''])
            print(i)
            c = net.clip_encode_text(cin)
            file_name = "train_sub_{}_{}.npy".format(sub, i)
            np.save(save_train_path + file_name, c)
            train_clip[i] = c.to('cpu').numpy().mean(0)

