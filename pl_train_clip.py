# -*- coding: utf-8 -*-
# import pydevd_pycharm
# pydevd_pycharm.settrace('localhost', port=9999, stdoutToServer=True, stderrToServer=True)
# ... rest of your code
import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["NCCL_DEBUG"] = "INFO"

os.environ["NCCL_P2P_DISABLE"] = "1"

from dataset_clip import FmriDataSet
import torch
from torch.utils.data import DataLoader, random_split, DistributedSampler
from pl_model_clip import Pct
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import LearningRateMonitor
import datetime
from lightning.pytorch import Trainer
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.loggers import TensorBoardLogger
import lightning.pytorch as pl
import numpy as np
import random
from datetime import timedelta


def seed_everything(seed=42):
    random.seed(seed)  # Python 内置随机数种子
    np.random.seed(seed)  # NumPy 随机数种子
    torch.manual_seed(seed)  # PyTorch CPU 随机数种子
    torch.cuda.manual_seed(seed)  # PyTorch GPU 随机数种子
    torch.cuda.manual_seed_all(seed)  # 如果使用多个 GPU
    torch.backends.cudnn.deterministic = True  # 确保 CUDA 可复现
    torch.backends.cudnn.benchmark = False  # 关闭 cuDNN 的自动优化
    pl.seed_everything(seed, workers=True)  # PyTorch Lightning 设定种子


if __name__ == '__main__':

    # 检查 CUDA 是否可用
    cuda_available = torch.cuda.is_available()
    print("CUDA available:", cuda_available)

    # 如果 CUDA 可用，打印出 CUDA 设备的数量和名称
    if cuda_available:
        print("Number of CUDA devices:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f"CUDA device {i}: {torch.cuda.get_device_name(i)}")

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    seed_everything(42)
    multi_scale_size = [(256, 256), (512, 128), (1024, 64)]
    torch.set_float32_matmul_precision('highest')

    print("start load data")

    train_subs = [1,2,5,7]
    test_subs = [1,2,5,7]
    train_dataset = FmriDataSet(multi_scale_size=multi_scale_size, istrain=True, train_subs=train_subs,
                                test_subs=test_subs)
    test_dataset = FmriDataSet(multi_scale_size=multi_scale_size, istrain=False, train_subs=train_subs,
                               test_subs=test_subs)
    print("test_dataset")
    train_loader = DataLoader(train_dataset, num_workers=16, batch_size=32, drop_last=True,
                              persistent_workers=True, prefetch_factor=2, pin_memory=True, shuffle=True)
    print("train_loader")
    test_loader = DataLoader(test_dataset, num_workers=16, batch_size=32, drop_last=True,
                             persistent_workers=True)
    print("test_loader")
    model = Pct(tf_drop=0.2, lr=2e-4, weight_decay=0.05, multi_scale_size=multi_scale_size, use_sub_info=True,use_xyz=True)
    print("model load")

    torch.set_float32_matmul_precision('medium')
    lr_monitor = LearningRateMonitor(logging_interval='step')
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    dirpath="Your_path"
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        verbose=True,
        filename='{epoch:02d}-{val_loss:.2f}',
        dirpath=dirpath+current_time,
        every_n_epochs=1,
    )
    callbacks = [checkpoint_callback, lr_monitor]
    tensorboard_logger = TensorBoardLogger("tb_logs", name="my_model")
trainer = Trainer(callbacks=callbacks, max_epochs=400, devices=4, precision="bf16-mixed",
                  strategy=DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=30 * 10)),
                  profiler="simple",
                  accelerator="gpu")
trainer.fit(model, train_loader, test_loader)
