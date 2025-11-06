# -*- coding: utf-8 -*-
# import pydevd_pycharm
# pydevd_pycharm.settrace('localhost', port=9999, stdoutToServer=True, stderrToServer=True)
# ... rest of your code
import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["NCCL_DEBUG"] = "INFO"
# os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"    # 避免 NCCL 内部挂死
os.environ["NCCL_P2P_DISABLE"] = "1"  # 禁用直接 P2P 通信，防止驱动死锁
os.environ["NCCL_IB_DISABLE"] = "1"               # 如果你没用 RDMA/InfiniBand（大多数没），必须加
os.environ["NCCL_SHM_DISABLE"] = "1"
# os.environ["NCCL_SOCKET_IFNAME"] = "eth0"
# from dataset_flattent import FmriDataSet
from dataset_img_text import FmriDataSet
import torch
from torch.utils.data import DataLoader, random_split
from pl_model_text import FineTunedModel
from pl_model_tranfomer_con_loss import Pct
import argparse
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import LearningRateMonitor
import datetime
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
import lightning.pytorch as pl
import numpy as np
import random
from torch.utils.data.distributed import DistributedSampler
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

    cuda_available = torch.cuda.is_available()
    print("CUDA available:", cuda_available)

    if cuda_available:
        print("Number of CUDA devices:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f"CUDA device {i}: {torch.cuda.get_device_name(i)}")

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    seed_everything(42)
    torch.set_float32_matmul_precision('medium')

    print("model load")
    multi_scale_size = [(256,256),(512, 128),(1024,64)]
    save_checkpoin_path="your_path"
    img_checkpoint_path = "your_path/xx.ckpt"
    pct_model = Pct.load_from_checkpoint(
        checkpoint_path=img_checkpoint_path,
        tf_drop=0.2, lr=2e-4, weight_decay=0.05, multi_scale_size=multi_scale_size, use_sub_info=True, type="img",map_location=lambda storage, loc: storage.cuda(0))
    ft_Model = FineTunedModel(pct=pct_model, tf_drop=0.3, lr=2e-4, weight_decay=0.1, multi_scale_size=multi_scale_size,
                              use_sub_info=True)
    train_subs = [1,2,5,7]
    test_subs = [1,2,5,7]
    train_dataset = FmriDataSet(multi_scale_size=multi_scale_size, istrain=True, train_subs=train_subs,
                                test_subs=test_subs)
    test_dataset = FmriDataSet(multi_scale_size=multi_scale_size, istrain=False, train_subs=train_subs,
                               test_subs=test_subs)
    train_loader = DataLoader(train_dataset, num_workers=16, batch_size=32, drop_last=True,
                              persistent_workers=True, prefetch_factor=2, pin_memory=True, shuffle=True)
    test_loader = DataLoader(test_dataset, num_workers=16, batch_size=32, drop_last=True,
                             persistent_workers=True)
    torch.set_float32_matmul_precision('medium')
    lr_monitor = LearningRateMonitor(logging_interval='step')
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        verbose=True,
        filename='{epoch:02d}-{val_loss:.2f}',
        dirpath=f'{save_checkpoin_path}/{current_time}',
        every_n_epochs=1,
    )
    callbacks = [checkpoint_callback, lr_monitor]
    tensorboard_logger = TensorBoardLogger("text_logs", name="my_model")
    trainer = Trainer(callbacks=callbacks, max_epochs=200, devices=4, precision="bf16-mixed",
                      strategy=DDPStrategy(find_unused_parameters=False, timeout=timedelta(seconds=30 * 5)),
                      profiler="simple", accelerator="gpu")
    trainer.fit(ft_Model, train_loader, test_loader)
