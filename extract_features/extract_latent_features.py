# -*- coding: utf-8 -*-
import os
from typing import Tuple, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms

import kornia
from kornia.augmentation.container import AugmentationSequential

from diffusers.models.autoencoder_kl import AutoencoderKL  # HuggingFace Diffusers



# ======================
# Configuration
# ======================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SUBJECTS = list(range(1, 9))          # subj01 ~ subj08
SPLITS = ["train", "test"]            # Process both train / test

# Data root directory
DATA_ROOT = "/your_data-dir/data/processed_data"  # will append subj0{n}/{split}_images

# Output directory suffixes (consistent with existing setup)
LATENT_DIRNAME = "{}_latents_blurred"    # e.g., "train_latents_blurred"

# Model and processing flags
USE_BLU_RRED_TRAINING = True         # Apply median blur before feeding to VAE

AUTOENC_CKPT = "versatile_diffusion/pretrained/sd_image_var_autoenc.pth"


# ======================
# Model loading
# ======================
def load_vae(device: torch.device) -> AutoencoderKL:
    autoenc = AutoencoderKL(
        down_block_types=['DownEncoderBlock2D'] * 4,
        up_block_types=['UpDecoderBlock2D'] * 4,
        block_out_channels=[128, 256, 512, 512],
        layers_per_block=2,
        sample_size=256
    )
    state = torch.load(AUTOENC_CKPT, map_location="cpu")
    autoenc.load_state_dict(state, strict=True)
    autoenc.requires_grad_(False).eval().to(device)
    return autoenc


def build_augs(device: torch.device) -> AugmentationSequential:
    train_augs = AugmentationSequential(
        kornia.augmentation.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
        kornia.augmentation.RandomGrayscale(p=0.2),
        kornia.augmentation.RandomSolarize(p=0.2),
        kornia.augmentation.RandomGaussianBlur(kernel_size=(7, 7), sigma=(0.1, 2.0), p=0.1),
        kornia.augmentation.RandomResizedCrop((512, 512), scale=(0.5, 1.0)),
        data_keys=["input"],
    ).to(device)
    return train_augs


# ======================
# Preprocessing and encoding
# ======================
to_tensor = transforms.ToTensor()

def preprocess_and_encode(
    img_path: str,
    autoenc: AutoencoderKL,
    device: torch.device,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Returns:
      latent:        (1, 4, 64, 64)
      cnx_embeds:    optional
      cnx_aug_embeds:optional
    """
    image = Image.open(img_path).convert("RGB")
    image = to_tensor(image).unsqueeze(0).to(device, dtype=torch.float32)  # (1,3,H,W)

    # Resize to 512
    image_512 = F.interpolate(image, (512, 512), mode="bilinear", align_corners=False, antialias=True)

    with torch.no_grad():
        # Optional median blur before VAE encoding
        enc_input = kornia.filters.median_blur(image_512, (15, 15)) if USE_BLU_RRED_TRAINING else image_512
        latent = autoenc.encode(2 * enc_input - 1).latent_dist.mode() * 0.18215  # (1,4,64,64)

    return latent


import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset


class batch_generator_external_images(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.im = np.load(data_path).astype(np.uint8)

    def __getitem__(self, idx):
        img = Image.fromarray(self.im[idx])
        return img

    def __len__(self):
        return len(self.im)

if __name__ == "__main__":
    autoenc = load_vae(DEVICE)

    total_files, done_files, skipped_files = 0, 0, 0

    for sub in SUBJECTS:
        for split in SPLITS:
            image_dir = os.path.join(DATA_ROOT, f"subj0{sub}", f"{split}_images")
            if not os.path.isdir(image_dir):
                print(f"[WARN] Skipped: directory not found -> {image_dir}")
                continue

            latent_dir = os.path.join(DATA_ROOT, f"subj0{sub}", LATENT_DIRNAME.format(split))
            os.makedirs(latent_dir, exist_ok=True)

            image_filenames = sorted(os.listdir(image_dir))
            print(f"\n[INFO] subj0{sub} | split={split} | total files to process: {len(image_filenames)}")

            for filename in image_filenames:
                # Only process common image file extensions
                if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    skipped_files += 1
                    continue

                total_files += 1
                img_path = os.path.join(image_dir, filename)
                base = os.path.splitext(filename)[0]

                try:
                    latent, cnx_embeds, cnx_aug_embeds = preprocess_and_encode(
                        img_path, autoenc, DEVICE
                    )

                    np.save(
                        os.path.join(latent_dir, f"sub{sub}_latent_{split}_{base}.npy"),
                        latent.squeeze(0).cpu().numpy()
                    )

                    done_files += 1
                    if done_files % 50 == 0:
                        print(f"[PROGRESS] Saved {done_files} / currently processing subj0{sub}-{split}-{base}")

                except Exception as e:
                    skipped_files += 1
                    print(f"[ERROR] Failed -> subj0{sub} {split} {filename} | {e}")

    print("\n========== SUMMARY ==========")
    print(f"Total candidate images: {total_files + skipped_files}")
    print(f"Successfully saved latents: {done_files}")
    print(f"Skipped/failed: {skipped_files}")
