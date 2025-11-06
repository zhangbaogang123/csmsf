# csmsf
Official repository for the paper "fMRI-based Cross-subject Visual Reconstruction via Structural Consistency and Multi-scale Cortical Modeling"
# Installation
### 1) git clone this repository:
```bash
git clone https://github.com/zhangbaogang123/csmsf.git
cd csmsf

```
### 2) install  virtual environment:
```bash
conda create -n fmri python=3.11
conda activate fmri
pip install -r requirements.txt
```

# Data Download and Preprocessing

### 1) Download fMRI data from NSD
```bash
python ./data/down_data.py
```
### 2) fmri data preprocessing and standardization
```bash
python ./data/processed_data.py
python ./data/normal_fmri.py
```
### 3) group fmri
```bash
python ./data/group_fmri_data.py
```
# Extract features
Before extracting features, please download pretrained Versatile Diffusion model 
"vd-four-flow-v1-0-fp16-deprecated.pth", "kl-f8.pth" and "optimus-vae.pth" from 
[HuggingFace](https://huggingface.co/shi-labs/versatile-diffusion/tree/main/pretrained_pth) 
download ¡°sd_image_var_autoenc.pth¡± from [HuggingFace](https://huggingface.co/zhangbaogang/csmsf/tree/main)
and put them in `versatile_diffusion/pretrained/` folder
### 1) Extract image features
```bash
python ./extract_features/clipvision_extract_features.py
```
### 2) Extract text features
```bash
python ./data/processed_captions.py
python ./extract_features/cliptext_extract_features.py
```

### 3) Extract VAE features
```bash
python ./extract_features/export_npy_images.py
python ./extract_features/extract_latent_features.py
python ./extract_features/normal_latent_features.py
```
# Train
### 1) train fmri to clip image emb 
```bash
python pl_train_clip.py
```
### 2) train fmri to clip text emb
```bash
python pl_train_text.py
```
### 3) train fmri to VAE latent
```bash
python pl_train_latent.py
```

# Inference

We provide pretrained CSMSF models on [Hugging Face](https://huggingface.co/zhangbaogang/csmsf/tree/main).  
If you prefer not to train the models from scratch, you can directly use our released checkpoints:

- `fmri2img_emb.ckpt` - for fMRI-to-image embedding reconstruction  
- `fmri2text_emb.ckpt` - for fMRI-to-text embedding reconstruction  

Simply download them from the link above and place them in the corresponding directory before running the inference scripts.
### 1) inference fmri to blurry image
```bash
python reconstruce_latent_img_demo.py
```
### 1) inference fmri to final image
```bash
python reconstruce_final_image.py
```


