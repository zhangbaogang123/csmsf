# csmsf
Official repository for the paper "fMRI-based Cross-subject Visual Reconstruction via Structural Consistency and Multi-scale Cortical Modeling"
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
### 1) train fmri to image
```bash
python pl_train_clip.py
```
### 2) train fmri to text
```bash
python pl_train_text.py
```
```bash
python test.py
```
