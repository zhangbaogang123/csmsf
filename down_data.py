import os
import boto3
from botocore import UNSIGNED
from botocore.config import Config

# ====== 1. Initialize an anonymous S3 client ======
# NSD S3 bucket is public, so we use unsigned (no credentials required)
s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
bucket = "natural-scenes-dataset"

# ====== 2. Generic download function ======
def download_prefix(bucket, prefix, local_dir):
    """
    Recursively download all objects from a given S3 prefix to a local directory.

    Args:
        bucket (str): S3 bucket name.
        prefix (str): S3 key prefix to download.
        local_dir (str): Local directory to store downloaded files.
    """
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            rel_path = os.path.relpath(key, prefix)
            local_path = os.path.join(local_dir, rel_path)

            # Skip directories (S3 keys ending with '/')
            if key.endswith("/"):
                continue

            # Make sure the local directory exists
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            # Download only if file doesn’t already exist
            if not os.path.exists(local_path):
                print(f"Downloading: {key} → {local_path}")
                s3.download_file(bucket, key, local_path)
            else:
                print(f"Skipped (exists): {local_path}")

# ====== 3. Set the base output directory ======
BASE_DIR = "/root/data-tmp/new_data"

# ====== 4. Download all required NSD components ======

# (1) Experiment files
download_prefix(
    bucket,
    "nsddata/experiments/nsd/",
    os.path.join(BASE_DIR, "nsddata/experiments/nsd")
)

# (2) fsaverage label files
download_prefix(
    bucket,
    "nsddata/freesurfer/fsaverage/label/",
    os.path.join(BASE_DIR, "nsddata/freesurfer/fsaverage/label")
)

# (3) Stimulus file (single HDF5)
stim_path = os.path.join(BASE_DIR, "nsddata_stimuli/stimuli/nsd")
os.makedirs(stim_path, exist_ok=True)
s3.download_file(
    bucket,
    "nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5",
    os.path.join(stim_path, "nsd_stimuli.hdf5")
)
print("Downloaded nsd_stimuli.hdf5")

# (4) Beta data for subjects subj01–subj08
for sid in range(1, 9):
    subj = f"subj{sid:02d}"
    prefix = f"nsddata_betas/ppdata/{subj}/fsaverage/betas_fithrf_GLMdenoise_RR/"
    local_dir = os.path.join(BASE_DIR, f"nsddata_betas/ppdata/{subj}/fsaverage/betas_fithrf_GLMdenoise_RR")
    download_prefix(bucket, prefix, local_dir)
