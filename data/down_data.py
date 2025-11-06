# coding:utf-8
import os
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from tqdm.auto import tqdm

# ====== 1) Anonymous S3 client (public bucket, no creds needed) ======
s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
bucket = "natural-scenes-dataset"

# ====== 2) Utilities ======
def list_objects_with_size(bucket: str, prefix: str):
    """List all S3 objects under a prefix, returning a list of dicts with Key and Size."""
    objs = []
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for it in page.get('Contents', []):
            # Skip "directory placeholder" keys
            if it['Key'].endswith('/'):
                continue
            objs.append({'Key': it['Key'], 'Size': it['Size']})
    return objs

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def skip_if_complete(local_path: str, size: int) -> bool:
    """Return True if local file exists and size matches."""
    return os.path.exists(local_path) and os.path.getsize(local_path) == size

def download_one_with_progress(bucket: str, key: str, local_path: str, overall_pbar=None):
    """Download a single S3 object with a per-file tqdm bar and optional overall pbar."""
    size = s3.head_object(Bucket=bucket, Key=key)['ContentLength']


    if skip_if_complete(local_path, size):
        tqdm.write(f"✔ Skipped (already complete): {local_path}")
        if overall_pbar is not None:
            overall_pbar.update(0)  # no-op for clarity
        return

    ensure_dir(os.path.dirname(local_path))
    desc = os.path.basename(key) or key
    file_pbar = tqdm(total=size, unit='B', unit_scale=True, desc=desc, leave=False)

    def _cb(bytes_amount):
        file_pbar.update(bytes_amount)
        if overall_pbar is not None:
            overall_pbar.update(bytes_amount)


    s3.download_file(bucket, key, local_path, Callback=_cb)
    file_pbar.close()
    tqdm.write(f"↓ Downloaded: {key} -> {local_path}")

def download_prefix_with_progress(bucket: str, prefix: str, local_root: str):
    """Download an entire S3 prefix with an overall progress bar."""
    objs = list_objects_with_size(bucket, prefix)
    if not objs:
        tqdm.write(f"(empty) {prefix}")
        return

    total_bytes = sum(o['Size'] for o in objs)
    overall_desc = f"{prefix.rstrip('/')}"
    with tqdm(total=total_bytes, unit='B', unit_scale=True, desc=overall_desc) as overall_pbar:
        for o in objs:
            key = o['Key']
            rel = os.path.relpath(key, prefix)
            local_path = os.path.join(local_root, rel)
            # Fast-path skip if already complete
            if skip_if_complete(local_path, o['Size']):
                tqdm.write(f"Skipped (already complete): {local_path}")
                overall_pbar.update(0)  # keep bars consistent
                continue
            download_one_with_progress(bucket, key, local_path, overall_pbar)

# ====== 3) Base output directory ======
BASE_DIR = "/root/data-tmp/data"

# ====== 4) Downloads ======
if __name__ == "__main__":

    # (4) betas for subj01–subj08
    for sid in range(1, 9):
        subj = f"subj{sid:02d}"
        prefix = f"nsddata_betas/ppdata/{subj}/fsaverage/betas_fithrf_GLMdenoise_RR/"
        local_dir = os.path.join(BASE_DIR, f"nsddata_betas/ppdata/{subj}/fsaverage/betas_fithrf_GLMdenoise_RR")
        download_prefix_with_progress(bucket, prefix, local_dir)

    # (1) experiments/nsd/
    download_prefix_with_progress(
        bucket,
        "nsddata/experiments/nsd/",
        os.path.join(BASE_DIR, "nsddata/experiments/nsd")
    )

    # (2) freesurfer/fsaverage/label/
    download_prefix_with_progress(
        bucket,
        "nsddata/freesurfer/fsaverage/label/",
        os.path.join(BASE_DIR, "nsddata/freesurfer/fsaverage/label")
    )

    # (3) stimuli file (single HDF5) — show a per-file bar + tiny overall
    stim_key = "nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5"
    stim_dir = os.path.join(BASE_DIR, "nsddata_stimuli/stimuli/nsd")
    ensure_dir(stim_dir)
    size = s3.head_object(Bucket=bucket, Key=stim_key)['ContentLength']
    with tqdm(total=size, unit='B', unit_scale=True, desc=os.path.basename(stim_key)) as pbar:
        download_one_with_progress(bucket, stim_key, os.path.join(stim_dir, "nsd_stimuli.hdf5"), overall_pbar=pbar)


