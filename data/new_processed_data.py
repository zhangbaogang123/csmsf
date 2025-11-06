# coding:utf-8
import os
import sys
import numpy as np
import h5py
import scipy.io as spio
import nibabel as nib

import argparse


def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''

    def _check_keys(d):
        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for key in d:
            if isinstance(d[key], spio.matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _todict(matobj):
        '''
        A recursive function which constructs from matobjects nested dictionaries
        '''
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        '''
        A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        '''
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, spio.matlab.mio5_params.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list

    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub", help="Subject Number", default=1)
args = parser.parse_args()
sub = int(1)
s_nums = [40, 40, 32, 30, 40, 32, 40, 30]
# assert sub in [1, 2, 5, 7]
# 感兴趣区
l_voxel_indices = np.empty((0, 0)).astype(int)
r_voxel_indices = np.empty((0, 0)).astype(int)
base_label_dir = "/your_data_dir/data/nsddata/freesurfer/fsaverage/label"
all_files = os.listdir(base_label_dir)
probmap_files = [f for f in all_files if "probmap" in f]
for file_name in probmap_files:
    # 构建完整的文件路径
    label_path = os.path.join(base_label_dir, file_name)
    mask = nib.load(label_path)
    data = mask.get_fdata().squeeze()
    voxel_indices = np.where(data >= 0.7)[0]
    if file_name.startswith("lh."):
        l_voxel_indices = np.append(l_voxel_indices, voxel_indices)
        # print()
    elif file_name.startswith("rh."):
        r_voxel_indices = np.append(r_voxel_indices, voxel_indices)
num_voxel = len(l_voxel_indices) + len(r_voxel_indices)

print("Stimuli are loaded.")

stim_order_f = '/your_data_dir/data/nsddata/experiments/nsd/nsd_expdesign.mat'
stim_order = loadmat(stim_order_f)

sig_train = {}
sig_test = {}
s_num = s_nums[sub - 1]
num_trials = s_num * 750
my_train_im_idx = []
my_test_im_idx = []
my_sig_train = []
my_sig_test = []
for idx in range(num_trials):
    ''' nsdId as in design csv files'''
    nsdId = stim_order['subjectim'][sub - 1, stim_order['masterordering'][idx] - 1] - 1
    if stim_order['masterordering'][idx] > 1000:
        if nsdId not in sig_train:
            sig_train[nsdId] = []
        sig_train[nsdId].append(idx)
        my_train_im_idx.append(nsdId)
        my_sig_train.append(idx)
    else:
        if nsdId not in sig_test:
            sig_test[nsdId] = []
        sig_test[nsdId].append(idx)
        my_test_im_idx.append(nsdId)
        my_sig_test.append(idx)
test_im_idx = list(sig_test.keys())
num_train, num_test = len(my_train_im_idx), len(test_im_idx)



fmri = np.zeros((num_trials, num_voxel, 4)).astype(np.float32)
betas_dir = '/your_data_dir/data/nsddata_betas/ppdata/subj{:02d}/betas_fithrf_GLMdenoise_RR/'.format(sub)
surf_base_dir = "/your_data_dir/data/nsddata/freesurfer/fsaverage/surf/"
l_surf_dir = surf_base_dir + "lh.pial"
r_surf_dir = surf_base_dir + "rh.pial"
l_vertices, l_faces = nib.freesurfer.read_geometry(l_surf_dir)
r_vertices, r_faces = nib.freesurfer.read_geometry(r_surf_dir)
l_vertices = np.repeat(np.expand_dims(l_vertices, axis=0), 750, axis=0)
r_vertices = np.repeat(np.expand_dims(r_vertices, axis=0), 750, axis=0)
for i in range(s_num):
    l_beta_filename = "lh.betas_session{0:02d}.mgh".format(i + 1)
    l_beta_f = nib.load(betas_dir + l_beta_filename).get_fdata().astype(np.float32)
    l_beta_f = np.squeeze(l_beta_f)
    l_beta_f = np.nan_to_num(l_beta_f)
    l_beta_f = np.transpose(l_beta_f, (1, 0))
    l_beta_f = np.expand_dims(l_beta_f, axis=2)
    l_beta_f = np.concatenate((l_vertices, l_beta_f), axis=2)

    r_beta_filename = "rh.betas_session{0:02d}.mgh".format(i + 1)
    r_beta_f = nib.load(betas_dir + r_beta_filename).get_fdata().astype(np.float32)
    r_beta_f = np.squeeze(r_beta_f)
    r_beta_f = np.nan_to_num(r_beta_f)
    r_beta_f = np.transpose(r_beta_f, (1, 0))
    r_beta_f = np.expand_dims(r_beta_f, axis=2)
    r_beta_f = np.concatenate((r_vertices, r_beta_f), axis=2)
    print(i)
    l_extracted_voxels = l_beta_f[:, l_voxel_indices, :]
    r_extracted_voxels = r_beta_f[:, r_voxel_indices, :]
    all_voxels = np.concatenate((l_extracted_voxels, r_extracted_voxels), axis=1)
    fmri[i * 750:(i + 1) * 750] = all_voxels

print("fMRI Data are loaded.")
test_im_idx = list(sig_test.keys())
# num_train, num_test = len(train_im_idx), len(test_im_idx)
num_train, num_test = len(my_train_im_idx), len(test_im_idx)
vox_dim, im_dim, im_c = num_voxel, 425, 3
fmri_array = np.zeros((num_train, vox_dim, 4))
stim_array = np.zeros((num_train, im_dim, im_dim, im_c))
for i, idx in enumerate(my_train_im_idx):
    # stim_array[i] = stim[idx]
    fmri_array[i] = fmri[my_sig_train[i]]
    print(i)

np.save(
    '/your_data_dir/data/processed_data/subj{:02d}/fsaverage_not_mean_1000_new/nsd_train_fmriavg_nsdgeneral_sub{}.npy'.format(
        sub,
        sub),
    fmri_array)


print("Training data is saved.")

fmri_array = np.zeros((num_test, vox_dim, 4))
# stim_array = np.zeros((num_test, im_dim, im_dim, im_c))
for i, idx in enumerate(test_im_idx):
    # stim_array[i] = stim[idx]
    sig_test_idx = sig_test[idx]
    sig_test_idx_stored = sorted(sig_test_idx)
    fmri_sig_test_idx_stored = fmri[sig_test_idx_stored]
    fmri_sig_test_idx_stored_mean = fmri_sig_test_idx_stored.mean(axis=0)
    fmri_array[i] = fmri_sig_test_idx_stored_mean
    print(i)


np.save(
    '/your_data_dir/data/processed_data/subj{:02d}/fsaverage_not_mean_1000_new/nsd_test_fmriavg_nsdgeneral_sub{}.npy'.format(
        sub, sub), fmri_array)


print("Test data is saved.")
