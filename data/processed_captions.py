# coding:utf-8
import os
import sys
import numpy as np
import h5py
import scipy.io as spio
import nibabel as nib

import argparse
sub = int(8)
s_nums = [40, 40, 32, 30, 40, 32, 40, 30]
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
for sub in range(8):
    sub=sub+1
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
    annots_cur = np.load('/your_data_dir/data/annots/COCO_73k_annots_curated.npy')

    captions_array = np.empty((num_train, 5), dtype=annots_cur.dtype)
    for i, idx in enumerate(my_train_im_idx):
        captions_array[i, :] = annots_cur[idx, :]
        print(i)
    np.save('/your_data_dir/data/processed_data/subj{:02d}/fsaverage_not_mean_1000/nsd_train_cap_sub{}.npy'.format(sub, sub),
            captions_array)

    captions_array = np.empty((num_test, 5), dtype=annots_cur.dtype)
    for i, idx in enumerate(test_im_idx):
        captions_array[i, :] = annots_cur[idx, :]
        print(i)
    np.save('/your_data_dir/data/subj{:02d}/fsaverage_not_mean_1000/nsd_test_cap_sub{}.npy'.format(sub, sub),
            captions_array)