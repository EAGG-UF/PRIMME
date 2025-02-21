#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 05:05:54 2023

@author: yang.kang
"""
from tqdm import tqdm
import os.path
import torch
import h5py
import numpy as np
from pathlib import Path
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#device  = 'cpu'
import functions as fs
import PRIMME

### generate dataset

#trainset = fs.create_SPPARKS_dataset(size=[257,257], ngrains_rng=[256, 256], kt=0.66, cutoff=0.0, nsets=360, max_steps=500, offset_steps=301, future_steps=10)

### init parameters
paras_dict = {"num_eps": 25,
              "mode": "Single_Step",
              "dims": 2, 
              "obs_dim":17, 
              "act_dim":17, 
              "lr": 5e-5, 
              "reg": 1, 
              "pad_mode": "circular",
              "if_plot": True}

trainset_dict = {"case1": ["./data/trainset_spparks_sz(257x257)_ng(256-256)_nsets(200)_future(10)_max(100)_kt(0.66)_cut(0).h5"]}
n_samples_dict = {"case1": [[200]]}
n_step_dict = {"case1": [[1, 5]]}

test_case_dict = {"case1": ["circular", [[257, 257], 64]],
                  "case2": ["circular", [[512, 512], 200]], 
                  "case3": ["square", [[512, 512], 64]],
                  "case4": ["square", [[512, 512], 200]],
                  "case5": ["hex", [[512, 512], 64]],
                  "case6": ["grain", [[512, 512], 512]],
                  "case7": ["grain", [[1024, 1024], 2**12]]} # "case8": ["grain", [[2400, 2400], 24000]]

### Train PRIMME using the above training set from SPPARKS
for key_data in trainset_dict.keys():
    trainset = trainset_dict[key_data][0]
    for n_samples in n_samples_dict[key_data]:
        for n_step in n_step_dict[key_data]:
            modelname = PRIMME.train_primme(trainset, n_step, n_samples, test_case_dict)
            
### Run SPPARKS model
sub_folder = "SPPARK"
for key in test_case_dict.keys():
    grain_shape, grain_sizes = test_case_dict[key]
    if grain_sizes[0][0] > 1000:
        nsteps = 1400;
    else:
        nsteps = 1800
    if grain_shape == "hex":
        ic_shape = grain_shape
    else:   
        ic_shape = grain_shape + "(" + ("_").join([str(grain_sizes[0][0]), str(grain_sizes[0][1]), str(grain_sizes[1])]) + ")"
    filename_test = ic_shape + ".pickle"    
    path_load = Path('./data').joinpath(filename_test)
    if os.path.isfile(str(path_load)):  
        data_dict = fs.load_picke_files(load_dir = Path('./data'), filename_save = filename_test)
        ic, ea, miso_array, miso_matrix = data_dict["ic"], data_dict["ea"], data_dict["miso_array"], data_dict["miso_matrix"]  
    else:
        ic, ea, miso_array, miso_matrix = fs.generate_train_init(filename_test, grain_shape, grain_sizes, device)
    fp_save = './data/spparks_ic_shape(%s)_nsteps(%d)_kt(%.2f)_cut(%d).h5'%(ic_shape, nsteps, 0.66, 0.0)    
    ims_id, fp_spparks = fs.run_spparks(ic, ea, nsteps=nsteps, kt=0.66, cut=0.0, fp_save = fp_save) # 1000
    fs.compute_grain_stats(fp_spparks) # gps='sim0'
    fs.make_videos(fp_spparks, sub_folder, ic_shape) #saves to 'plots'
    if grain_shape == "grain":
        fs.make_time_plots(fp_spparks, sub_folder, ic_shape) #saves to 'plots'
