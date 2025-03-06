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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

trainset_dict = {"case1": [r"T:\trainset_spparks_sz(257x257)_ng(256-256)_nsets(200)_future(4)_max(100)_kt(0.66)_cut(0).h5"]}
n_samples_dict = {"case1": [[200]]}
n_step_dict = {"case1": [[5]]}


test_case_dict = {"case6": ["grain", [[512, 512], 512]]} # "case8": ["grain", [[2400, 2400], 24000]]

# ### Train PRIMME using the above training set from SPPARKS
for key_data in trainset_dict.keys():
    trainset = trainset_dict[key_data][0]
    for n_samples in n_samples_dict[key_data]:
        for n_step in n_step_dict[key_data]:
            modelname = PRIMME.train_primme(trainset, n_step, n_samples, test_case_dict)
