#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 04:12:31 2023

@author: yang.kang
"""

# IMPORT PACKAGES

import os.path
import torch
import matplotlib.pyplot as plt
import functions as fs
import PRIMME
from pathlib import Path

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

### Create training set by running SPPARKS
# trainset_location = fs.create_SPPARKS_dataset(size=[257,257], ngrains_rng=[256, 256], kt=0.66, cutoff=0.0, nsets=200, max_steps=100, offset_steps=1, future_steps=4)

# Define training set and model locations
trainset = "./data/trainset_spparks_sz(257x257)_ng(256-256)_nsets(200)_future(4)_max(100)_kt(0.66)_cut(0).h5"
model_location = "./data/model_dim(2)_sz(17_17)_lr(5e-05)_reg(1)_ep(1000)_kt(0.66)_cut(0).h5"
fp_primme = "./data/primme_shape(grain(512_512_512))_model_dim(2)_sz(17_17)_lr(5e-05)_reg(1)_ep(1000)_kt(0.66)_cut(0).h5"

# Define test case parameters
nsteps = 800
grain_shape = "grain" # Alternatives include "circle", "hex", "square"
grain_sizes = [[512, 512], 512] # Also tested for 257x257, 1024x1024, 2048x2048, 2400x2400
ic_shape = f"{grain_shape}({grain_sizes[0][0]}_{grain_sizes[0][1]}_{grain_sizes[1]})" if grain_shape != "hex" else "hex"

# Define filename for potential saved data
filename_test = f"{ic_shape}.pickle"
path_load = Path('./data') / filename_test

# Load or generate initial conditions and misorientation data
if path_load.is_file():
    data_dict = fs.load_picke_files(load_dir=Path('./data'), filename_save=filename_test)
    ic, ea, miso_array, miso_matrix = data_dict["ic"], data_dict["ea"], data_dict["miso_array"], data_dict["miso_matrix"]
else:
    ic, ea, miso_array, miso_matrix = fs.generate_train_init(filename_test, grain_shape, grain_sizes, device)

### Train PRIMME using the above training set from SPPARKS
model_location = PRIMME.train_primme(trainset, n_step=nsteps, n_samples=200, mode="Single_Step", num_eps=100, dims=2, obs_dim=17, act_dim=17, lr=5e-5, reg=1, pad_mode="circular", if_plot=False)

# Run PRIMME model
ims_id, fp_primme = PRIMME.run_primme(ic, ea, miso_array, miso_matrix, nsteps=nsteps, ic_shape=ic_shape, modelname=model_location, pad_mode='circular', if_plot=False)

# Generate plots
fs.compute_grain_stats(fp_primme)
fs.make_videos(fp_primme, ic_shape=ic_shape) #saves to 'plots'
fs.make_time_plots(fp_primme, ic_shape=ic_shape) #saves to 'plots'