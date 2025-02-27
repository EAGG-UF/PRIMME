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

### Create training set by running SPPARKS
# trainset_location = fs.create_SPPARKS_dataset(size=[257,257], ngrains_rng=[256, 256], kt=0.66, cutoff=0.0, nsets=200, max_steps=100, offset_steps=1, future_steps=4)

### Train PRIMME using the above training set from SPPARKS
# trainset = "./data/trainset_spparks_sz(257x257)_ng(256-256)_nsets(200)_future(4)_max(100)_kt(0.66)_cut(0).h5"
# model_location = PRIMME.train_primme(trainset, num_eps=1000, dims=2, obs_dim=17, act_dim=17, lr=5e-5, reg=1, pad_mode="circular", if_plot=True)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# MAC Fix
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

### VALIDATION

## Choose initial conditions
ic, ea = fs.generate_circleIC(size=[257,257], r=64) #nsteps=200, pad_mode='circular'
# ic, ea = fs.generate_circleIC(size=[512,512], r=200) #nsteps=200, pad_mode='circular'
# ic, ea = fs.generate_3grainIC(size=[512,512], h=350) #nsteps=300, pad_mode=['reflect', 'circular']
# ic, ea = fs.generate_hexIC() #nsteps=500, pad_mode='circular'
# ic, ea = fs.generate_SquareIC(size=[512,512], r=64) 
# ic, ea, _ = fs.voronoi2image(size=[512, 512], ngrain=512) #nsteps=500, pad_mode='circular'
# ic, ea, _ = fs.voronoi2image(size=[1024, 1024], ngrain=2**12) #nsteps=1000, pad_mode='circular'
# ic, ea, _ = fs.voronoi2image(size=[2048, 2048], ngrain=2**14) #nsteps=1500, pad_mode='circular'
# ic, ea, _ = fs.voronoi2image(size=[2400, 2400], ngrain=24000) #nsteps=1500, pad_mode='circular'


trainset = "./data/trainset_spparks_sz(257x257)_ng(256-256)_nsets(200)_future(4)_max(100)_kt(0.66)_cut(0).h5"
## Run PRIMME model
model_location = "./data/model_dim(2)_sz(17_17)_lr(5e-05)_reg(1)_ep(1000)_kt(0.66)_cut(0).h5"
ic_shape = "grain(512-512)"

nsteps = 800
test_case_dict = {"case6": ["grain", [[512, 512], 512]]}
for key in test_case_dict.keys():
    grain_shape, grain_sizes = test_case_dict[key]
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

# def train_primme(trainset, n_step, n_samples, test_case_dict, mode = "Single_Step", num_eps=25, dims=2, obs_dim=17, act_dim=17, lr=5e-5, reg=1, pad_mode="circular", if_plot=False):
model_location = PRIMME.train_primme(trainset, n_step=1000, n_samples=200, test_case_dict=test_case_dict, mode="Single_Step", num_eps=100, dims=2, obs_dim=17, act_dim=17, lr=5e-5, reg=1, pad_mode="circular", if_plot=False)

# def run_primme(ic, ea, miso_array, miso_matrix, nsteps, ic_shape, modelname, pad_mode='circular',  mode = "Single_Step", if_plot=False):
ims_id, fp_primme = PRIMME.run_primme(ic, ea, miso_array, miso_matrix, nsteps=1800, ic_shape=ic_shape, modelname=model_location, pad_mode='circular', if_plot=False)
# run_primme(ic, ea, miso_array, miso_matrix, nsteps, ic_shape, modelname, pad_mode='circular',  mode = "Single_Step", if_plot=False):
fs.compute_grain_stats(fp_primme)
fs.make_videos(fp_primme) #saves to 'plots'
fs.make_time_plots(fp_primme) #saves to 'plots'


## Run SPPARKS model
# ims_id, fp_spparks = fs.run_spparks(ic, ea, nsteps=1000, kt=0.66, cut=0.0)
# fs.compute_grain_stats(fp_spparks) 
# fs.make_videos(fp_spparks) #saves to 'plots'
# fs.make_time_plots(fp_spparks) #saves to 'plots'
