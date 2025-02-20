#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IF THIS CODE IS USED FOR A RESEARCH PUBLICATION, please cite:
    Yan, W., Melville, J., Yadav, V., Everett, K., Yang, L., Kesler, M. S., ... & Harley, J. B. (2022). A novel physics-regularized interpretable machine learning model for grain growth. Materials & Design, 222, 111032.
"""

### Import functions
import functions as fs
import torch
import PRIMME

# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# for mac, do this:
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

### Train PRIMME using the training set from SPPARKS
trainset_location = "./data/spparks_data_size257x257_ngrain256-256_nsets200_future4_max100_offset1_kt05.h5"
# model_location = PRIMME.train_primme(trainset_location, num_eps=10, obs_dim=17, act_dim=17, lr=5e-5, reg=1, pad_mode="circular")
# model_location = "./data/model_dim(3)_sz(9_9)_lr(5e-05)_reg(1)_ep(1000)_kt(0.66)_freq(0.5)_cut(0).h5"
ic_shape = "grain(1024-4096)" 

### VALIDATION

## Choose initial conditions
ic, ea, _ = fs.voronoi2image(size=[256, 256], ngrain=512) #nsteps=500, pad_mode='circular'

## Run PRIMME model
fp_primme = PRIMME.run_primme(ic, ea, nsteps=100, modelname=model_location, pad_mode='circular')# fp_primme = "./data/primme_sz(512x512)_ng(512)_nsteps(3)_freq(1)_kt(0.66)_cut(25).h5"
fs.compute_grain_stats(fp_primme)
fs.make_videos(fp_primme) #saves to 'plots'
fs.make_time_plots(fp_primme) #saves to 'plots'