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
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
device = "cpu"
import functions as fs
import PRIMME

### Create training set by running SPPARKS
# trainset_location = fs.create_SPPARKS_dataset(size=[257,257], ngrains_rng=[256, 256], kt=0.66, cutoff=0.0, nsets=200, max_steps=100, offset_steps=1, future_steps=4)

### Train PRIMME using the above training set from SPPARKS
# trainset = "./data/trainset_spparks_sz(257x257)_ng(256-256)_nsets(200)_future(4)_max(100)_kt(0.66)_cut(0).h5"
# model_location = PRIMME.train_primme(trainset, num_eps=1000, dims=2, obs_dim=17, act_dim=17, lr=5e-5, reg=1, pad_mode="circular", if_plot=True)


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


## Run PRIMME model
model_location = "./data/model_dim(2)_sz(17_17)_lr(5e-05)_reg(1)_ep(1000)_kt(0.66)_cut(0).h5"
ic_shape = "grain(512-512)"
ims_id, fp_primme = PRIMME.run_primme(ic, ea, ic_shape, nsteps=1800, modelname=model_location, pad_mode='circular', if_plot=True)
fs.compute_grain_stats(fp_primme)
fs.make_videos(fp_primme) #saves to 'plots'
fs.make_time_plots(fp_primme) #saves to 'plots'


## Run SPPARKS model
# ims_id, fp_spparks = fs.run_spparks(ic, ea, nsteps=1000, kt=0.66, cut=0.0)
# fs.compute_grain_stats(fp_spparks) 
# fs.make_videos(fp_spparks) #saves to 'plots'
# fs.make_time_plots(fp_spparks) #saves to 'plots'


## Compare PRIMME and SPPARKS statistics
fp_spparks = './data/spparks_sz(1024x1024)_ng(4096)_nsteps(1000)_freq(1)_kt(0.66)_cut(0).h5'
fp_primme = './data/primme_sz(1024x1024)_ng(4096)_nsteps(1000)_freq(1)_kt(0.66)_cut(0).h5'
hps = [fp_spparks, fp_primme]
fs.make_time_plots(hps)
