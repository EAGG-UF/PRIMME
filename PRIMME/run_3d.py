#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IF THIS CODE IS USED FOR A RESEARCH PUBLICATION, please cite:
    Yan, W., Melville, J., Yadav, V., Everett, K., Yang, L., Kesler, M. S., ... & Harley, J. B. (2022). A novel physics-regularized interpretable machine learning model for grain growth. Materials & Design, 222, 111032.
"""

### Import functions
import functions as fs



### Train PRIMME using the training set from SPPARKS
trainset_location = "./data/trainset_spparks_sz(32x32x32)_ng(16-16)_nsets(3)_future(4)_max(20)_kt(0.66)_cut(25).h5"
model_location = fs.train_primme(trainset_location, num_eps=3, obs_dim=17, act_dim=17, lr=5e-5, reg=1, pad_mode="circular", if_plot=True)
# model_location = "./data/model_dim(3)_sz(17_17)_lr(5e-05)_reg(1)_ep(3)_kt(0.66)_cut(25).h5"



### VALIDATION

## Choose initial conditions
ic, ea, _ = fs.voronoi2image(size=[32, 32, 32], ngrain=512) #nsteps=500, pad_mode='circular'

## Run PRIMME model
ims_id, fp_primme = fs.run_primme(ic, ea, nsteps=3, modelname=model_location, pad_mode='circular', if_plot=True)
# fp_primme = "./data/primme_sz(32x32x32)_ng(512)_nsteps(3)_freq(1)_kt(0.66)_cut(25).h5"
fs.compute_grain_stats(fp_primme)
fs.make_videos(fp_primme) #saves to 'plots'
fs.make_time_plots(fp_primme) #saves to 'plots'