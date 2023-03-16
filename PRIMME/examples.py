#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IF THIS CODE IS USED FOR A RESEARCH PUBLICATION, please cite:
    Yan, W., Melville, J., Yadav, V., Everett, K., Yang, L., Kesler, M. S., ... & Harley, J. B. (2022). A novel physics-regularized interpretable machine learning model for grain growth. Materials & Design, 222, 111032.
"""

### Import functions
import functions as fs










### Create training set by running SPPARKS
# trainset_location = fs.create_SPPARKS_dataset(size=[93,93,93], ngrains_rng=[256, 256], kt=0.66, cutoff=0.0, nsets=200, max_steps=50, offset_steps=1, future_steps=4, freq = (.5,.5))


### Train PRIMME using the above training set from SPPARKS
trainset_location = './data/trainset_spparks_sz(93x93x93)_ng(256-256)_nsets(200)_future(4)_max(50)_kt(0.66)_freq(0.5)_cut(0).h5'
model_location = fs.train_primme(trainset_location, num_eps=200, obs_dim=17, act_dim=17, lr=5e-5, reg=1, pad_mode="circular", if_plot=True)



#batch training so it can run on large structures
#what size 

import numpy as np
np.sqrt(256**2/256/np.pi)

np.cbrt(93**3/256*3/4/np.pi)


#kept number of grains and average grain radius the same


np.sqrt(1024**2/4096/np.pi)

np.cbrt(233**3/4096*3/4/np.pi)


ic, ea, _ = fs.voronoi2image(size=[233, 233, 233], ngrain=4096) #nsteps=500, pad_mode='circular'



### VALIDATION

## Choose initial conditions
# ic, ea = fs.generate_circleIC(size=[257,257], r=64) #nsteps=200, pad_mode='circular'
# ic, ea = fs.generate_3grainIC(size=[512,512], h=350) #nsteps=300, pad_mode=['circular', 'reflect']
# ic, ea = fs.generate_hexIC() #nsteps=500, pad_mode='circular'
# ic, ea, _ = fs.voronoi2image(size=[64, 64], ngrain=16) #nsteps=500, pad_mode='circular'
# ic, ea, _ = fs.voronoi2image(size=[128, 128, 128], ngrain=2*4096) #nsteps=500, pad_mode='circular'
# ic, ea, _ = fs.voronoi2image(size=[92, 92, 92], ngrain=256) #nsteps=500, pad_mode='circular'
# ic, ea, _ = fs.voronoi2image(size=[256, 256], ngrain=256) #nsteps=500, pad_mode='circular'
# ic, ea, _ = fs.voronoi2image(size=[128,128,128], ngrain=8192) #nsteps=500, pad_mode='circular'
# ic, ea, _ = fs.voronoi2image(size=[1024, 1024], ngrain=2**12) #nsteps=1000, pad_mode='circular'
# ic, ea, _ = fs.voronoi2image(size=[2048, 2048], ngrain=2**14) #nsteps=1500, pad_mode='circular'


## Run PRIMME model
# model_location = "./data/model_dim(2)_sz(17_17)_lr(5e-05)_reg(1)_ep(200)_kt(0.66)_cut(0).h5"
model_location = "./data/model_dim(3)_sz(7_7)_lr(5e-05)_reg(1)_ep(200)_kt(0.66)_cut(0).h5"
ims_id, fp_primme = fs.run_primme(ic, ea, nsteps=1000, modelname=model_location, pad_mode='circular', if_plot=False)
fs.compute_grain_stats(fp_primme)
# fs.make_videos(fp_primme) #saves to 'plots'
fs.make_time_plots(fp_primme) #saves to 'plots'











#Get initial condition
#Run mode filter
#Get SPPARKS running in parallel and show it;s working the same

import h5py

fp = './data/primme_sz(128x128x128)_ng(8192)_nsteps(1000)_freq(1)_kt(0.66)_cut(0).h5'
with h5py.File(fp, 'r') as f:
    ic = f['sim0/ims_id'][0,0]
    ea = f['sim0/euler_angles'][:]
    

     


#Then run the large spparks 3D simulation
#Compare primme, spparks, mf statistically
#(try again with new spparks training set if needed)


## Run SPPARKS model
fp_spparks = fs.run_spparks(ic, ea, nsteps=100, kt=0.66, cut=0.0, freq=(.1,.1), which_sim='eng', num_processors=32)
fs.compute_grain_stats(fp_spparks) 
# fs.make_videos(fp_spparks) #saves to 'plots'
fs.make_time_plots(fp_spparks) #saves to 'plots'


fp_spparks = './data/spparks_sz(128x128x128)_ng(8192)_nsteps(100)_freq(0.1)_kt(0.66)_cut(0).h5'






fp0 = './data/spparks_sz(128x128x128)_ng(8192)_nsteps(100)_freq(0.1)_kt(0.66)_cut(0).h5'
fp1 = './data/primme_sz(128x128x128)_ng(8192)_nsteps(1000)_freq(1)_kt(0.66)_cut(0).h5'
hps = [fp0, fp1]
fs.make_time_plots(hps)


with h5py.File(fp1, 'r') as f: print(f.keys())


## Compare PRIMME and SPPARKS statistics
fp_spparks_iso = './data/spparks_sz(1024x1024)_ng(4096)_nsteps(1000)_freq(1)_kt(0.66)_cut(0).h5'
fp_primme_iso = './data/primme_sz(1024x1024)_ng(4096)_nsteps(1000)_freq(1)_kt(0.66)_cut(0).h5'
fp_spparks_ani = './data/spparks_sz(1024x1024)_ng(4096)_nsteps(1000)_freq(1)_kt(0.66)_cut(25).h5'
fp_primme_ani = './data/primme_sz(1024x1024)_ng(4096)_nsteps(1000)_freq(1)_kt(0.66)_cut(25).h5'
hps = [fp_spparks_iso, fp_primme_iso, fp_spparks_ani, fp_primme_ani]
fs.compute_grain_stats(hps)
fs.make_time_plots(hps)

hp0 = './data/spparks_sz(1024x1024)_ng(4096)_nsteps(1000)_freq(1)_kt(0.66)_cut(0).h5'
hp1 = './data/spparks_sz(1024x1024)_ng(4096)_nsteps(1000)_freq(1.0)_kt(0.66)_cut(0).h5'

hps = [hp0, hp1]
fs.make_time_plots(hps)












