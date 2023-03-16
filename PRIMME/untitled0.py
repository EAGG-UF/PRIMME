#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 16:44:27 2023

@author: joseph.melville
"""
import functions as fs

import h5py

fp = './data/primme_sz(128x128x128)_ng(8192)_nsteps(1000)_freq(1)_kt(0.66)_cut(0).h5'
with h5py.File(fp, 'r') as f:
    ic = f['sim0/ims_id'][0,0]
    ea = f['sim0/euler_angles'][:]
    

import matplotlib.pyplot as plt
plt.plot(aaa)

#Then run the large spparks 3D simulation
#Compare primme, spparks, mf statistically
#(try again with new spparks training set if needed)


## Run SPPARKS model
fp_spparks = fs.run_spparks(ic, ea, nsteps=100, kt=0.66, cut=0.0, freq=(.1,.1), which_sim='eng', num_processors=32)
fs.compute_grain_stats(fp_spparks) 
# fs.make_videos(fp_spparks) #saves to 'plots'
fs.make_time_plots(fp_spparks) #saves to 'plots'