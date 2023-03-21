#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 09:56:02 2023

@author: joseph.melville
"""



import functions as fs



# trainset = './data/trainset_spparks_sz(257x257)_ng(256-256)_nsets(200)_future(4)_max(100)_kt(0.66)_cut(0).h5'

# trainset = './data/trainset_spparks_sz(93x93x93)_ng(256-256)_nsets(200)_future(4)_max(50)_kt(0.66)_freq(0.5)_cut(0).h5'
# fs.train_primme(trainset, 100, batch_size=10000, reg=1, if_plot=True)


import h5py
fp ='./data/spparks_sz(1024x1024)_ng(4096)_nsteps(1000)_freq(1.0)_kt(0.66)_cut(0).h5'
with h5py.File(fp, 'r') as f:
    ic = f['sim0/ims_id'][0,0,].astype(int)
    ea = f['sim0/euler_angles'][:]

model_location = "./data/model_dim(2)_sz(17_17)_lr(5e-05)_reg(1)_ep(100)_kt(0.66)_cut(0).h5"
ims_id, fp_primme = fs.run_primme(ic, ea, nsteps=1000, modelname=model_location, pad_mode='circular', if_plot=False)
fs.compute_grain_stats(fp_primme)


fps = ['./data/spparks_sz(1024x1024)_ng(4096)_nsteps(1000)_freq(1.0)_kt(0.66)_cut(0).h5',
       './data/primme_sz(1024x1024)_ng(4096)_nsteps(1000)_freq(1)_kt(0.66)_cut(0).h5']
fs.compute_grain_stats(fps)
fs.make_videos(fp_primme) #saves to 'plots'
fs.make_time_plots(fps) 






# fp ='./data/primme_sz(1024x1024)_ng(4096)_nsteps(1000)_freq(1)_kt(0.66)_cut(0).h5'
# with h5py.File(fp, 'r') as f:
#     im = f['sim0/ims_id'][900,0,]

# import matplotlib.pyplot as plt
# plt.imshow(im)






#validate
#change to pytorch


#implement better training and validation split
#save model with lowest validation loss


#could I regulate better, more traditionally
#do I need future steps

#ani
#3D

    