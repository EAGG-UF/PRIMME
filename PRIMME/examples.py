#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IF THIS CODE IS USED FOR A RESEARCH PUBLICATION, please cite:
    Yan, W., Melville, J., Yadav, V., Everett, K., Yang, L., Kesler, M. S., ... & Harley, J. B. (2022). A novel physics-regularized interpretable machine learning model for grain growth. Materials & Design, 222, 111032.
"""





### Import functions
import functions as fs
import PRIMME as fsp


### Create training set by running SPPARKS
# trainset_location = fs.create_SPPARKS_dataset(size=[257,257], ngrains_rng=[256, 256], kt=0.66, cutoff=25.0, nsets=200, max_steps=100, offset_steps=1, future_steps=4)


# ### Train PRIMME using the above training set from SPPARKS
# trainset_location = "./data/trainset_spparks_sz(257x257)_ng(256-256)_nsets(200)_future(4)_max(100)_kt(0.66)_cut(0).h5"
trainset_location = "./data/trainset_spparks_sz(93x93x93)_ng(256-256)_nsets(200)_future(4)_max(50)_kt(0.66)_freq(0.5)_cut(0).h5"
model_location = fsp.train_primme(trainset_location, num_eps=1000, obs_dim=17, act_dim=17, lr=5e-5, reg=1, pad_mode=None, plot_freq=100)



#Polcrystalline 
import h5py
fp ='./data/spparks_sz(1024x1024)_ng(4096)_nsteps(1000)_freq(1.0)_kt(0.66)_cut(0)_old.h5'
with h5py.File(fp, 'r') as f:
    ic = f['sim0/ims_id'][0,0,].astype(int)
    ea = f['sim0/euler_angles'][:]
    
    




model_location = "./data/model_dim(2)_sz(17_17)_lr(5e-05)_reg(1)_ep(1000)_kt(0.66)_cut(0).h5"
fp_primme = fsp.run_primme(ic, ea, nsteps=1000, modelname=model_location, pad_mode=None, plot_freq=50)
fs.compute_grain_stats(fp_primme)
# fs.make_videos(fp_primme) #saves to 'plots'
fs.make_time_plots(fp_primme) 


#Small circle
ic, ea = fs.generate_circleIC(size=[256,256], r=64)
fp_primme = fsp.run_primme(ic, ea, nsteps=500, modelname=model_location, pad_mode=None, plot_freq=50)
fs.circle_stats(fp_primme)
# fs.circle_videos(fp_primme)
fs.circle_plots(fp_primme)


#Large circle
ic, ea = fs.generate_circleIC(size=[512,512], r=200)
fp_primme = fsp.run_primme(ic, ea, nsteps=1000, modelname=model_location, pad_mode=None, plot_freq=50)
fs.circle_stats(fp_primme)
# fs.circle_videos(fp_primme)
fs.circle_plots(fp_primme)






fps = ['./data/sims/spparks_sz(1024x1024)_ng(4096)_nsteps(1000)_freq(1.0)_kt(0.66)_cut(0).h5',
        './data/sims/primme_sz(1024x1024)_ng(4096)_nsteps(1000)_freq(1)_kt(0.66)_cut(0)_0.h5',
        './data/sims/primme_sz(1024x1024)_ng(4096)_nsteps(1000)_freq(1)_kt(0.66)_cut(0)_37.h5']
fs.make_time_plots(fps) 







# fps = ['./data/sims/spparks_sz(512x512)_ng(2)_nsteps(1000)_freq(1.0)_kt(0.66)_cut(0).h5',
#         './data/sims/primme_sz(512x512)_ng(2)_nsteps(1000)_freq(1)_kt(0.66)_cut(0)_20.h5']
# fs.circle_plots(fps)





# fps = ['./data/sims/spparks_sz(256x256)_ng(2)_nsteps(500)_freq(1.0)_kt(0.66)_cut(0).h5',
#         './data/sims/primme_sz(256x256)_ng(2)_nsteps(500)_freq(1)_kt(0.66)_cut(0)_0.h5']
# fs.circle_plots(fps)














### VALIDATION

## Choose initial conditions
# ic, ea = fs.generate_circleIC(size=[257,257], r=64) #nsteps=200, pad_mode='circular'
# ic, ea = fs.generate_3grainIC(size=[512,512], h=350) #nsteps=300, pad_mode=['reflect', 'circular']
# ic, ea = fs.generate_hexIC() #nsteps=500, pad_mode='circular'
# ic, ea, _ = fs.voronoi2image(size=[512, 512], ngrain=512) #nsteps=500, pad_mode='circular'
# ic, ea, _ = fs.voronoi2image(size=[1024, 1024], ngrain=2**12) #nsteps=1000, pad_mode='circular'
# ic, ea, _ = fs.voronoi2image(size=[2048, 2048], ngrain=2**14) #nsteps=1500, pad_mode='circular'


## Run PRIMME model
# model_location = "./data/model_dim(2)_sz(17_17)_lr(5e-05)_reg(1)_ep(100)_kt(0.66)_cut(0).h5"
# ims_id, fp_primme = fsp.run_primme(ic, ea, nsteps=1000, modelname=model_location, pad_mode='circular', plot_freq=20)
# fs.compute_grain_stats(fp_primme)
# fs.make_videos(fp_primme) #saves to 'plots'
# fs.make_time_plots(fp_primme) #saves to 'plots'


## Run SPPARKS model
# fp_spparks = fs.run_spparks(ic, ea, nsteps=1000, kt=0.66, cut=0.0)
# fs.compute_grain_stats(fp_spparks) 
# fs.make_videos(fp_spparks) #saves to 'plots'
# fs.make_time_plots(fp_spparks) #saves to 'plots'


## Compare PRIMME and SPPARKS statistics
# fp_spparks = './data/spparks_sz(1024x1024)_ng(4096)_nsteps(1000)_freq(1)_kt(0.66)_cut(0).h5'
# fp_primme = './data/primme_sz(1024x1024)_ng(4096)_nsteps(1000)_freq(1)_kt(0.66)_cut(0).h5'
# hps = [fp_spparks, fp_primme]
# fs.make_time_plots(hps)


## Cut number of features to create new trainset
# fp = './data/trainset_spparks_sz(257x257)_ng(256-256)_nsets(200)_future(4)_max(100)_kt(0.66)_cut(0).h5'
# window_size=17
# cut_f = 5000000
# nf = fs.trainset_calcNumFeatures(fp, window_size)
# fs.trainset_cutNumFeatures(fp, window_size, cut_f)