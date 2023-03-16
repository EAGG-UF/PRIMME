#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 15:56:14 2023

@author: joseph.melville
"""


        
               
        
        
        
from tqdm import tqdm
import numpy as np
import torch
import functions as fs
import matplotlib.pyplot as plt
import h5py





ic, ea, _ = fs.voronoi2image(size=[1024, 1024], ngrain=2**12) #nsteps=1000, pad_mode='circular'

# model_location = "./data/model_dim(2)_sz(17_17)_lr(5e-05)_reg(1)_ep(200)_kt(0.66)_cut(0).h5"
# ims_id, fp_primme = fs.run_primme(ic, ea, nsteps=100, modelname=model_location, pad_mode='circular', if_plot=False)

model_location = "./data/model_dim(2)_sz(17_17)_lr(5e-05)_reg(1)_ep(200)_kt(0.66)_cut(0).h5"
miso_matrix = np.ones([4096,4096])
miso_array = fs.miso_matrix_to_array(miso_matrix)
ims_id, fp_primme = fs.run_primme(ic, ea, nsteps=100, modelname=model_location, miso_array=miso_array, pad_mode='circular', if_plot=False)

ims_id, fp_primme = fs.run_primme(ic, ea, nsteps=100, modelname=model_location, pad_mode='circular', if_plot=False)



#cut(0) - trained on cut=0, sim0 uses number of different neighbors, sim1 uses neighborhood miso
#cut(25) - trained on cut=25, sim0 uses number of different neighbors, sim1 uses neighborhood miso




#miso features, no miso features, miso features and miso_array=1

# Plot change in avg miso over time
with h5py.File('./data/primme_sz(1024x1024)_ng(4096)_nsteps(100)_freq(1)_kt(0.66)_cut(0).h5', 'r') as f:
    m0 = f['sim0/ims_miso_avg'][:]
    d0 = f['sim0/dihedral_std'][:]
    
with h5py.File('./data/primme_sz(1024x1024)_ng(4096)_nsteps(100)_freq(1)_kt(0.66)_cut(25).h5', 'r') as f:
    m25 = f['sim0/ims_miso_avg'][:]
    d25 = f['sim0/dihedral_std'][:]

plt.plot(m0)
plt.plot(m25)
plt.show()

plt.plot(d0)
plt.plot(d25)
plt.show()

#Does spparks look the same?


h0 = './data/spparks_sz(1024x1024)_ng(4096)_nsteps(1000)_freq(1.0)_kt(0.66)_cut(0).h5'
h1 = './data/spparks_sz(1024x1024)_ng(4096)_nsteps(1000)_freq(1.0)_kt(0.66)_cut(25).h5'
h2 = './data/primme_sz(1024x1024)_ng(4096)_nsteps(100)_freq(1)_kt(0.66)_cut(0).h5'
h3 = './data/primme_sz(1024x1024)_ng(4096)_nsteps(100)_freq(1)_kt(0.66)_cut(25).h5'





hps = [h2, h3]
gps = ['sim0','sim0']
fs.make_time_plots(hps)

fs.compute_grain_stats(hps, gps)


# Plot the change in std of diheral angles over time
tmp0 = []
tmp25 = []
for i in tqdm(np.arange(100,1100,100)):
    with h5py.File('./data/primme_sz(1024x1024)_ng(4096)_nsteps(1000)_freq(1)_kt(0.66)_cut(0).h5', 'r') as f:
        im0 = torch.Tensor(f['sim0/ims_id'][i][None,].astype('int'))
        mm = f['sim0/miso_matrix'][:]
    
    with h5py.File('./data/primme_sz(1024x1024)_ng(4096)_nsteps(1000)_freq(1)_kt(0.66)_cut(25).h5', 'r') as f:
        im25 = torch.Tensor(f['sim0/ims_id'][i][None,].astype('int'))
    
    da0 = fs.find_dihedral_angles(im0, if_plot=False, num_plot_jct=10)
    da25 = fs.find_dihedral_angles(im25, if_plot=False, num_plot_jct=10)

    t0 = da0[3:,:].flatten()
    t25 = da25[3:,:].flatten()
    
    tmp0.append(torch.std(t0))
    tmp25.append(torch.std(t25))

plt.plot(tmp0)
plt.plot(tmp25)
plt.show()





#Three grain test


ic, ea = fs.generate_3grainIC(size=[256,256], h=180) #nsteps=300, pad_mode=['circular', 'reflect']

model_location = "./data/model_dim(2)_sz(17_17)_lr(5e-05)_reg(1)_ep(200)_kt(0.66)_cut(0).h5"
miso_array = np.array([1,1,1]) 
ims_id, fp_primme = fs.run_primme(ic, ea, nsteps=300,miso_array=miso_array, modelname=model_location, pad_mode=['circular', 'reflect'], if_plot=False)
plt.imshow(ims_id[300,0])

model_location = "./data/model_dim(2)_sz(17_17)_lr(5e-05)_reg(1)_ep(200)_kt(0.66)_cut(0).h5"
miso_array = np.array([0,1,1])
ims_id, fp_primme = fs.run_primme(ic, ea, nsteps=300,miso_array=miso_array, modelname=model_location, pad_mode=['circular', 'reflect'], if_plot=False)
plt.imshow(ims_id[300,0])

model_location = "./data/model_dim(2)_sz(17_17)_lr(5e-05)_reg(1)_ep(200)_kt(0.66)_cut(25).h5"
miso_array = np.array([1,1,1]) 
ims_id, fp_primme = fs.run_primme(ic, ea, nsteps=300,miso_array=miso_array, modelname=model_location, pad_mode=['circular', 'reflect'], if_plot=False)
plt.imshow(ims_id[300,0])

model_location = "./data/model_dim(2)_sz(17_17)_lr(5e-05)_reg(1)_ep(200)_kt(0.66)_cut(25).h5"
miso_array = np.array([0,1,1])
ims_id, fp_primme = fs.run_primme(ic, ea, nsteps=300,miso_array=miso_array, modelname=model_location, pad_mode=['circular', 'reflect'], if_plot=False)
plt.imshow(ims_id[300,0])






#Hex
ic, ea = fs.generate_hexIC() #nsteps=500, pad_mode='circular'

model_location = "./data/model_dim(2)_sz(17_17)_lr(5e-05)_reg(1)_ep(200)_kt(0.66)_cut(25).h5"
miso_matrix = np.ones([64,64])
miso_array = fs.miso_matrix_to_array(miso_matrix)
ims_id, fp_primme = fs.run_primme(ic, ea, nsteps=300,miso_array=miso_array, modelname=model_location, pad_mode='circular', if_plot=False)
plt.imshow(ims_id[300,0])

model_location = "./data/model_dim(2)_sz(17_17)_lr(5e-05)_reg(1)_ep(200)_kt(0.66)_cut(25).h5"
miso_matrix = np.ones([64,64])
miso_matrix[15,:] = 0.1
miso_matrix[:,15] = 0.1
miso_array = fs.miso_matrix_to_array(miso_matrix)
ims_id, fp_primme = fs.run_primme(ic, ea, nsteps=100,miso_array=miso_array, modelname=model_location, pad_mode='circular', if_plot=False)
plt.imshow(ims_id[100,0]==15)
plt.imshow(ims_id[100,0])

model_location = "./data/model_dim(2)_sz(17_17)_lr(5e-05)_reg(1)_ep(200)_kt(0.66)_cut(25).h5"
miso_matrix = np.ones([64,64])*0.1
miso_matrix[15,:] = 1
miso_matrix[:,15] = 1
miso_array = fs.miso_matrix_to_array(miso_matrix)
ims_id, fp_primme = fs.run_primme(ic, ea, nsteps=100,miso_array=miso_array, modelname=model_location, pad_mode='circular', if_plot=False)
plt.imshow(ims_id[100,0])

plt.imshow(ic==15)








