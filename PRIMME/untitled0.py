#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 17:03:24 2023

@author: joseph.melville
"""

import h5py

fp = './data/spparks_sz(1024x1024)_ng(4096)_nsteps(1000)_freq(1.0)_kt(0.66)_cut(0)_old.h5'
with h5py.File(fp, 'r') as f:
            
    print(f['sim0/ims_id'].shape)
    
    ims = f['sim0/ims_id'][:5,]
    
fp = './data/test.h5'
with h5py.File(fp, 'a') as f:
    
    print(f.keys())
    
    # f['d0'] = ims
    f['d2'] = f['d0'][:]
    



import torch
import torch.nn as nn
import functions as fs
import matplotlib.pyplot as plt

ic, ea, _ = fs.voronoi2image(size=[1024, 1024], ngrain=2**12) 
im = torch.Tensor(ic)


pad = (8,8,8,8)

im = fs.pad_mixed(im[None,None], pad, pad_mode="circular")



batch_sz = 1000
kernel_sz = (1,1,17,17)
stride = (1,1,1,1)
if_shuffle = False


a = fs.unfold_in_batches(im, batch_sz, kernel_sz, stride, if_shuffle)



#unfolding has always been the memory hungry operation
#unfold in batches function works great (reducing memory)
#padding function works great with it
#can I now use these to batch primme without all the bugs?
#The plan is to rewrite the features and label functions to batch unfolding
#leave the other functions the same as they were before












l = []
for b in a:
    l.append(b[:,0,0,8,8])


ll = torch.cat(l).reshape(1,1,1024,1024)

(im[:,:,8:-8,8:-8]-ll).sum()

b = next(a)

plt.imshow(ll[0,0])


import os, psutil
process = psutil.Process()
print(process.memory_info().rss) 

