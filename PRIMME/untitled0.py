#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 15:42:55 2023

@author: joseph.melville
"""
# in this console, I would like to figure out my batching
# I want to use everything the way it is, but just split the image into sections
# I want to create a generator to return an image in padded chunks
# Then I can 



import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
import functions as fs


#make sure it works for 3d


data, _, _ = fs.voronoi2image(size=[32,32,32], ngrain=512)

slices_txt = '-1, 3:100, 10:15'

tmp = fs.wrap_slice(data, slices_txt)

tmp.shape


eval('data['+slices_txt+'].shape')


low = [-1, -3, -10]
high = [-0, 12, 5]
slices_txt = '-1, -3:12, -10:5'

tmp = fs.wrap_slice(data, slices_txt)
    
iii = torch.meshgrid(torch.arange(low[0],high[0]),torch.arange(low[1],high[1]),torch.arange(low[2],high[2]))
tmp0 = data[:][iii]

t = tmp
t0 = tmp0.squeeze()
(t!=t0).sum()








fp ='./data/trainset_spparks_sz(257x257)_ng(256-256)_nsets(200)_future(4)_max(100)_kt(0.66)_cut(0).h5'
with h5py.File(fp, 'r') as f:
    data = f['ims_id']
    
    tmp = fs.wrap_slice(data, slices_txt)
    
    iii = torch.meshgrid(torch.arange(low[0],high[0]),torch.arange(low[1],high[1]),torch.arange(low[2],high[2]),torch.arange(low[3],high[3]),torch.arange(low[4],high[4]))
    tmp0 = data[:][iii]

t = tmp
t0 = tmp0.squeeze()
(t!=t0).sum()




plt.imshow(t[0,0,0])













#For training - grab a batch size and run with it
#For stepping





#incorperate into primme
#run final isotropic test, make comparisons

#Tomorrow:
#revalidate isotropic with batching
#ani 2d tests
#iso 3d tests



















