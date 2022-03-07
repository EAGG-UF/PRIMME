#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 10:24:51 2022

@author: joseph.melville
"""


import functions as fs
import os


fp = './data_training/'
if not os.path.exists(fp): os.makedirs(fp)


size = [257,257]
ngrains_range = [256, 256]
nsets = 200
future_steps = 4
max_steps = 100
offset_steps = 1
fp_data = fp + 'spparks_data_size%dx%d_ngrain%d-%d_nsets%d_future%d_max%d_offset%d_kt05'%(size[0],size[1],ngrains_range[0],ngrains_range[1],nsets,future_steps,max_steps,offset_steps)


fs.create_SPPARKS_dataset(fp_data, size, ngrains_range, nsets, future_steps, max_steps, offset_steps)