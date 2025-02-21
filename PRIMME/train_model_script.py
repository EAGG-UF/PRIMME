#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DESCRIPTION:
    This script is used to train the PRIMME model using data from SPPARKS, generated in real time.
    A SPPARKS environment must be set up for this script to run properly. Please see ./spparks_files/Getting_Started
    The model is saved to "./saved models/" after each training epoch
    Training evaluation files are saved to "./results_training" after training is complete

IF THIS CODE IS USED FOR A RESEARCH PUBLICATION, please cite (https://arxiv.org/abs/2203.03735): 
    Yan, Weishi, et al. "Predicting 2D Normal Grain Growth using a Physics-Regularized Interpretable Machine Learning Model." arXiv preprint arXiv:2203.03735 (2022).
"""

import h5py
import numpy as np
import functions as fs
import torch
import os
import matplotlib.pyplot as plt
import PRIMME


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

num_episodes = 1000
obs_dim = 17
act_dim = 17
pad_mode ="circular"
lr = 0.00005
dims = 2
reg = 1
num_eps = 1
trainset = './data/trainset_spparks_sz(257x257)_ng(256-256)_nsets(200)_future(4)_max(100)_kt(0.66)_cut(0).h5'

append_name = trainset.split('_kt')[1]
modelname = "./data/model_dim(%d)_sz(%d_%d)_lr(%.0e)_reg(%d)_ep(%d)_kt%s"%(dims, obs_dim, act_dim, lr, reg, num_eps, append_name)
agent = PRIMME(obs_dim=obs_dim, act_dim=act_dim, pad_mode=pad_mode, learning_rate=lr, 
               num_dims=dims, device = device).to(device)

if os.path.exists(modelname):
    agent.load_state_dict(torch.load(modelname))


batch_size = 1
with h5py.File(trainset, 'r') as f:
    print("Keys: %s" % f.keys())          
    i_max = f['ims_id'].shape[0]
    i_batch = np.sort(np.random.randint(low=0, high=i_max, size=(batch_size,)))
    batch = f['ims_id'][i_batch,]
    miso_array = f['miso_array'][i_batch,] 
    ims_T =  np.array(list(f['ims_id']))

im_seq = torch.from_numpy(batch[0,].astype(float)).to(device)
miso_array = torch.from_numpy(miso_array.astype(float)).to(device)
miso_matrix = fs.miso_conversion(miso_array)

im = im_seq[0:1,]
#Compute features and labels
features = fs.compute_features(im_seq[0:1,], obs_dim=obs_dim, pad_mode=pad_mode)
labels = fs.compute_labels(im_seq, obs_dim=obs_dim, act_dim=act_dim, reg=reg, pad_mode=pad_mode)

agent.step(im, miso_matrix)



print(agent.im_seq.shape)
plt.figure(figsize = (18, 5))
for n, i in enumerate([0, 2, 4]):
    plt.subplot(1, 3, n+1)
    plt.imshow(ims_T[i, 0, 0])
plt.show()

