#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DESCRIPTION:
    This script is used to validate the PRIMME model
    It simulates grain growth with PRIMME using various intial conditions and generates grain growth statistics to compare to SPPARKS
    A SPPARKS environment does not need to be set up for this script to run properly
    All simulations and validation files are saved to "./results_validation" 

IF THIS CODE IS USED FOR A RESEARCH PUBLICATION, please cite (https://arxiv.org/abs/2203.03735): 
    Yan, Weishi, et al. "Predicting 2D Normal Grain Growth using a Physics-Regularized Interpretable Machine Learning Model." arXiv preprint arXiv:2203.03735 (2022).
"""



# IMPORT PACKAGES
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from PRIMME import PRIMME
import torch
import imageio
import functions as fs
import h5py
import os


num_episodes = 1000
obs_dim=9
act_dim=9
pad_mode="circular"
learning_rate=0.00005
num_dims=2

# SET UP FILE TO LOAD
#model_name = 'primme_episodes(200)_obs(31)_act(9)_pad(circular)_lr(5e-05)_dim(2)2'
#model_name = 'primmepy_ep(10000)_obs(31)_act(9)_pad(circular)_lr(5e-05)_dim(2)2'
model_name = "primmepy_ep(%s)_obs(%s)_act(%s)_pad(%s)_lr(%s)_dim(%s)2"%(str(num_episodes),str(obs_dim),str(act_dim),str(pad_mode),str(learning_rate), str(num_dims))


# SET UP SIMULATOR
agent = PRIMME(obs_dim=obs_dim, act_dim=act_dim, pad_mode='circular', learning_rate=0.00005, num_dims=2)
#agent.load('./saved_models/%s'%model_name)
agent.load_state_dict(torch.load('./saved_models/%s'%model_name))
fp_sims = './results_validation/'
fp_val = './data_validation/'  
if not os.path.exists(fp_sims): os.makedirs(fp_sims)


# CREATE TEST CASE INITIAL CONDITION
for i in [1]: #range(8): #make 8 to run all tests (0 through 7)

    if i==0:
        f = "Case1_circle30" #file name
        size = [70, 70] #size of each dimension in SPPARKS
        pad_mode = "circular"
        img, EulerAngles = fs.generate_circleIC(size=size, r=30)
        num_steps = 100

    if i==1:
        f = "Case1_circle64" #file name
        size = [200, 200] #size of each dimension in SPPARKS
        pad_mode = "circular"
        img, EulerAngles = fs.generate_circleIC(size=size, r=64)
        num_steps = 300

    if i==2:
        f = "Case1_circle200" #file name
        size = [512, 512] #size of each dimension in SPPARKS
        pad_mode = "circular"
        img, EulerAngles = fs.generate_circleIC(size=size, r=200)
        num_steps = 1000

    if i==3:
        f = "Case1_3grain" #file name
        size = [512, 512] #size of each dimension in SPPARKS
        pad_mode = ["circular", "reflect"]
        img, euler_angles = fs.generate_3grainIC(size=size, h=350)
        num_steps = 1000

    if i==4:
        f = "Case2_hex" #file name
        size = [443, 512] #size of each dimension in SPPARKS
        pad_mode = "circular"
        grain_centers = fs.read_grain_centers_txt(fp="%s/Case2_grain_centers"%fp_val)
        img, EulerAngles, center_coords = fs.voronoi2image(size=size, ngrain=64, center_coords0=grain_centers)
        num_steps = 500

    if i==5:
        f = "Case3_512p" #file name, periodic or 'circular' boundaries
        size = [512, 512] #size of each dimension in SPPARKS
        pad_mode = "circular"
        grain_centers = fs.read_grain_centers_txt(fp="%s/Case3_grain_centers"%fp_val)
        img, EulerAngles, center_coords = fs.voronoi2image(size=size, ngrain=grain_centers.shape[0], center_coords0=grain_centers)
        num_steps = 500

    if i==6:
        f = "Case3_512np" #file name, nonperiodic or 'reflect' boundaries
        size = [512, 512] #size of each dimension in SPPARKS
        pad_mode = "reflect"
        grain_centers = fs.read_grain_centers_txt(fp="%s/Case3_grain_centers"%fp_val)
        img, EulerAngles, center_coords = fs.voronoi2image(size=size, ngrain=grain_centers.shape[0], center_coords0=grain_centers)
        num_steps = 1500
    
    if i==7:
        f = "Case4_2400p" #file name, periodic or 'circular' boundaries
        size = [2400, 2400] #size of each dimension in SPPARKS
        pad_mode = "circular"
        grain_centers = fs.read_grain_centers_txt(fp="%s/Case4_grain_centers"%fp_val)
        img, EulerAngles, center_coords = fs.voronoi2image(size=size, ngrain=grain_centers.shape[0], center_coords0=grain_centers)
        # # np.save("Case4.npy", img)
        # img = np.load("Case4.npy")
        num_steps = 1500
        #
        # f = "Case4_512p3" #file name, periodic or 'circular' boundaries
        # size = [512, 512, 512] #size of each dimension in SPPARKS
        # pad_mode = "circular"
        # # img, EulerAngles, center_coords = fs.voronoi2image(size=size, ngrain=50000, memory_limit=30e9)
        # num_steps = 500
        # # np.save("512x512x512.npy", img)
        # img = np.load("512x512x512.npy")
        
        
        ##RUN THIS NEXT!!!
        # f = "Case4_512p3"
        # fp = './data_validation/'
        # fs.image2init(img, EulerAngles, fp=fp_val+f+'.init')
        # fs.write_grain_centers_txt(center_coords, fp=fp_val+f+'_grain_centers')


    # RUN SIMULATION
    agent.pad_mode = pad_mode
    im = torch.Tensor(img).unsqueeze(0).unsqueeze(0).float().to(agent.device)
    sim = [im.squeeze().cpu().numpy().astype(np.uint)]
    for _ in tqdm(range(num_steps), f):
        im = agent.step(im)[0]
        sim.append(im.squeeze().cpu().numpy().astype(np.uint))

        # print(len(torch.unique(im)))
        # plt.imshow(im[0,0,].cpu());
        # plt.savefig('image.png')
        #plt.show()


    # SAVE RESULTS
    
    # Save images to hdf5 file
    sim_arr = np.array(sim).astype(np.uint16)
    with h5py.File("%s/%s%s.hdf5"%(fp_sims, f, model_name), 'w') as fl:
        fl["images"] = sim_arr
        
    # Save to a gif
    sim_arr = np.array(sim).astype(np.uint8)
    #sim_arr = sim_arr[...,int(sim_arr.shape[-1]/2)] #bisect last dimension for 2D images
    sim_arr_norm = (255/np.max(sim_arr)*sim_arr).astype(np.uint8)
    imageio.mimsave('%s/%s%s.mp4' % (fp_sims, f, model_name), sim_arr_norm)
    #imageio.mimsave( '%s/%s.mp4'%(fp_sims, f), sim_arr_norm)
    
    # Plot the center of mass for nonzero values of each image in the sequence
    plt.close()
    if i==0 or i==1 or i==2 or i==4:
        c = []
        for e in sim:
            b = np.nonzero(e)
            c.append([np.mean(b[0]), np.mean(b[1])])
        c = np.array(c)
        plt.plot(c[:,0], c[:,1])
        plt.savefig('%s/%s%s_drift.png'%(fp_sims, f, model_name))


# # RUN STATISTICS FOR CASE 4
# if i==7: #if the Case4 simulation was run up above
#     fp = fp_sims
#     h5_path = '%s/Case4_2400p.hdf5'%fp
#     device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
#     arrays, array_stats = fs.apply_grain_func(h5_path, func=fs.grain_size, device=device)
#     np.savetxt("%s/grain_sizes.csv"%fp, arrays, delimiter=",")
#     np.savetxt("%s/grain_size_stats.csv"%fp, array_stats, delimiter=",")
    
#     arrays, array_stats = fs.apply_grain_func(h5_path, func=fs.grain_num_neighbors, device=device)
#     np.savetxt("%s/grain_num_neighbors.csv"%fp, arrays, delimiter=",")
#     np.savetxt("%s/grain_num_neighbor_stats.csv"%fp, array_stats, delimiter=",")