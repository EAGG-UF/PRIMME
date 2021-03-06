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



# IMPORT LIBRARIES
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from PRIMME import PRIMME
from SPPARKS import SPPARKS
import torch
import imageio
import functions as fs
import h5py
import os



# SET UP SPPARKS ENVIRONEMENT
modelname = "./saved_models/primme_grains256_size257_episodes200_maxsteps100_obs17_act17_kt05"
fp_sims = './results_validation'
fp_val = './data_validation'  
if not os.path.exists(fp_sims): os.makedirs(fp_sims)


# CREATE TEST CASE INITIAL CONDITION
for i in range(8): #make 8 to run all tests (0 through 7)

    if i==0:
        f = "Case1_circle30" #file name
        size = [256, 256] #size of each dimension in SPPARKS
        pad_mode = "circular"
        img, EulerAngles = fs.generate_circleIC(size=size, r=30)
        num_steps = 100

    if i==1:
        f = "Case1_circle64" #file name
        size = [512, 512] #size of each dimension in SPPARKS
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


    # SET UP PRIMME SIMULATOR
    env = SPPARKS(size=size) 
    agent = PRIMME(env)
    agent.ID = torch.Tensor(img).unsqueeze(2).float().to(agent.device)
    agent.pad_mode = pad_mode
    agent.load(modelname)


    # RUN SIMULATION
    sim = []
    for _ in tqdm(range(num_steps), f):
        sim.append(agent.ID[...,0].cpu().numpy().astype(np.uint))
        agent.predict()
        agent.step()


    # SAVE RESULTS
    
    # Save images to hdf5 file
    sim_arr = np.array(sim).astype(np.uint16)
    with h5py.File("%s/%s.hdf5"%(fp_sims, f), 'w') as fl:
        fl["images"] = sim_arr
        
    # Save to a gif
    sim_arr = np.array(sim).astype(np.uint8)
    sim_arr_norm = int(255/np.max(sim_arr))*sim_arr
    imageio.mimsave('%s/%s.mp4'%(fp_sims, f), sim_arr_norm)
    
    # Plot the center of mass for nonzero values of each image in the sequence
    plt.close()
    if i==0 or i==1 or i==2 or i==4:
        c = []
        for e in sim:
            b = np.nonzero(e)
            c.append([np.mean(b[0]), np.mean(b[1])])
        c = np.array(c)
        plt.plot(c[:,0], c[:,1])
        plt.savefig('%s/%s_drift.png'%(fp_sims, f))



# RUN STATISTICS FOR CASE 4
if i==7: #if the Case4 simulation was run up above
    fp = fp_sims
    h5_path = '%s/Case4_2400p.hdf5'%fp
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    arrays, array_stats = fs.apply_grain_func(h5_path, func=fs.grain_size, device=device)
    np.savetxt("%s/grain_sizes.csv"%fp, arrays, delimiter=",")
    np.savetxt("%s/grain_size_stats.csv"%fp, array_stats, delimiter=",")
    
    arrays, array_stats = fs.apply_grain_func(h5_path, func=fs.grain_num_neighbors, device=device)
    np.savetxt("%s/grain_num_neighbors.csv"%fp, arrays, delimiter=",")
    np.savetxt("%s/grain_num_neighbor_stats.csv"%fp, array_stats, delimiter=",")
