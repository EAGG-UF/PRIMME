#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DESCRIPTION:
    The script defines a SPPARKS simulator class used to simulate microstructural growth
    This class is essential a wrapper for command line operations executed through python
    This class is passed into a PRIMME class to share an initial condition and provide data, features, and labels for training PRIMME
    The main functions of this class are to generate and initial condition, simulate grain growth through SPPARKS, and calculate features and labels from the SPPARKS simulation


CONTRIBUTORS: 
    Weishi Yan (1), Joel Harley (1), Joseph Melville (1), Kristien Everett (1), Lin Yang (2)

AFFILIATIONS:
    1. University of Florida, SmartDATA Lab, Department of Electrical and Computer Engineering
    2. University of Florida, Tonks Research Group, Department of Material Science and Engineering

FUNDING SPONSORS:
    U.S. Department of Energy, Office of Science, Basic Energy Sciences under Award \#DE-SC0020384
    U.S. Department of Defence through a Science, Mathematics, and Research for Transformation (SMART) scholarship
    
-------------------------------------------------------------------------
Copyright (C) 2021-2022  Joseph Melville

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or (at
your option) any later version. You should have received a copy of the
GNU General Public License along with this program. If not, see
<http://www.gnu.org/licenses/>.

-------------------------------------------------------------------------
IF THIS CODE IS USED FOR A RESEARCH PUBLICATION, please cite:
    ***will place reference to Arxiv paper here***
    
-------------------------------------------------------------------------
"""



# IMPORT LIBRARIES
import numpy as np
import functions as fs
import torch
import torch.nn.functional as F
import h5py



class SPPARKS:
    
    def __init__(self, size=[257,257], ngrain=512, nsteps=10, step_size=1, obs_dim=3, act_dim=25, future_window=4):
        
        # SPPARKS parameters
        self.size = size
        self.ngrain = ngrain
        self.nsteps = nsteps
        self.step_size = step_size
        self.rseed = np.random.randint(10000) #self.seed()%10000000
        
        # Label calculation parameters
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.future_window = future_window
        self.counter = self.nsteps
        
        # self.counter = 0 
        # self.h5step = 0


    def spk_init(self): 
        
        # Sets up initial condition for SPPARKS
        self.counter = self.nsteps
        self.img, self.EulerAngles, _ = fs.voronoi2image(self.size, self.ngrain)
        fs.image2init(self.img, self.EulerAngles)
        
        # print("empty")
    
    
    def spk_forward(self): 
        
        # Run SPPARKS
        s = [self.size[0]+0.5, self.size[1]+0.5]
        tot_steps = self.nsteps + self.step_size + self.future_window
        _ = fs.run_spparks(s, self.ngrain, tot_steps, self.step_size, self.step_size, self.rseed, del_files=False)
        
        # Get SPPARKS data
        euler_angle_images, sim_steps, EulerAngles, grain_ID_images, energy_images = fs.extract_spparks_dump(len(self.size)) 
        self.spk_ID = torch.from_numpy(grain_ID_images)
        
        # # Load presimulated data
        # fp = './data_training/spparks_data_size257x257_ngrain256-256_nsets200_future4_max100_offset1_kt05'
        # with h5py.File(fp, 'r') as f:
        #     self.spk_ID = torch.from_numpy(f["dataset"][self.h5step]).float()
        # self.h5step += 1
        
        # Compute energy changes for firstfuture_windows steps
        self.energy_change = []
        for i in range(self.counter, self.counter+self.future_window):
            self.energy_change.append(self.compute_energy_change(i))
        
        # Compute labels and features
        self.compute_labels()
        self.compute_features()
    
    
    def step(self):
        
        # Shift energy change window
        self.energy_change = self.energy_change[1:]
        self.energy_change.append(self.compute_energy_change(self.counter+self.future_window))
        
        # Increase counter
        self.counter += 1
        
        # Compute labels and features
        self.compute_labels()
        self.compute_features()
        
        
    def compute_energy_change(self, step, energy_dim=3):
        
        # Setup unfold parameters
        pad_mode = "circular"
        pad_energy = tuple([int(np.floor(energy_dim/2))]*4)
        pad_act = tuple([int(np.floor(self.act_dim/2))]*4)
        unfold_next = torch.nn.Unfold(kernel_size=energy_dim)
        unfold_act = torch.nn.Unfold(kernel_size=self.act_dim)
        
        # Select images needed for energy calculation
        ims_curr = self.spk_ID[step:step+1,]
        ims_next = self.spk_ID[step+1:step+2,]
        
        # Calculate current energy
        windows_curr = unfold_next(F.pad(ims_curr, pad_energy, pad_mode))
        current_energy = fs.num_diff_neighbors_inline(windows_curr)
        
        # Calculate energyy after taking eac possible action at each location
        windows_curr_act = unfold_act(F.pad(ims_curr, pad_act, pad_mode))
        windows_next = unfold_next(F.pad(ims_next, pad_energy, pad_mode))
        windows_next_ex = windows_next.unsqueeze(dim=1).repeat(1, self.act_dim**2, 1, 1) #copy the matrix for the number of actions needed
        windows_next_ex[:,:,int(energy_dim**2/2),:] = windows_curr_act #place one action into each matrix copy
        windows_next_ex = windows_next_ex.unsqueeze(4).transpose(0,4)[0] #reshape for the below function
        action_energy = fs.num_diff_neighbors_inline(windows_next_ex) #find the energy for each copy, each having a different action
        
        # Calculate energy labels - the energy change, discounted for each future step
        energy_change = (current_energy.transpose(0,1)-action_energy)/(energy_dim**2-1)
        return energy_change
    
    
    def compute_energy_labels(self):
        
        # Compute labels for the current counter stepCalculate all needed energy changes
        energy_change = torch.cat(self.energy_change, dim=2)
        energy_labels = energy_change[:,:,0]*(1/2)
        for i in range(1, self.future_window): energy_labels += energy_change[:,:,i]*(1/2)**(i+1) #dsicount future energy changes
        energy_labels = energy_labels.transpose(0,1).reshape(np.product(self.size), self.act_dim, self.act_dim) #the final energy label
        return energy_labels


    def compute_action_labels(self):
        # Compute labels for the current counter step
        
        # Setup unfold parameters
        pad_mode = "circular"
        pad_act = tuple([int(np.floor(self.act_dim/2))]*4)
        unfold_act = torch.nn.Unfold(kernel_size=self.act_dim)
        
        # Select images needed for energy calculation
        ims_curr = self.spk_ID[self.counter:self.counter+1,]
        ims_next = self.spk_ID[self.counter+1:self.counter+self.future_window+1,]
        
        window_curr_act = unfold_act(F.pad(ims_curr, pad_act, pad_mode))[0]
        ims_next_flat = ims_next.view(ims_next.shape[0], -1)
        action_labels = (ims_next_flat[0]==window_curr_act)*(1/2)
        for i in range(1,ims_next_flat.shape[0]): action_labels += (ims_next_flat[i]==window_curr_act)*(1/2)**(i+1) #discount future action labels
        action_labels = action_labels.transpose(0,1).reshape(np.product(self.size), self.act_dim, self.act_dim) #the final action label
        return action_labels
    
    
    def compute_labels(self): 
        # Combine for final labels
        self.labels = (self.compute_energy_labels() + self.compute_action_labels()).cpu().numpy()
        
        
    def compute_features(self):
        ims_curr = self.spk_ID[self.counter:self.counter+1,]
        local_energy = fs.num_diff_neighbors(ims_curr, window_size=7, pad_mode='circular')
        self.features = fs.my_unfoldNd(local_energy.float(), self.obs_dim).T.reshape(np.product(self.size),self.obs_dim,self.obs_dim,1).numpy()