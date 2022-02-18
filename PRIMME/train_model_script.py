#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DESCRIPTION:
    This script is used to train the PRIMME model using data from SPPARKS, generated in real time.
    A SPPARKS environment must be set up for this script to run properly. Please see ./spparks_files/Getting_Started
    The model is saved to "./saved models/" after each training epoch
    Training evaluation files are saved to "./results_training" after training is complete


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
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from PRIMME import PRIMME
from SPPARKS import SPPARKS
import torch



# SETUP VARIABLES
EPISODES = 200   # Number of episodes to complete
max_SPKSTEPS = 100 # Max SPPARKS simulation steps                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
max_NSTEPS = 1 # Number of times steps between SPPARKS images
action_window_dim = 17 # Size of observation field
observ_window_dim = 17 # Size of action field
future_window = 4 #Number of future SPPARKS windows to consider for training
max_grain = 256 # Maximum number of grains in SPPARKS image
min_grain = 256 # Minimum number of grains in SPPARKS image
size      = 257 # Size of each dimension in SPPARKS
modelname = "./saved_models/primme_grains%s_size%s_episodes%s_maxsteps%s_obs%s_act%s_kt0.5_dummy" % (str(max_grain),str(size),str(EPISODES), str(max_SPKSTEPS), str(observ_window_dim), str(action_window_dim))
fp_results = './results_training'
  


# RUN TRAINING EPOCHS
env = SPPARKS(size=[size, size], obs_dim=observ_window_dim, act_dim=action_window_dim, future_window=future_window) 
agent = PRIMME(env)
grain_area_list = []

for _ in tqdm(range(EPISODES), desc='Episodes', leave=True):
    
    
    # Simulate training data with SPPARKS
    env.ngrain = np.random.randint(min_grain, max_grain+1)
    env.nsteps = np.random.randint(max_SPKSTEPS+1)
    env.step_size = np.random.randint(max_NSTEPS)+1
    env.spk_init()
    env.spk_forward()
    
    
    # Predict next step before training (compute validation error)
    agent.init_state()
    agent.predict()
    agent.compute_accuracy()
    agent.validation_acc.append(agent.acc)
    
    print()
    print('####')
    print('PRE-TRAINING ACCURACY')
    print(env.nsteps)
    print(agent.acc)
    print('####')
    
    
    # Start plot for SPPARKS/PRIMME comparison
    env_next_ID = env.spk_ID[env.counter+1, 0].unsqueeze(-1)
    fig, axs = plt.subplots(2,3)
    axs[0,0].matshow(agent.ID.cpu())
    axs[0,0].set_title('Current')
    axs[0,0].axis('off')
    axs[0,1].matshow(agent.next_ID.cpu()) 
    axs[0,1].set_title('Predicted Next')
    axs[0,1].axis('off')
    axs[0,2].matshow(env_next_ID) 
    axs[0,2].set_title('True Next')
    axs[0,2].axis('off')
    
    
    # Train, then predict next step after training (compute training error)
    agent.train()
    agent.predict()
    agent.compute_accuracy()
    agent.training_acc.append(agent.acc)

    print()
    print('####')
    print('POST-TRAINING ACCURACY')
    print(agent.acc)
    print('####')
    
    
    # Finish plot for SPPARKS/PRIMME comparison
    axs[1,0].matshow(agent.ID.cpu()); axs[1,0].axis('off')
    axs[1,1].matshow(agent.next_ID.cpu()); axs[1,1].axis('off')
    axs[1,2].matshow(env_next_ID); axs[1,2].axis('off')
    plt.show()
    
    
    # Plot training and validation accuracy
    plt.plot(agent.training_acc, '-*')
    plt.plot(agent.validation_acc, '-*')
    plt.legend(['Training Accuracy', 'Validation Accuracy'])
    plt.show()
    
    
    # Plot the mean of the current action likelihood
    action_likelihood = agent.action_likelihood.cpu().numpy().reshape(-1,action_window_dim,action_window_dim)
    ctr = int((action_window_dim-1)/2)
    plt.matshow(np.mean(action_likelihood,axis=0)); plt.colorbar()
    plt.plot(ctr,ctr,marker='x'); 
    plt.show()
    
    
    # Save the model
    agent.save("%s" % modelname)
    
    
    # Save the grain area used for this training step
    _, areas = torch.unique(env.spk_ID[env.counter], return_counts=True)
    grain_area_list.append(areas.numpy())



# CREATE AND SAVE FINAL PLOTS and data

# Plot of SPPARKS/PRIMME simulation comparison
env_next_ID = env.spk_ID[env.counter+1, 0].unsqueeze(-1)
fig, axs = plt.subplots(1,3)
axs[0].matshow(agent.ID.cpu())
axs[0].set_title('Current')
axs[0].axis('off')
axs[1].matshow(agent.next_ID.cpu()) 
axs[1].set_title('Predicted Next')
axs[1].axis('off')
axs[2].matshow(env_next_ID) 
axs[2].set_title('True Next')
axs[2].axis('off')
plt.savefig('%s/sim_vs_true.png'%fp_results)
plt.show()
    
# Plot of mean of action likelihood
action_likelihood = agent.action_likelihood.cpu().numpy().reshape(-1,action_window_dim,action_window_dim)
ctr = int((action_window_dim-1)/2)
plt.matshow(np.mean(action_likelihood,axis=0)); plt.colorbar()
plt.plot(ctr,ctr,marker='x'); 
plt.savefig('%s/action_likelihood.png'%fp_results)
plt.show()

# Plot of training and validation accuracy
plt.plot(agent.training_acc, '-*')
plt.plot(agent.validation_acc, '-*')
plt.legend(['Training Accuracy', 'Validation Accuracy'])
plt.savefig('%s/train_val_accuracy.png'%fp_results)
plt.show()

# File with the area of all the grains used to train the current model
grain_areas = np.hstack(grain_area_list)
np.savetxt('%s/grain_areas.csv'%fp_results, grain_areas)