#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DESCRIPTION:
    This script defines the PRIMME simulator class used to simulate microstructural grain growth
    The neural network model used to predict the action likelihood is written in Tensorflow (Keras)
    The functions besides of the model are written in Pytorch to parallelize the operations using GPUs if available
    This class must be passed a SPPARKS class ('env'), which provides an initial condition, and training data, features, and labels 
    The main functions of the class include predicting the action likelihood (given an intial condition) and training the model (given features and labels)

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
from keras.models import Model, load_model
from keras.layers import Dense, Input, Flatten, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
import functions as fs
import torch



# Setup tesnorflow gpu access
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices[0], 'GPU') #0:5



class PRIMME:
    def __init__(self, env, cfg='./cfg/dqn_setup.json'):
        self.env = env
        self.size = self.env.size
        self.pad_mode = "circular"
        self.learning_rate = 0.00005 
        self.model = self._build_model()
        self.training_losses = []
        self.training_acc = []
        self.validation_acc = []
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    
    def _build_model(self):
        state_input = Input(shape=(self.env.obs_dim, self.env.obs_dim, 1))
        h0 = state_input
        h1 = Flatten()(h0)
        h2 = BatchNormalization()(h1)
        h3 = Dense(21*21*4, activation='relu')(h2)
        h4 = Dropout(0.25)(h3)
        h5 = BatchNormalization()(h4)
        h6 = Dense(21*21*2, activation='relu')(h5)
        h7 = Dropout(0.25)(h6)
        h9 = BatchNormalization()(h7)
        h8 = Dense(21*21, activation='relu')(h7)
        h9 = BatchNormalization()(h8)
        output = Dense(self.env.act_dim**2,  activation='sigmoid')(h9)
        model = Model(inputs=state_input, outputs=output)
        adam = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=adam, loss='mse')
        return model
    
    
    def init_state(self):
        self.ID = torch.Tensor(self.env.spk_ID[self.env.counter,0].float()).unsqueeze(2).to(self.device)
      

    def compute_action(self):
        n = fs.num_diff_neighbors(self.ID[...,0].reshape([1,1]+self.size), window_size=7)
        self.features = fs.my_unfoldNd(n.float(), kernel_size=self.env.obs_dim, pad_mode=self.pad_mode)[0,].transpose(0,1).reshape([np.product(self.size), self.env.obs_dim, self.env.obs_dim, 1])
        
        # Batch the model inputs to limit memory usage 
        mem_lim = 1e9
        nsplits = np.ceil(np.product(self.features.shape)*64/mem_lim)
        batch_size = int(np.ceil((self.features.shape[0])/nsplits))
        nbatches = int(self.features.shape[0]/batch_size)
            
        feature_batch = self.features[:batch_size].cpu().numpy()
        action_likelihood = torch.from_numpy(self.model(feature_batch).numpy()).to(self.device)
        # action_values = torch.argmax(action_likelihood, dim=1)
        action_values = fs.rand_argmax(action_likelihood)
        
        
        for i in range(1, nbatches+1):
            start = batch_size*i
            end = batch_size*(i+1)
            if batch_size*(i+1)>self.features.shape[0]: end = self.features.shape[0]
            feature_batch = self.features[start:end].cpu().numpy()
            action_likelihood = torch.from_numpy(self.model(feature_batch).numpy()).to(self.device)
            # action_values = torch.cat((action_values, torch.argmax(action_likelihood, dim=1)))
            action_values = torch.cat((action_values, fs.rand_argmax(action_likelihood)))
        self.action_likelihood = action_likelihood
        self.action_values = action_values.to(self.device)
        

    def apply_action(self):
        afeatures = fs.my_unfoldNd(self.ID[...,0].reshape([1,1]+self.size), kernel_size=self.env.act_dim, pad_mode=self.pad_mode)[0,]
        self.next_ID = torch.gather(afeatures, dim=0, index=self.action_values.unsqueeze(0)).reshape(self.size[0], self.size[1], 1)
    

    def predict(self):
        self.compute_action()
        self.apply_action()


    def step(self):
        self.ID    = self.next_ID
        

    def compute_accuracy(self):
        env_next_ID = self.env.spk_ID[self.env.counter+1,0,].to(self.device)
        self.acc = torch.mean((self.next_ID[...,0]==env_next_ID).float()).cpu().numpy()
    

    def train(self):
        features, labels = fs.unison_shuffled_copies(self.env.features, self.env.labels) #random shuffle 
        history = self.model.fit(features, np.reshape(labels,(-1,self.env.act_dim**2)), epochs=1, verbose=1)
        self.training_losses.append(history.history['loss'][0])
    
    
    def load(self, name):
        self.model = load_model(name)


    def save(self, name):
        self.model.save(name)