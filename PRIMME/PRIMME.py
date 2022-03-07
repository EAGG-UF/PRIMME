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
        n = fs.num_diff_neighbors(self.ID[...,0].reshape([1,1]+self.size), window_size=7, pad_mode=self.pad_mode)
        self.features = fs.my_unfoldNd(n.float(), kernel_size=self.env.obs_dim, pad_mode=self.pad_mode)[0,].transpose(0,1).reshape([np.product(self.size), self.env.obs_dim, self.env.obs_dim, 1])
        
        mem_lim = 1e9
        mem_sample = np.product(self.features.shape[1:])*64
        batch_size = int(mem_lim/mem_sample)
        splits = torch.split(self.features, batch_size)
        
        action_likelihood_sum = torch.zeros([self.env.act_dim**2]).to(self.device)
        action_values = []
        for s in splits: 
            action_likelihood = torch.from_numpy(self.model.predict_on_batch(s.cpu().numpy())).to(self.device)
            action_values.append(torch.argmax(action_likelihood, dim=1))
            action_likelihood_sum += torch.sum(action_likelihood, dim=0)
        self.action_likelihood_mean = (action_likelihood_sum/self.features.shape[0]).reshape(self.env.act_dim, self.env.act_dim)
        self.action_values = torch.hstack(action_values).to(self.device)
        

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
        _, self.env.obs_dim, _, _ = self.model.layers[0].get_output_at(0).get_shape().as_list()
        _, act_dim_sqr = self.model.layers[-1].get_output_at(0).get_shape().as_list()
        self.env.act_dim = int(np.sqrt(act_dim_sqr))


    def save(self, name):
        self.model.save(name)