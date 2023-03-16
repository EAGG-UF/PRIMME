#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IF THIS CODE IS USED FOR A RESEARCH PUBLICATION, please cite:
    Yan, W., Melville, J., Yadav, V., Everett, K., Yang, L., Kesler, M. S., ... & Harley, J. B. (2022). A novel physics-regularized interpretable machine learning model for grain growth. Materials & Design, 222, 111032.
"""



# IMPORT LIBRARIES
import numpy as np
from keras.models import Model, load_model
from keras.layers import Dense, Input, Flatten, BatchNormalization, Dropout
import keras.backend as K
from tensorflow.keras.optimizers import Adam
import functions as fs
import torch
import h5py
import matplotlib.pyplot as plt
# import os
# from torch import nn
# import torch.nn.functional as F



# Setup gpu access
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices[1], 'GPU') #0:5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



# Pytorch replacement for keras
# class PRIMME_DNN(nn.Module):
#     def __init__(self, input_layer=3*3, output_layer=25*25):
#         super(PRIMME_DNN, self).__init__()
        
#         self.layers = [input_layer, 21*21*4, 21*21*2, 21*21, output_layer]
        
#         self.fc1 = nn.Linear(self.layers[0], self.layers[1])
#         self.fc2 = nn.Linear(self.layers[1], self.layers[2])
#         self.fc3 = nn.Linear(self.layers[2], self.layers[3])
#         self.fc4 = nn.Linear(self.layers[3], self.layers[4])
        
#         self.d1 = nn.Dropout(p=0.25)
#         self.d2 = nn.Dropout(p=0.25)
        
#         self.n1 = nn.BatchNorm1d(self.layers[0], affine=False)
#         self.n2 = nn.BatchNorm1d(self.layers[1], affine=False)
#         self.n3 = nn.BatchNorm1d(self.layers[2], affine=False)
#         self.n4 = nn.BatchNorm1d(self.layers[3], affine=False)
        
#     def forward(self, samples):
#         x = samples.reshape(samples.shape[0],self.layers[0])
#         x = self.d1(F.relu(self.fc1(self.n1(x))))
#         x = self.d2(F.relu(self.fc2(self.n2(x))))
#         x = F.relu(self.fc3(self.n3(x)))
#         predictions = torch.sigmoid(self.fc4(self.n4(x)))
#         return predictions
    
#     def setup_training(self, learning_rate):
#         self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
#         self.loss_func = torch.nn.MSELoss()
#         # self.eval()
#         # self.train()

#     def fit(self, samples, labels):
#         labels = labels.reshape(labels.shape[0],self.layers[-1])
#         self.train()
#         # self.optimizer.zero_grad()
#         predictions = self.forward(samples)
#         loss = self.loss_func(predictions, labels) 
#         # self.optimizer.zero_grad()
#         loss.backward()        
#         self.optimizer.step()  
#         self.optimizer.zero_grad()
#         # self.eval()
#         return predictions, loss
    
#     def evaluate(self, samples, labels):
#         self.eval()
#         labels = labels.reshape(labels.shape[0],self.layers[-1])
#         with torch.no_grad():
#             predictions = self.forward(samples)
#             loss = self.loss_func(predictions, labels)
#         return predictions, loss





class PRIMME:
    def __init__(self, obs_dim=9, feat_dim=7, act_dim=9, pad_mode="circular", learning_rate=0.00005, batch_size=1, reg=1, num_dims=2, cfg='./cfg/dqn_setup.json'):
        self.device = device
        self.obs_dim = obs_dim
        self.feat_dim = feat_dim
        self.act_dim = act_dim
        self.pad_mode = pad_mode
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.reg = reg
        self.num_dims = num_dims
        self.model = self._build_model()#.to(self.device)
        self.training_loss = []
        self.validation_loss = []
        self.training_acc = []
        self.validation_acc = []

    
    def _build_model(self):
        state_input = Input(shape=(self.obs_dim,)*self.num_dims)
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
        output = Dense(self.act_dim**self.num_dims,  activation='sigmoid')(h9)
        model = Model(inputs=state_input, outputs=output)
        adam = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=adam, loss='mse')
        return model
    
    
    # def _build_model(self): #pytorch
    #     model = PRIMME_DNN(input_layer=self.obs_dim**self.num_dims, output_layer=self.act_dim**self.num_dims)
    #     model.setup_training(self.learning_rate)
    #     return model
    
    
    def sample_data(self, h5_path='spparks_data_size257x257_ngrain256-256_nsets200_future4_max100_offset1_kt0.h5', batch_size=1):
        with h5py.File(h5_path, 'r') as f:
            
            # Initial variables
            s = np.array(f['ims_id'].shape)
            d = len(s)-3
            
            # Retrieve one image and its misorientation array
            i_batch = np.random.randint(low=0, high=s[0])
            self.im_seq = torch.from_numpy(f['ims_id'][i_batch,])
            miso_array = torch.from_numpy(f['miso_array'][i_batch,][None,])
            
        # Sample features and labels
        rnd_shift = [1,1,]+list(s[3:])
        rng_size = list(self.im_seq.shape[:2])+[self.act_dim,]*d
        _, shifts, i_lab = fs.rnd_batch(self.batch_size, rng_size, rnd_shift)
        rng_size = list(self.im_seq.shape[:2])+[self.act_dim+self.feat_dim-1,]*d
        _, shifts, i_feat = fs.rnd_batch(self.batch_size, rng_size, rnd_shift, shifts=shifts)
        
        ims_lab = self.im_seq[i_lab]
        ims_feat = self.im_seq[i_feat]
        self.miso_matrix = fs.miso_array_to_matrix(miso_array)
        self.labels = fs.compute_labels_batch(ims_lab, obs_dim=self.obs_dim, act_dim=self.act_dim, reg=self.reg, pad_mode=self.pad_mode)
        self.features = fs.compute_features_batch(ims_feat[:,0,], obs_dim=self.obs_dim, window_size=self.feat_dim, pad_mode=self.pad_mode)
        # self.features = fs.compute_features_miso_batch(ims_feat[:,0,].transpose(0,1), self.miso_matrix, obs_dim=self.obs_dim, window_size=self.feat_dim, pad_mode=self.pad_mode)
        
        
    def step(self, im, miso_matrix, evaluate=True):
        
        
        ims = im
        batch_size = 3
        rng_size = [1,1,23,23]
        
        
        
        
        s = ims.shape
        d = len(s)
        c_shift = [0,0,]+list((-np.array(rng_size[2:])/2).astype(int)) #constant shift
        rngs = [torch.arange(c_shift[i], c_shift[i]+rng_size[i]) for i in range(d)]
        meshes = torch.meshgrid(rngs)
        
        
        
        
        rnd_shift = [1,1,]+list(s[2:])
        shifts = fs.rand_int_multi(rnd_shift, batch_size, add_dims=d)
        
        
        rng_shift = list(s)#[1,1,]+list(s[2:])
        
        i_gen = range(int(np.prod(rng_shift)))
        
        for i in range(0,100+batch_size,batch_size): 
            torch.arange()
        
        
        high = rnd_shift
        add_dims=d
        
        d = len(high)
        size = (batch_size,)+(1,)*(add_dims)
        low=torch.Tensor([0,]*d).long()
        indices = [torch.randint(low=low[i], high=high[i], size=size) for i in range(d)]
        
        i_gen = range(int(np.prod(s[2:])))
        
        i_tot.split
        
        
        from tqdm import tqdm
        for i in tqdm(range(10)):
            print(i)
        
        
        def grid_sample():
            
        
        
        
        
        
        
        indices = [(meshes[i][None]+shifts[i])%s[i] for i in range(d)]  
        
        
        
        
        # features = fs.compute_features(im, obs_dim=self.obs_dim, pad_mode=self.pad_mode)
        features = fs.compute_features_miso(im, miso_matrix, obs_dim=self.obs_dim, pad_mode=self.pad_mode) #use miso functions
        mid_ix = (np.array(features.shape[1:])/2).astype(int)
        ind = tuple([slice(None)]) + tuple(mid_ix)
        indx_use = torch.nonzero(features[ind])[:,0]
        features = features[indx_use,]
        
        action_features = fs.my_unfoldNd(im, kernel_size=self.act_dim, pad_mode=self.pad_mode)[0,] 
        action_features = action_features[...,indx_use]
        
        batch_size = 5000
        features_split = torch.split(features, batch_size)
        predictions_split = []
        action_values_split = []
        
        for e in features_split:
            
            predictions = torch.Tensor(self.model.predict_on_batch(e.cpu().numpy())).to(self.device)
            # predictions = self.model(e)
            
            action_values = torch.argmax(predictions, dim=1)
            
            if evaluate==True: 
                predictions_split.append(predictions)
            action_values_split.append(action_values)
                
        if evaluate==True: self.predictions = torch.cat(predictions_split, dim=0)
        action_values = torch.hstack(action_values_split)
        
        # self.im_next = torch.gather(action_features, dim=0, index=action_values.unsqueeze(0)).reshape(im.shape)
        upated_values = torch.gather(action_features, dim=0, index=action_values.unsqueeze(0))[0,]
        self.im_next = im.flatten().float()
        self.im_next[indx_use] = upated_values
        self.im_next = self.im_next.reshape(im.shape)
        self.indx_use = indx_use
        
        return self.im_next


    def compute_metrics(self):
        im_next_predicted = self.step(self.im_seq[0:1,], self.miso_matrix)
        im_next_actual = self.im_seq[1:2,]
        accuracy = torch.mean((im_next_predicted==im_next_actual).float())
        loss = np.mean(tf.keras.losses.mse(self.predictions.cpu().numpy(), np.reshape(self.labels[self.indx_use,].cpu(),(-1,self.act_dim**self.num_dims))))
        # _, loss = self.model.evaluate(self.predictions, self.labels[self.indx_use,].reshape(-1,self.act_dim**self.num_dims))
        return loss, accuracy
        
    
    def train(self, evaluate=True):
        
        if evaluate: 
            loss, accuracy = self.compute_metrics()
            self.validation_loss.append(loss)
            self.validation_acc.append(accuracy)
        
        # features, labels = fs.unison_shuffled_copies(self.features.cpu().numpy(), self.labels.cpu().numpy()) #random shuffle 
        features, labels = fs.unison_shuffled_copies(self.features, self.labels) #random shuffle 
        # ss = int(self.obs_dim/2)
        # indx_use = np.nonzero(features[:,ss,ss,ss])[0]
        
        mid_ix = (np.array(features.shape[1:])/2).astype(int)
        ind = tuple([slice(None)]) + tuple(mid_ix)
        indx_use = torch.nonzero(features[ind])[:,0]
        
        features = features[indx_use,].cpu().numpy()
        labels = labels[indx_use,].cpu().numpy()
        _ = self.model.fit(features, np.reshape(labels,(-1,self.act_dim**self.num_dims)), epochs=1, verbose=0)
        # self.training_loss.append(history.history['loss'][0])
        # features, labels = fs.unison_shuffled_copies(self.features, self.labels) #random shuffle 
        # ss = int(self.obs_dim/2)
        # indx_use = torch.nonzero(features[:,ss,ss,ss])[:,0]
        # features = features[indx_use,]
        # labels = labels[indx_use,]
        # _ = self.model.fit(features, labels.reshape(-1,self.act_dim**self.num_dims))
        
        if evaluate: 
            loss, accuracy = self.compute_metrics()
            self.training_loss.append(loss)
            self.training_acc.append(accuracy)
        
    
    def plot(self, fp_results='./plots'):
        
        if self.num_dims==2:
            #Plot the next images, predicted and true, together
            fig, axs = plt.subplots(1,3)
            axs[0].matshow(self.im_seq[0,0,].cpu().numpy())
            axs[0].set_title('Current')
            axs[0].axis('off')
            axs[1].matshow(self.im_next[0,0,].cpu().numpy()) 
            axs[1].set_title('Predicted Next')
            axs[1].axis('off')
            axs[2].matshow(self.im_seq[1,0,].cpu().numpy()) 
            axs[2].set_title('True Next')
            axs[2].axis('off')
            plt.savefig('%s/sim_vs_true.png'%fp_results)
            plt.show()
            
            #Plot the action distributions, predicted and true, together
            ctr = int((self.act_dim-1)/2)
            pred = self.predictions.reshape(-1, self.act_dim, self.act_dim).detach().cpu().numpy()
            fig, axs = plt.subplots(1,2)
            p1 = axs[0].matshow(np.mean(pred, axis=0), vmin=0, vmax=1)
            fig.colorbar(p1, ax=axs[0])
            axs[0].plot(ctr,ctr,marker='x')
            axs[0].set_title('Predicted')
            axs[0].axis('off')
            p2 = axs[1].matshow(np.mean(self.labels.cpu().numpy(), axis=0), vmin=0, vmax=1) 
            
            # p2 = axs[1].matshow(np.mean(self.labels.cpu().numpy(), axis=0)) 
            # p2 = axs[1].matshow(self.labels.cpu().numpy()[0]) 
            
            fig.colorbar(p2, ax=axs[1])
            axs[1].plot(ctr,ctr,marker='x')
            axs[1].set_title('True')
            axs[1].axis('off')
            plt.savefig('%s/action_likelihood.png'%fp_results)
            plt.show()
            
            #Plot loss and accuracy
            fig, axs = plt.subplots(1,2)
            axs[0].plot(self.validation_loss, '-*', label='Validation')
            axs[0].plot(self.training_loss, '--*', label='Training')
            axs[0].set_title('Loss')
            axs[0].legend()
            axs[1].plot(self.validation_acc, '-*', label='Validation')
            axs[1].plot(self.training_acc, '--*', label='Training')
            axs[1].set_title('Accuracy')
            axs[1].legend()
            plt.savefig('%s/train_val_loss_accuracy.png'%fp_results)
            plt.show()
            
            plt.close('all')
        
        if self.num_dims==3:
            bi = int(self.im_seq.shape[-1]/2)
            
            #Plot the next images, predicted and true, together
            fig, axs = plt.subplots(1,3)
            axs[0].matshow(self.im_seq[0,0,...,bi].cpu().numpy())
            axs[0].set_title('Current')
            axs[0].axis('off')
            axs[1].matshow(self.im_next[0,0,...,bi].cpu().numpy()) 
            axs[1].set_title('Predicted Next')
            axs[1].axis('off')
            axs[2].matshow(self.im_seq[1,0,...,bi].cpu().numpy()) 
            axs[2].set_title('True Next')
            axs[2].axis('off')
            plt.savefig('%s/sim_vs_true.png'%fp_results)
            plt.show()
            
            #Plot the action distributions, predicted and true, together
            ctr = int((self.act_dim-1)/2)
            pred = self.predictions.reshape(-1, self.act_dim, self.act_dim, self.act_dim).detach().cpu().numpy()
            fig, axs = plt.subplots(1,2)
            p1 = axs[0].matshow(np.mean(pred, axis=0)[...,ctr], vmin=0, vmax=1)
            fig.colorbar(p1, ax=axs[0])
            axs[0].plot(ctr,ctr,marker='x')
            axs[0].set_title('Predicted')
            axs[0].axis('off')
            p2 = axs[1].matshow(np.mean(self.labels.cpu().numpy(), axis=0)[...,ctr], vmin=0, vmax=1) 
            fig.colorbar(p2, ax=axs[1])
            axs[1].plot(ctr,ctr,marker='x')
            axs[1].set_title('True')
            axs[1].axis('off')
            plt.savefig('%s/action_likelihood.png'%fp_results)
            plt.show()
            
            #Plot loss and accuracy
            fig, axs = plt.subplots(1,2)
            axs[0].plot(self.validation_loss, '-*', label='Validation')
            axs[0].plot(self.training_loss, '--*', label='Training')
            axs[0].set_title('Loss')
            axs[0].legend()
            axs[1].plot(self.validation_acc, '-*', label='Validation')
            axs[1].plot(self.training_acc, '--*', label='Training')
            axs[1].set_title('Accuracy')
            axs[1].legend()
            plt.savefig('%s/train_val_loss_accuracy.png'%fp_results)
            plt.show()
            
            plt.close('all')
        
    
    def load(self, name):
        self.model = load_model(name)
        self.num_dims = len(self.model.layers[0].get_output_at(0).get_shape().as_list()) - 1
        self.obs_dim = self.model.layers[0].get_output_at(0).get_shape().as_list()[1]
        model_out_dim = self.model.layers[-1].get_output_at(0).get_shape().as_list()[1]
        self.act_dim = int(np.rint(model_out_dim**(1/self.num_dims)))
        self.learning_rate = K.eval(self.model.optimizer.lr)


    def save(self, name):
        self.model.save(name)