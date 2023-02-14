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


# Setup gpu access
import tensorflow as tf
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.set_visible_devices(physical_devices[0], 'GPU') 
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


class PRIMME:
    def __init__(self, obs_dim=9, act_dim=9, pad_mode="circular", learning_rate=0.00005, reg=1, num_dims=2, cfg='./cfg/dqn_setup.json'):
        self.device = device
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.pad_mode = pad_mode
        self.learning_rate = learning_rate
        self.reg = reg
        self.num_dims = num_dims
        self.model = self._build_model()
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
    
    
    def sample_data(self, h5_path='spparks_data_size257x257_ngrain256-256_nsets200_future4_max100_offset1_kt0.h5', batch_size=1):
        with h5py.File(h5_path, 'r') as f:
            i_max = f['ims_id'].shape[0]
            i_batch = np.sort(np.random.randint(low=0, high=i_max, size=(batch_size,)))
            batch = f['ims_id'][i_batch,]
            miso_array = f['miso_array'][i_batch,] 
        self.im_seq = torch.from_numpy(batch[0,].astype(float)).to(self.device)
        miso_array = torch.from_numpy(miso_array.astype(float)).to(self.device)
        self.miso_matrix = fs.miso_conversion(miso_array)
        
        #Compute features and labels
        self.features = fs.compute_features(self.im_seq[0:1,], obs_dim=self.obs_dim, pad_mode=self.pad_mode)
        self.labels = fs.compute_labels(self.im_seq, obs_dim=self.obs_dim, act_dim=self.act_dim, reg=self.reg, pad_mode=self.pad_mode)
        
        
    def step(self, im, miso_matrix, evaluate=True):
        features = fs.compute_features(im, obs_dim=self.obs_dim, pad_mode=self.pad_mode)
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
            
            action_values = torch.argmax(predictions, dim=1)
            
            if evaluate==True: 
                predictions_split.append(predictions)
            action_values_split.append(action_values)
                
        if evaluate==True: self.predictions = torch.cat(predictions_split, dim=0)
        action_values = torch.hstack(action_values_split)
        
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
        return loss, accuracy
        
    
    def train(self, evaluate=True):
        
        if evaluate: 
            loss, accuracy = self.compute_metrics()
            self.validation_loss.append(loss)
            self.validation_acc.append(accuracy)
        
        features, labels = fs.unison_shuffled_copies(self.features, self.labels) #random shuffle 
        
        mid_ix = (np.array(features.shape[1:])/2).astype(int)
        ind = tuple([slice(None)]) + tuple(mid_ix)
        indx_use = torch.nonzero(features[ind])[:,0]
        
        features = features[indx_use,].cpu().numpy()
        labels = labels[indx_use,].cpu().numpy()
        _ = self.model.fit(features, np.reshape(labels,(-1,self.act_dim**self.num_dims)), epochs=1, verbose=0)
        
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