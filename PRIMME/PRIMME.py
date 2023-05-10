#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IF THIS CODE IS USED FOR A RESEARCH PUBLICATION, please cite:
    Yan, W., Melville, J., Yadav, V., Everett, K., Yang, L., Kesler, M. S., ... & Harley, J. B. (2022). A novel physics-regularized interpretable machine learning model for grain growth. Materials & Design, 222, 111032.
"""

# IMPORT LIBRARIES
import numpy as np
import functions as fs
import torch
import h5py
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm



# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


class PRIMME(nn.Module):
    def __init__(self, obs_dim=17, act_dim=17, energy_dim=3, pad_mode='circular', learning_rate=5e-5, reg=1, num_dims=2, if_miso=False):
        super(PRIMME, self).__init__()
        
        # self.device = device
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.energy_dim = energy_dim
        self.pad_mode = pad_mode
        self.learning_rate = learning_rate
        self.reg = reg
        self.num_dims = num_dims
        self.if_miso = if_miso
        self.training_loss = []
        self.validation_loss = []
        self.training_acc = []
        self.validation_acc = []
        
        # DEFINE NEURAL NETWORK
        self.f1 = nn.Linear(self.obs_dim ** self.num_dims, 21 * 21 * 4)
        self.f2 = nn.Linear(21 * 21 * 4, 21 * 21 * 2)
        self.f3 = nn.Linear(21 * 21 * 2, 21 * 21)
        self.f4 = nn.Linear(21 * 21, self.act_dim ** self.num_dims)
        self.dropout = nn.Dropout(p = 0.25) 
        self.BatchNorm1 = nn.BatchNorm1d(21 * 21 * 4)
        self.BatchNorm2 = nn.BatchNorm1d(21 * 21 * 2)
        self.BatchNorm3 = nn.BatchNorm1d(21 * 21)

        # DEFINE NEURAL NETWORK OPTIMIZATION
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)#, weight_decay=1e-5)
        self.loss_func = torch.nn.MSELoss()  # Mean squared error loss
        self.optimizer.zero_grad()  # Make all the gradients zero
    
    
    def forward(self, x):
        # def forward: Run input X through the neural network
        #   Inputs--
        #        x: microstructure features around a center pixel
        #   Outputs--
        #       y: "action likelihood" for the center pixel to flip to each the grain associated with each other pixel
        
        out = F.relu(self.f1(x))
        out = self.dropout(out)   
        out = self.BatchNorm1(out)
        out = F.relu(self.f2(out))
        out = self.dropout(out)
        out = self.BatchNorm2(out)
        out = F.relu(self.f3(out))
        out = self.dropout(out)
        out = self.BatchNorm3(out)
        y  = torch.relu(self.f4(out))
        
        return y
    
    
    def sample_data(self, h5_path='spparks_data_size257x257_ngrain256-256_nsets200_future4_max100_offset1_kt0.h5'):
        #Extracts training and validation image sequences ("im_seq") and misorientation matricies ("miso_matrix") from "h5_path"
        #One image sequence extracted at a time, at random, given and 80/20, train/validation split
        #Calculates the batch size ("batch_sz") and number of iterations needed to iterate through a generator with that batch size ("num_iter")
        
        with h5py.File(h5_path, 'r') as f:
            i_max = f['ims_id'].shape[0]
            i_split = int(i_max*0.8)
            
            i_train = np.sort(np.random.randint(low=0, high=i_split, size=(1,)))
            batch = f['ims_id'][i_train,]
            miso_array = f['miso_array'][i_train,] 
            
            i_val = np.sort(np.random.randint(low=i_split, high=i_max, size=(1,)))
            batch_val = f['ims_id'][i_val,]
            miso_array_val = f['miso_array'][i_val,] 
         
        #Convert image sequences to Tenors and copy to "device"
        self.im_seq = torch.from_numpy(batch[0,].astype(float)).to(device)
        self.im_seq_val = torch.from_numpy(batch_val[0,].astype(float)).to(device)
        
        #Calculate misorientation matricies if indicated
        if self.if_miso:
            miso_array = miso_array[:, miso_array[0,]!=0] #cut out zeros, each starts with different number of grains
            miso_array = torch.from_numpy(miso_array.astype(float)).to(device)
            self.miso_matrix = fs.miso_array_to_matrix(miso_array)
            
            miso_array_val = miso_array_val[:, miso_array_val[0,]!=0] #cut out zeros, each starts with different number of grains
            miso_array = torch.from_numpy(miso_array_val.astype(float)).to(device)
            self.miso_matrix_val = fs.miso_array_to_matrix(miso_array)
        else:
            self.miso_matrix = None
            self.miso_matrix_val = None
        
        # #Calculate batch size to maintain memory usage limit 
        # unfold_mem_lim = 48e9
        # num_future = self.im_seq.shape[0]-1 #number of future steps
        # self.batch_sz = int(unfold_mem_lim/((num_future)*self.act_dim**self.num_dims*self.energy_dim**self.num_dims*64)) #set to highest memory functions - "compute_energy_labels_gen"
        # self.num_iter = int(np.ceil(np.prod(self.im_seq.shape[1:])/self.batch_sz))
        
        # #Compute features 
        # self.features_gen = fs.compute_features_gen(self.im_seq[0:1,], self.batch_sz, self.obs_dim, self.pad_mode)
        # self.features_val_gen = fs.compute_features_gen(self.im_seq_val[0:1,], self.batch_sz, self.obs_dim, self.pad_mode)
        
        # # self.features_gen = fs.compute_features_miso_gen(self.im_seq[0:1,], self.batch_sz, self.miso_matrix, self.obs_dim, self.pad_mode)
        # # self.features_val_gen = fs.compute_features_miso_gen(self.im_seq_val[0:1,], self.batch_sz, self.miso_matrix, self.obs_dim, self.pad_mode)
        
        # # Compute labels
        # self.labels_gen = fs.compute_labels_gen(self.im_seq, self.batch_sz, self.act_dim, self.energy_dim, self.reg, self.pad_mode)
        # self.labels_val_gen = fs.compute_labels_gen(self.im_seq_val, self.batch_sz, self.act_dim, self.energy_dim, self.reg, self.pad_mode)
        
        
        # #Compute features
        # self.features = fs.compute_features(self.im_seq[0:1,], obs_dim=self.obs_dim, pad_mode=self.pad_mode)
        # self.features_val = fs.compute_features(self.im_seq_val[0:1,], obs_dim=self.obs_dim, pad_mode=self.pad_mode)
        
        # # self.features = fs.compute_features_miso(self.im_seq[0:1,], self.miso_matrix, obs_dim=self.obs_dim, pad_mode=self.pad_mode)
        # # self.features_val = fs.compute_features_miso(self.im_seq_val[0:1,], self.miso_matrix_val, obs_dim=self.obs_dim, pad_mode=self.pad_mode)
        
        # # Compute labels
        # self.labels = fs.compute_labels(self.im_seq, act_dim=self.act_dim, reg=self.reg, pad_mode=self.pad_mode)
        # self.labels_val = fs.compute_labels(self.im_seq_val, act_dim=self.act_dim, reg=self.reg, pad_mode=self.pad_mode)
        
        # # self.labels = fs.compute_labels_miso(self.im_seq, self.miso_matrix, obs_dim=self.obs_dim, act_dim=self.act_dim, reg=self.reg, pad_mode=self.pad_mode)
        # # self.labels_val = fs.compute_labels_miso(self.im_seq_val, self.miso_matrix_val, obs_dim=self.obs_dim, act_dim=self.act_dim, reg=self.reg, pad_mode=self.pad_mode)
        
    
    def step_old(self, im, miso_matrix, evaluate=True): #delete later
        
        # self.eval()
        
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
            
            #remove this later!!!!! added it to fix an error cause by a batch of 1, switch to self.eval() in the future
            jjj=0
            if e.shape[0]==1: 
                jjj=1
                e = e.repeat(2,1,1)
            
            
            with torch.no_grad():
                predictions = self.forward(e.reshape(-1, self.obs_dim**self.num_dims))
                
            #remove this later!!!!!
            if jjj:
                predictions = predictions[0:1,]
            
            
            action_values = torch.argmax(predictions, dim=1)
            
            if evaluate==True: 
                predictions_split.append(predictions)
            action_values_split.append(action_values)
                
        if evaluate==True: self.predictions = torch.cat(predictions_split, dim=0)
        action_values = torch.hstack(action_values_split)
        
        # self.im_next = torch.gather(action_features, dim=0, index=action_values.unsqueeze(0)).reshape(im.shape)
        updated_values = torch.gather(action_features, dim=0, index=action_values.unsqueeze(0))[0,]
        
        self.im_next = im.flatten().float() 
        self.im_next[indx_use] = updated_values.float()
        self.im_next = self.im_next.reshape(im.shape) 
        self.indx_use = indx_use
        
        return self.im_next
    

    def step(self, im_seq, miso_matrix=None, unfold_mem_lim=.1e9):
        #"im_seq" can be of shape=[1,1,dim0,dim1,dim2] or a sequence of shape=[num_future, 1, dim0, dim1, dim2]
        #Find the image after "im" given the trained model
        #Also calculates loss and accurate if given an image sequence
        #Calculates misorientation features of "miso_matrix" is given
        
        #Calculate batch size to maintain memory usage limit 
        num_future = im_seq.shape[0] #number of future steps
        batch_sz = int(unfold_mem_lim/((num_future)*self.act_dim**self.num_dims*self.energy_dim**self.num_dims*64)) #set to highest memory functions - "compute_energy_labels_gen"
        num_iter = int(np.ceil(np.prod(im_seq.shape[1:])/batch_sz))
        
        # Initialize variables and generators
        im = im_seq[0:1,]
        num_future = im_seq.shape[0]-1
        if num_future>0: 
            labels_gen = fs.compute_labels_gen(im_seq, batch_sz, self.act_dim, self.energy_dim, self.reg, self.pad_mode)
            im_next_true_split = im_seq[1:2,].flatten().split(batch_sz)
        
        im_unfold_gen = fs.unfold_in_batches(im[0,0], batch_sz, [self.obs_dim,]*self.num_dims, [1,]*self.num_dims, self.pad_mode)
        if miso_matrix is None:
            features_gen = fs.compute_features_gen(im, batch_sz, self.obs_dim, self.pad_mode)
        else: 
            features_gen = fs.compute_features_miso_gen(im, batch_sz, miso_matrix, self.obs_dim, self.pad_mode)
        
        # Find next image
        action_likelyhood = torch.zeros((self.act_dim,)*self.num_dims)
        action_likelyhood_true = torch.zeros((self.act_dim,)*self.num_dims)
        loss = 0
        accuracy = 0
        num_features = 0
        im_next_log = []
        for i in range(num_iter):
            
            print(i)
            
            # Only use neighborhoods that have more than one ID (have a potential to change ID)
            im_unfold = next(im_unfold_gen).reshape(-1, self.obs_dim**self.num_dims)
            mid_i = int(self.obs_dim**self.num_dims/2)
            current_ids = im_unfold[:,mid_i]
            use_i = (im_unfold != current_ids[:,None]).sum(1).nonzero()[:,0]
            
            print(len(use_i))
            
            if len(use_i)==1: 
                add_i = (use_i+1)%im_unfold.shape[0] #keep the next index whether or not it has all the same IDs
                use_i = torch.cat([use_i, add_i]) #at least two samples needed for batch normalization when training
            
            # Pass features through model
            features = next(features_gen).reshape(-1, self.obs_dim**self.num_dims)
            outputs = self.forward(features[use_i,])
            switch_i = outputs.argmax(1)
            action_likelyhood += outputs.sum(0).detach().cpu().reshape((self.act_dim,)*self.num_dims)
            
            # Find predicted IDs
            next_ids = current_ids.clone()
            next_ids[use_i] = im_unfold[use_i,][torch.arange(len(use_i)),switch_i]
            im_next_log.append(next_ids)
        
            # Calculate loss and accuracy
            if num_future>0: 
                labels = next(labels_gen).reshape(-1, self.act_dim**self.num_dims)
                action_likelyhood_true += labels[use_i,].sum(0).detach().cpu().reshape((self.act_dim,)*self.num_dims)
                loss += self.loss_func(outputs, labels[use_i,])*len(use_i) #convert MSE loss back to sum to find average of total
                next_ids_true = im_next_true_split[i]
                accuracy += torch.sum(next_ids[use_i] == next_ids_true[use_i]).float() #sum number of correct ID predictions
                num_features += len(use_i) #track total number of features for averaging later
                
        # Concatenate batches to form next image (as predicted)
        im_next = torch.cat(im_next_log).reshape(im.shape)
        
        # Find average of loss and accuracy
        if num_future>0: 
            action_likelyhood /= num_features
            action_likelyhood_true /= num_features
            loss /= num_features 
            accuracy /= num_features
            return im_next, loss, accuracy, action_likelyhood, action_likelyhood_true
            
        return im_next


    def evaluate_model_old(self): #delete later
        
        # self.eval()
        
        #Training loss and accuracy
        im_next_predicted = self.step(self.im_seq[0:1,], self.miso_matrix)
        im_next_actual = self.im_seq[1:2,] 
        accuracy = torch.mean((im_next_predicted==im_next_actual).float())
        loss = self.loss_func(self.predictions, self.labels[self.indx_use].reshape(-1, self.act_dim**self.num_dims)).item()
        
        self.training_loss.append(loss)
        self.training_acc.append(accuracy)
        
        #Validation loss and accuracy
        im_next_predicted = self.step(self.im_seq_val[0:1,], self.miso_matrix_val)
        im_next_actual = self.im_seq_val[1:2,] 
        accuracy = torch.mean((im_next_predicted==im_next_actual).float())
        loss = self.loss_func(self.predictions, self.labels_val[self.indx_use].reshape(-1, self.act_dim**self.num_dims)).item()
        
        self.validation_loss.append(loss)
        self.validation_acc.append(accuracy)
        
        
    def evaluate_model(self):
        
        # self.eval()
        
        with torch.no_grad():
            im_next, loss, accuracy, action_likelyhood, action_likelyhood_true = self.step(self.im_seq, self.miso_matrix)
            self.im_next_val, loss_val, accuracy_val, self.action_likelyhood_val, self.action_likelyhood_true_val = self.step(self.im_seq_val, self.miso_matrix_val)
            self.training_loss.append(loss.item())
            self.training_acc.append(accuracy.item())
            self.validation_loss.append(loss_val.item())
            self.validation_acc.append(accuracy_val.item())
    
    
    def train_model_old0(self): #delete later
        
        # self.train()
        
        features, labels = fs.unison_shuffled_copies(self.features, self.labels) #random shuffle 
        
        mid_ix = (np.array(features.shape[1:])/2).astype(int)
        ind = tuple([slice(None)]) + tuple(mid_ix)
        indx_use = torch.nonzero(features[ind])[:,0]
        
        features = features[indx_use,].reshape(-1, self.act_dim**self.num_dims)
        labels = labels[indx_use,].reshape(-1, self.act_dim**self.num_dims)
        
        outputs = self.forward(features)
        loss = self.loss_func(outputs, labels)
        self.optimizer.zero_grad()  # Zero the gradient
        loss.backward()             # Perform backpropagation
        self.optimizer.step()       # Step with optimizer  
        
        
    def train_model_old(self):
        
        # self.train()
        
        features_gen = fs.compute_features_gen(self.im_seq[0:1,], self.batch_sz, self.obs_dim, self.pad_mode)
        labels_gen = fs.compute_labels_gen(self.im_seq, self.batch_sz, self.act_dim, self.energy_dim, self.reg, self.pad_mode)
        
        loss = 0
        num_features = 0
        for _ in range(self.num_iter):
            features = next(features_gen)
            labels = next(labels_gen)
        
            features, labels = fs.unison_shuffled_copies(features, labels) #random shuffle 
            
            mid_ix = (np.array(features.shape[1:])/2).astype(int)
            ind = tuple([slice(None)]) + tuple(mid_ix)
            indx_use = torch.nonzero(features[ind])[:,0]
            if len(indx_use)==1: indx_use = torch.cat([indx_use, indx_use]) #at least two samples needed for batch normalization when training
            
            features = features[indx_use,].reshape(-1, self.act_dim**self.num_dims)
            labels = labels[indx_use,].reshape(-1, self.act_dim**self.num_dims)
            
            outputs = self.forward(features)
            loss += self.loss_func(outputs, labels)*features.shape[0] #times by number of features at each step and average later
            num_features += features.shape[0] #track the number of features used for training
        
        loss = loss/num_features    # Average loss 
        self.optimizer.zero_grad()  # Zero the gradient
        loss.backward()             # Perform backpropagation
        self.optimizer.step()       # Step with optimizer 
        
        
    def train_model(self):
        
        # self.train()
        
        _, loss, _, _, _ = self.step(self.im_seq, self.miso_matrix)
        
        self.optimizer.zero_grad()  # Zero the gradient
        loss.backward()             # Perform backpropagation
        self.optimizer.step()       # Step with optimizer 
        
        
    def plot_old(self, fp_results='./plots'):
        
        if self.num_dims==2:
            #Plot the next images, predicted and true, together
            fig, axs = plt.subplots(1,3)
            axs[0].matshow(self.im_seq_val[0,0,].cpu().numpy())
            axs[0].set_title('Current')
            axs[0].axis('off')
            axs[1].matshow(self.im_next[0,0,].cpu().numpy()) 
            axs[1].set_title('Predicted Next')
            axs[1].axis('off')
            axs[2].matshow(self.im_seq_val[1,0,].cpu().numpy()) 
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
            p2 = axs[1].matshow(np.mean(self.labels_val.cpu().numpy(), axis=0), vmin=0, vmax=1) 
            fig.colorbar(p2, ax=axs[1])
            axs[1].plot(ctr,ctr,marker='x')
            axs[1].set_title('True')
            axs[1].axis('off')
            plt.savefig('%s/action_likelihood.png'%fp_results)
            plt.show()
            
        if self.num_dims==3:
            bi = int(self.im_seq.shape[-1]/2)
            bi0 = int(self.im_next.shape[-1]/2)
            
            #Plot the next images, predicted and true, together
            fig, axs = plt.subplots(1,3)
            axs[0].matshow(self.im_seq_val[0,0,...,bi].cpu().numpy())
            axs[0].set_title('Current')
            axs[0].axis('off')
            axs[1].matshow(self.im_next[0,0,...,bi0].cpu().numpy()) 
            axs[1].set_title('Predicted Next')
            axs[1].axis('off')
            axs[2].matshow(self.im_seq_val[1,0,...,bi].cpu().numpy()) 
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
            p2 = axs[1].matshow(np.mean(self.labels_val.cpu().numpy(), axis=0)[...,ctr], vmin=0, vmax=1) 
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
        axs[0].set_title('Loss (%.3f)'%np.min(self.validation_loss))
        axs[0].legend()
        axs[1].plot(self.validation_acc, '-*', label='Validation')
        axs[1].plot(self.training_acc, '--*', label='Training')
        axs[1].set_title('Accuracy (%.3f)'%np.max(self.validation_acc))
        axs[1].legend()
        plt.savefig('%s/train_val_loss_accuracy.png'%fp_results)
        plt.show()
        
        plt.close('all')
        
        
    def plot(self, fp_results='./plots'):
        
        # Plot the initial image and next images (predicted and true)
        s_3d = (int(self.im_seq_val.shape[-1]/2),)*(self.num_dims-2)
        s0 = (0,0,slice(None),slice(None)) + s_3d
        s1 = (1,0,slice(None),slice(None)) + s_3d
        
        fig, axs = plt.subplots(1,3)
        axs[0].matshow(self.im_seq_val[s0].cpu())
        axs[0].set_title('Current')
        axs[0].axis('off')
        axs[1].matshow(self.im_next_val[s0].cpu()) 
        axs[1].set_title('Predicted Next')
        axs[1].axis('off')
        axs[2].matshow(self.im_seq_val[s1].cpu()) 
        axs[2].set_title('True Next')
        axs[2].axis('off')
        plt.savefig('%s/sim_vs_true.png'%fp_results)
        plt.show()
        
        #Plot the action distributions (predicted and true)
        s_3d = (int(self.act_dim/2),)*(self.num_dims-2)
        s = (slice(None),slice(None),) + s_3d
        
        ctr = int((self.act_dim-1)/2)
        fig, axs = plt.subplots(1,2)
        p1 = axs[0].matshow(self.action_likelyhood_val[s].cpu(), vmin=0, vmax=1)
        fig.colorbar(p1, ax=axs[0])
        axs[0].plot(ctr,ctr,marker='x')
        axs[0].set_title('Predicted')
        axs[0].axis('off')
        p2 = axs[1].matshow(self.action_likelyhood_true_val[s].cpu(), vmin=0, vmax=1) 
        fig.colorbar(p2, ax=axs[1])
        axs[1].plot(ctr,ctr,marker='x')
        axs[1].set_title('True')
        axs[1].axis('off')
        plt.savefig('%s/action_likelihood.png'%fp_results)
        plt.show()
        
        #Plot loss and accuracy (training and validation)
        fig, axs = plt.subplots(1,2)
        axs[0].plot(self.validation_loss, '-*', label='Validation')
        axs[0].plot(self.training_loss, '--*', label='Training')
        axs[0].set_title('Loss (%.3f)'%np.min(self.validation_loss))
        axs[0].legend()
        axs[1].plot(self.validation_acc, '-*', label='Validation')
        axs[1].plot(self.training_acc, '--*', label='Training')
        axs[1].set_title('Accuracy (%.3f)'%np.max(self.validation_acc))
        axs[1].legend()
        plt.savefig('%s/train_val_loss_accuracy.png'%fp_results)
        plt.show()
        
        plt.close('all')
        
        
    def save(self, name):
        torch.save(self.state_dict(), name)
    
    
    
def train_primme(trainset, num_eps, obs_dim=17, act_dim=17, lr=5e-5, reg=1, pad_mode="circular", plot_freq=None):
    
    with h5py.File(trainset, 'r') as f: dims = len(f['ims_id'].shape)-3
    append_name = trainset.split('_kt')[1]
    modelname = "./data/model_dim(%d)_sz(%d_%d)_lr(%.0e)_reg(%d)_ep(%d)_kt%s"%(dims, obs_dim, act_dim, lr, reg, num_eps, append_name)
    agent = PRIMME(obs_dim=obs_dim, act_dim=act_dim, pad_mode=pad_mode, learning_rate=lr, reg=reg, num_dims=dims).to(device)
    
    best_validation_loss = 1e9
    
    for i in tqdm(range(num_eps), desc='Epochs', leave=True):
        agent.sample_data(trainset)
        agent.train_model()
        agent.evaluate_model()
        
        val_loss = agent.validation_loss[-1]
        if val_loss<best_validation_loss:
            best_validation_loss = val_loss
            agent.save(modelname)
            
        if plot_freq is not None: 
            if i%plot_freq==0:
                agent.plot()
    
    return modelname


def run_primme(ic, ea, nsteps, modelname, miso_array=None, pad_mode='circular', plot_freq=None, if_miso=False):
    
    # Setup variables
    agent = PRIMME().to(device)
    agent.load_state_dict(torch.load(modelname))
    agent.pad_mode = pad_mode
    im = torch.Tensor(ic).unsqueeze(0).unsqueeze(0).float().to(device)
    if miso_array is None: miso_array = fs.find_misorientation(ea, mem_max=1) 
    miso_matrix = fs.miso_array_to_matrix(torch.from_numpy(miso_array[None,])).to(device)
    size = ic.shape
    dims = len(size)
    ngrain = len(torch.unique(im))
    tmp = np.array([8,16,32], dtype='uint64')
    dtype = 'uint' + str(tmp[np.sum(ngrain>2**tmp)])
    append_name = modelname.split('_kt')[1]
    sz_str = ''.join(['%dx'%i for i in size])[:-1]
    fp_save = './data/primme_sz(%s)_ng(%d)_nsteps(%d)_freq(1)_kt%s'%(sz_str,ngrain,nsteps,append_name)
    
    
    # Simulate and store in H5
    with h5py.File(fp_save, 'a') as f:
        
        # If file already exists, create another group in the file for this simulaiton
        num_groups = len(f.keys())
        hp_save = 'sim%d'%num_groups
        g = f.create_group(hp_save)
        
        # Save data
        s = list(im.shape); s[0] = nsteps + 1
        dset = g.create_dataset("ims_id", shape=s, dtype=dtype)
        dset2 = g.create_dataset("euler_angles", shape=ea.shape)
        dset3 = g.create_dataset("miso_array", shape=miso_array.shape)
        dset4 = g.create_dataset("miso_matrix", shape=miso_matrix.shape)
        dset[0] = im[0].cpu()
        dset2[:] = ea
        dset3[:] = miso_array #radians (does not save the exact "Miso.txt" file values, which are degrees divided by the cutoff angle)
        dset4[:] = miso_matrix.cpu() #same values as mis0_array, different format
        
        for i in tqdm(range(nsteps), 'Running PRIMME simulation: '):
            
            # Simulate
            if if_miso: im_next = agent.step(im, miso_matrix)
            else: im_next = agent.step(im)
            im = im_next.clone()

            #Store
            dset[i+1,:] = im[0].cpu()
            
            #Plot
            if plot_freq is not None: 
                if i%plot_freq==0:
                    if dims==2: 
                        plt.imshow(im[0,0,].cpu()); plt.show()
                        
    return fp_save
    
    
    
    
    
    
    # Run simulation
    # ims_id = im
    # for i in tqdm(range(nsteps), 'Running PRIMME simulation: '):
        
        #split up the image and pass it through step in batch_dim sizes, then reassemble
        
        #find grid indices
        #find boundary expansion
        #wrap_cut sections and pass them through
        #place each output into a nested list
        #numpy block to get final image
        #save images one at a atime in an h5 file
        
        
    #assumes 2d for now too, just update the "c"s later 
    # with h5py.File(fp_save, 'a') as f:
        
    #     # If file already exists, create another group in the file for this simulaiton
    #     num_groups = len(f.keys())
    #     hp_save = 'sim%d'%num_groups
    #     g = f.create_group(hp_save)
        
    #     # Save data
    #     s = list(im.shape); s[0] = nsteps
    #     dset = g.create_dataset("ims_id", shape=s, dtype=dtype)
    #     dset2 = g.create_dataset("euler_angles", shape=ea.shape)
    #     dset3 = g.create_dataset("miso_array", shape=miso_array.shape)
    #     dset4 = g.create_dataset("miso_matrix", shape=miso_matrix.shape)
    #     dset2[:] = ea
    #     dset3[:] = miso_array #radians (does not save the exact "Miso.txt" file values, which are degrees divided by the cutoff angle)
    #     dset4[:] = miso_matrix.cpu() #same values as mis0_array, different format
        
    #     batch_dims = [np.clip(b, 0, int(2*size[i]/3)) for i, b in enumerate(batch_dims)]
    #     aaa = torch.Tensor(size/np.array(batch_dims)).long()+1
    #     bbb = torch.stack(torch.meshgrid([torch.arange(aa) for aa in aaa])).reshape(2,-1).T.numpy()
    #     n=8+3
    #     for i in tqdm(range(nsteps), 'Running PRIMME simulation: '):
            
    #         im_next = im.clone()
            
    #         for j in bbb:
                
    #             mi = j*np.array(batch_dims) - n
    #             ma = (np.clip((j+1)*np.array(batch_dims), np.zeros(2), size) + n)%size
    #             slices_txt = ':,:,%d:%d,%d:%d'%(mi[0],ma[0],mi[1],ma[1])
    #             batch = fs.wrap_slice(im, slices_txt)
                
    #             batch_p = agent.step(batch.clone(), miso_matrix, evaluate=False)
                
    #             strs = [str(int(mi[0]+n)),str(int(ma[0]-n)),str(int(mi[1]+n)),str(int(ma[1]-n))]
    #             for k in range(len(strs)): 
    #                 if strs[k]=='0': strs[k]=''
    #             exec('im_next[:,:,%s:%s,%s:%s]=batch_p'%tuple(strs))
            
    #         im = im_next
            
    #         #Store
    #         dset[i,:] = im[0].cpu()
            
    #         #Plot
    #         if plot_freq is not None: 
    #             if i%plot_freq==0:
    #                 if dims==2: 
    #                     plt.imshow(im[0,0,].cpu()); plt.show()
            
            
            
            #current limitation
            #2d
            #batch_dims can't be perfectly divisible into ic.shape
        
        
        
        
        
        
    #     im = agent.step(im.clone(), miso_matrix[None,].to(device), evaluate=False)
    #     ims_id = torch.cat([ims_id, im])
    #     if plot_freq is not None: 
    #         if i%plot_freq==0:
    #             if dims==2: 
    #                 plt.imshow(im[0,0,].cpu()); plt.show()
    #             else: 
    #                 m = int(size[0]/2)
    #                 plt.imshow(im[0,0,m].cpu()); plt.show()
            
    # ims_id = ims_id.cpu().numpy()
    
    # # Save Simulation
    # with h5py.File(fp_save, 'a') as f:
        
    #     # If file already exists, create another group in the file for this simulaiton
    #     num_groups = len(f.keys())
    #     hp_save = 'sim%d'%num_groups
    #     g = f.create_group(hp_save)
        
    #     # Save data
    #     dset = g.create_dataset("ims_id", shape=ims_id.shape, dtype=dtype)
    #     dset2 = g.create_dataset("euler_angles", shape=ea.shape)
    #     dset3 = g.create_dataset("miso_array", shape=miso_array.shape)
    #     dset4 = g.create_dataset("miso_matrix", shape=miso_matrix.shape)
    #     dset[:] = ims_id
    #     dset2[:] = ea
    #     dset3[:] = miso_array #radians (does not save the exact "Miso.txt" file values, which are degrees divided by the cutoff angle)
    #     dset4[:] = miso_matrix #same values as mis0_array, different format

    