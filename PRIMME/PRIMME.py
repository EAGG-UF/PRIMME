#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DESCRIPTION:
    This script defines the PRIMME simulator class used to simulate microstructural grain growth
    The neural network model used to predict the action likelihood is written in Tensorflow (Keras)
    The functions besides of the model are written in Pytorch to parallelize the operations using GPUs if available
    This class must be passed a SPPARKS class ('env'), which provides an initial condition, and training data, features, and labels 
    The main functions of the class include predicting the action likelihood (given an intial condition) and training the model (given features and labels)

IF THIS CODE IS USED FOR A RESEARCH PUBLICATION, please cite (https://arxiv.org/abs/2203.03735): 
    Yan, Weishi, et al. "Predicting 2D Normal Grain Growth using a Physics-Regularized Interpretable Machine Learning Model." arXiv preprint arXiv:2203.03735 (2022).
"""

# IMPORT LIBRARIES
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from pathlib import Path
import os
from random import shuffle
import functions as fs
# import dice
from tqdm import tqdm
device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# BUILD PRIMME CLASS
class PRIMME(nn.Module):
    def __init__(self, obs_dim=17, act_dim=17, pad_mode="circular", learning_rate=0.00005, reg=1, num_dims=2, mode = "Single_Step", device = "cpu"):
        super(PRIMME, self).__init__()

        # ESTABLISH PARAMETERS USED BY CLASS
        #self.device = torch.device("cpu")       # Device for processing
        self.device = device       # Device for processing
        self.obs_dim = obs_dim                  # Observation (input) size (one side of a square)
        self.act_dim = act_dim                  # Action (output) size (one side of a square)
        self.pad_mode = pad_mode                # Padding mode ("circular" or "reflect")
        self.learning_rate = learning_rate      # Learning rate
        self.reg = reg
        self.num_dims = num_dims                # Number of Dimensions (2 or 3)
        self.mode = mode

        # ESTABLISH VARIABLES TRACKED
        self.training_loss = []                 # Training loss (history)
        self.training_acc = []                  # Training Accuracy (history)
        self.validation_loss = []
        self.validation_acc = []
        self.seq_samples = []
        self.im_seq_T = None
        
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
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.loss_func = torch.nn.MSELoss()  # Mean squared error loss
        self.optimizer.zero_grad()  # Make all the gradients zero

    def forward(self, x):
        # def forward: Run input X through the neural network
        #   Inputs--
        #        x: microstructure features around a center pixel
        #   Outputs--
        #       y: "action likelihood" for the center pixel to flip to each the grain associated with each other pixel

        h1 = F.relu(self.f1(x))
        out = self.dropout(h1)   
        out = self.BatchNorm1(out)
        h2 = F.relu(self.f2(out))
        out = self.dropout(h2)
        out = self.BatchNorm2(out)
        h3 = F.relu(self.f3(out))
        out = self.dropout(h3)
        out = self.BatchNorm3(out)
        #y  = torch.sigmoid(self.f4(out))
        y  = F.relu(self.f4(out))
        
        return y

    def load_data(self, n_step, n_samples, h5_path = 'spparks_data_size257x257_ngrain256-256_nsets200_future4_max100_offset1_kt0.h5'):

        with h5py.File(h5_path, 'r') as f:
            print(f.keys())
            ims_id = f['ims_id'][:]
            miso_array = f['miso_array'][:]
        self.im_seq_T = torch.from_numpy(ims_id[:n_samples, :n_step])
        self.miso_array_T = miso_array[:n_samples]
        self.seq_samples = list(np.arange(len(self.im_seq_T)))

    def sample_data(self, batch_size = 1):
           
        i_max = self.im_seq_T.shape[0]
        i_batch = np.sort(np.random.randint(low=0, high=i_max, size=(batch_size,)))
        batch = self.im_seq_T[i_batch,]
        miso_array = self.miso_array_T[i_batch,] 
        
        self.im_seq = torch.from_numpy(batch[0,].astype(float)).to(self.device)
        miso_array = torch.from_numpy(miso_array.astype(float)).to(self.device)
        self.miso_matrix = fs.miso_conversion(miso_array)
        
        #Compute features and labels
        self.features = fs.compute_features(self.im_seq[0:1,], obs_dim=self.obs_dim, pad_mode=self.pad_mode)
        self.labels = fs.compute_labels(self.im_seq, obs_dim=self.obs_dim, act_dim=self.act_dim, reg=self.reg, pad_mode=self.pad_mode)
     
    def step(self, im, evaluate=True):
        # def step: Apply one step of growth to microstructure image IM
        #   Inputs--
        #        im: initial microstructure ID image
        #   Outputs--
        #    im_out: new microstructure ID image after one growth step

        features = fs.compute_features(im, obs_dim=self.obs_dim, pad_mode=self.pad_mode)
        mid_ix = (np.array(features.shape[1:])/2).astype(int)
        ind = tuple([slice(None)]) + tuple(mid_ix)
        indx_use = torch.nonzero(features[ind])[:,0]
        features = features[indx_use,]
        
        action_features = fs.my_unfoldNd(im, kernel_size=self.act_dim, pad_mode=self.pad_mode)[0,] 
        action_features = action_features[...,indx_use]
        
        batch_size = 500000
        features_split = torch.split(features, batch_size)
        predictions_split = []
        action_values_split = []
        
        for e in features_split:
            
            #print(e.shape)
            predictions = self.forward(e.reshape(-1, self.act_dim**self.num_dims))        
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

    def train(self, evaluate=True):
        # def train: Train the PRIMME neural network architecture with self.im_seq. The first image in self.im_seq
        # is used as the initial condition and the last image in self.im_seq is the desired end goal
         
        shuffle(self.seq_samples)
        for seq_sample in self.seq_samples:
            self.seq_sample = seq_sample
            self.im_seq = self.im_seq_T[seq_sample].to(self.device)            
            self.features = fs.compute_features(self.im_seq[0:1,], obs_dim=self.obs_dim, pad_mode=self.pad_mode)
            self.labels = fs.compute_labels(self.im_seq, obs_dim=self.obs_dim, act_dim=self.act_dim, reg=self.reg, pad_mode=self.pad_mode)
     
            if evaluate: 
                loss, accuracy = self.compute_metrics()
                self.validation_loss.append(loss.detach().cpu().numpy())
                self.validation_acc.append(accuracy.detach().cpu().numpy())            
        
            features, labels = fs.unison_shuffled_copies(self.features, self.labels) #random shuffle             
            mid_ix = (np.array(features.shape[1:])/2).astype(int)
            ind = tuple([slice(None)]) + tuple(mid_ix)
            indx_use = torch.nonzero(features[ind])[:,0]
            features = features[indx_use,]
            labels = labels[indx_use,]
            
            outputs = self.forward(features.reshape(-1, self.act_dim**self.num_dims))
            loss = self.loss_func(outputs, labels.reshape(-1, self.act_dim**self.num_dims))
            self.optimizer.zero_grad()  # Zero the gradient
            loss.backward()             # Perform backpropagation
            self.optimizer.step()       # Step with optimizer             
                
            if evaluate: 
                loss, accuracy = self.compute_metrics()
                self.training_loss.append(loss.detach().cpu().numpy())
                self.training_acc.append(accuracy.detach().cpu().numpy())
            
    def compute_metrics(self):
        
        im_next_predicted = self.step(self.im_seq[0:1,])
        im_next_actual = self.im_seq[1:2,]
        accuracy = torch.mean((im_next_predicted==im_next_actual).float())
        loss = self.loss_func(self.predictions, self.labels[self.indx_use,].reshape(-1,self.act_dim**self.num_dims))
        
        return loss, accuracy

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
            plt.savefig('%s/sim_vs_true(%s).png'%(fp_results, str(self.seq_sample)))
           
            
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
            plt.savefig('%s/action_likelihood(%s).png'%(fp_results, str(self.seq_sample)))
           
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
            plt.savefig('%s/train_val_loss_accuracy(%s).png'%(fp_results, str(self.seq_sample)))
            
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
        
    def save(self, name):
        # self.model.save(name)
        torch.save(self.state_dict(), name)

def train_primme(trainset, n_step, n_samples, mode = "Single_Step", num_eps=25,
                 dims=2, obs_dim=17, act_dim=17, lr=5e-5, reg=1, pad_mode="circular", if_plot=False):

    agent = PRIMME(obs_dim=obs_dim, act_dim=act_dim, pad_mode=pad_mode, learning_rate=lr, 
                   num_dims=dims, mode = mode, device = device).to(device)    

    agent.load_data(h5_path=trainset, n_step=n_step, n_samples=n_samples)
    append_name = trainset.split('_kt')[0].split("spparks_")[1]
    
    for epoch in tqdm(range(1, num_eps+1), desc='Epochs', leave=True):  
        #agent.sample_data(h5_path=trainset, batch_size=1)
        agent.train()
        if epoch % 5 == 0: 
            agent.subfolder = "pred(%s)_%s_ep(%d)_pad(%s)_md(%d)_sz(%d_%d)_lr(%.0e)_reg(%s)" % (agent.mode, append_name, epoch, agent.pad_mode, agent.num_dims,
                                                                                                agent.obs_dim, agent.act_dim, agent.learning_rate, agent.reg)
            agent.result_path = ("/").join(['./plots', agent.subfolder])
            if not os.path.exists(agent.result_path):
                os.makedirs(agent.result_path)              
            if if_plot: agent.plot()
            modelname = './data/' + agent.subfolder + '.h5'
            agent.save("%s" % modelname)
            ## Generate test case and Run PRIMME model    
            '''
            nsteps = 1800
            for key in test_case_dict.keys():
                grain_shape, grain_sizes = test_case_dict[key]
                if grain_shape == "hex":
                    ic_shape = grain_shape
                else:   
                    ic_shape = grain_shape + "(" + ("_").join([str(grain_sizes[0][0]), str(grain_sizes[0][1]), str(grain_sizes[1])]) + ")"
                filename_test = ic_shape + ".pickle"    
                path_load = Path('./data').joinpath(filename_test)
                if os.path.isfile(str(path_load)):  
                    data_dict = fs.load_picke_files(load_dir = Path('./data'), filename_save = filename_test)
                    ic, ea, miso_array, miso_matrix = data_dict["ic"], data_dict["ea"], data_dict["miso_array"], data_dict["miso_matrix"]  
                else:
                    ic, ea, miso_array, miso_matrix = fs.generate_train_init(filename_test, grain_shape, grain_sizes, device)
                ## Run PRIMME model
                ims_id, fp_primme = run_primme(ic, ea, miso_array, miso_matrix, nsteps, ic_shape, modelname)
                sub_folder = "pred" + fp_primme.split("/")[2].split(".")[0].split("pred")[1]
                fs.compute_grain_stats(fp_primme)
                fs.make_videos(fp_primme, sub_folder, ic_shape)
                if grain_shape == "grain":
                    fs.make_time_plots(fp_primme, sub_folder, ic_shape)    
            '''


def run_primme(ic, ea, miso_array, miso_matrix, nsteps, ic_shape, modelname, pad_mode='circular',  mode = "Single_Step", if_plot=False):
    
    # Setup
    agent = PRIMME(pad_mode=pad_mode, mode = mode, device = device).to(device)
    agent.load_state_dict(torch.load(modelname, map_location=torch.device('cpu')))
    im = torch.Tensor(ic).unsqueeze(0).unsqueeze(0).float()
    ngrain = len(torch.unique(im))
    tmp = np.array([8,16,32], dtype='uint64')
    dtype = 'uint' + str(tmp[np.sum(ngrain>2**tmp)])
    #if np.all(miso_array==None): miso_array = fs.find_misorientation(ea, mem_max=1) 
    #miso_matrix = fs.miso_conversion(torch.from_numpy(miso_array[None,]))[0]
    fp_save = './data/primme_shape(%s)_%s'%(ic_shape, modelname.split('/')[2])
    
    # Run simulation
    agent.eval()
    with torch.no_grad():    
        ims_id = im
        for _ in tqdm(range(nsteps), 'Running PRIMME simulation: '):
            im = agent.step(im.clone().to(device))
            ims_id = torch.cat([ims_id, im.detach().cpu()])
            if if_plot: plt.imshow(im[0,0,].detach().cpu().numpy()); plt.show()
    ims_id = ims_id.cpu().numpy()
    
    # Save Simulation
    with h5py.File(fp_save, 'w') as f:
        
        # If file already exists, create another group in the file for this simulaiton
        num_groups = len(f.keys())
        hp_save = 'sim%d'%num_groups
        g = f.create_group(hp_save)
        
        # Save data
        dset = g.create_dataset("ims_id", shape=ims_id.shape, dtype=dtype)
        dset2 = g.create_dataset("euler_angles", shape=ea.shape)
        dset3 = g.create_dataset("miso_array", shape=miso_array.shape)
        dset4 = g.create_dataset("miso_matrix", shape=miso_matrix.shape)
        dset[:] = ims_id
        dset2[:] = ea
        dset3[:] = miso_array #radians (does not save the exact "Miso.txt" file values, which are degrees divided by the cutoff angle)
        dset4[:] = miso_matrix #same values as mis0_array, different format

    return ims_id, fp_save

'''-------------------------------------------------------------------------'''

def sample_data(h5_path = "./data/trainset_spparks_sz(257x257)_ng(256-256)_nsets(200)_future(4)_max(100)_kt(0.66)_cut(0).h5", 
                batch_size = 1, obs_dim = 17, act_dim = 17, reg = 1, pad_mode = "circular", device = 'cpu'):
    
    with h5py.File(h5_path, 'r') as f:
        print("Keys: %s" % f.keys())          
        i_max = f['ims_id'].shape[0]
        i_batch = np.sort(np.random.randint(low=0, high=i_max, size=(batch_size,)))
        batch = f['ims_id'][i_batch,]
        miso_array = f['miso_array'][i_batch,] 
    
    im_seq = torch.from_numpy(batch[0,].astype(float)).to(device)
    miso_array = torch.from_numpy(miso_array.astype(float)).to(device)
    miso_matrix = fs.miso_conversion(miso_array)
    
    #Compute features and labels
    features = fs.compute_features(im_seq[0:1,], obs_dim=obs_dim, pad_mode=pad_mode)
    labels = fs.compute_labels(im_seq, obs_dim=obs_dim, act_dim=act_dim, reg=reg, pad_mode=pad_mode)
    
    plt.figure()
    plt.imshow(im_seq[0, 0].cpu().numpy())
    plt.show()
    
    #features.shape
    





