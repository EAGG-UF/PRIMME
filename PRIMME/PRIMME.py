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
import functions as fs
import dice
from tqdm import tqdm
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# BUILD PRIMME CLASS
class PRIMME(nn.Module):
    def __init__(self, obs_dim=17, act_dim=17, pad_mode="circular", learning_rate=0.00005, num_dims=2, device = "cpu"):
        super(PRIMME, self).__init__()

        # ESTABLISH PARAMETERS USED BY CLASS
        self.device = device      # Device for processing
        self.obs_dim = obs_dim                  # Observation (input) size (one side of a square)
        self.act_dim = act_dim                  # Action (output) size (one side of a square)
        self.pad_mode = pad_mode                # Padding mode ("circular" or "reflect")
        self.learning_rate = learning_rate      # Learning rate
        self.num_dims = num_dims                # Number of Dimensions (2 or 3)

        # ESTABLISH VARIABLES TRACKED
        self.training_loss = []                 # Training loss (history)
        self.training_acc = []                  # Training Accuracy (history)
        self.loss = 0                           # Current loss
        self.accuracy = 0                       # Current accuracy

        # ESTABLISH DATA
        self.im_out  = []
        self.prd_out = []

        # DEFINE NEURAL NETWORK
        self.f1 = nn.Linear(self.obs_dim ** self.num_dims, self.obs_dim ** self.num_dims * 4)
        self.f2 = nn.Linear(self.obs_dim ** self.num_dims * 4, self.obs_dim ** self.num_dims * 2)
        self.f3 = nn.Linear(self.obs_dim ** self.num_dims * 2, self.obs_dim ** self.num_dims)
        self.f4 = nn.Linear(self.obs_dim ** self.num_dims * 1, self.act_dim ** self.num_dims)
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
        h2 = F.relu(self.f2(h1))
        h3 = F.relu(self.f3(h2))
        y  = torch.sigmoid(self.f4(h3))
        
        return y

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
        
        batch_size = 50000
        features_split = torch.split(features, batch_size)
        predictions_split = []
        action_values_split = []
        
        for e in features_split:
            
            predictions = torch.Tensor(self.forward(e)).to(self.device)            
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
        loss = self.loss_func(self.predictions, self.labels.reshape(-1, self.act_dim**self.num_dims))
        
        return loss, accuracy
        

    def train(self,steps=1):
        # def train: Train the PRIMME neural network architecture with self.im_seq. The first image in self.im_seq
        # is used as the initial condition and the last image in self.im_seq is the desired end goal

        # PULL IMAGES AND LAST IMAGE IN SEQUENCE
        labels = fs.compute_action_labels(self.im_seq, act_dim=self.act_dim, pad_mode=self.pad_mode).float()

        # INITIALIZE LOSS AS ZERO
        loss = torch.zeros(1)

        # APPLY PRIMME UNTIL WE REACH LAST TIME STEP
        myim = self.im_seq[0:1, ]       # Initialize to first image
        self.im_out = []                # Initialize growth image
        self.prd_out = []               # Initialize output
        #self.eng_out = []               # Initialize energy output
        self.tot_err = []
        #self.tot_eng = []
        #for k in range(self.im_seq.shape[0]):
        for k in range(steps):
            myim, output = self.step(myim)  # PERFORM ONE GROWTH STEP

            # APPEND OUTPUTS TO A LIST
            self.im_out.append(myim)
            self.prd_out.append(output)
            #self.eng_out.append(energy_out)
            self.tot_err.append(torch.mean((myim == self.im_seq[k+1,0,]).float()))
            #self.tot_eng.append(torch.relu(torch.sum(energy_out)))

            energy_change = fs.compute_action_energy_change(self.im_seq[k:k+1,], self.im_seq[k+1:k+2,], energy_dim=3,
                                                            act_dim=self.act_dim, pad_mode=self.pad_mode)
            # loss = loss + self.tot_err[k] + self.tot_eng[k]
            loss = loss + F.mse_loss(output, (labels[:, :, k] + energy_change[:,:,0].T))

        #for k in range(steps):

        # COMPUTE LOSS (MSE + L1 LOSS TO MINIMIZE ENERGY)
        #features = fs.compute_features(myim, obs_dim=self.obs_dim, pad_mode=self.pad_mode).reshape([-1, self.obs_dim ** self.num_dims])  # COMPUTE NEW NEIGHBORS
        #loss = loss + F.mse_loss(output, labels[:, :, -1]) + (1-torch.mean((myim == self.im_seq[-1,]).float()))# + 0.5 * torch.norm(features, 1)  # Add losses
        #energy_change = fs.compute_action_energy_change(self.im_seq[0:1, ] , myim, energy_dim=3, act_dim=self.act_dim, pad_mode=self.pad_mode)
        #loss = loss + F.mse_loss(output, labels[:, :, steps-1]) + (1 - torch.mean((myim == self.im_seq[steps,]).float())) + torch.sum(torch.mean(torch.relu(energy_change),dim=1),dim=0) # + 0.5 * torch.norm(features, 1)  # Add losses
        #loss = loss + (1 - torch.mean((myim == self.im_seq[steps,]).float())) + torch.sum(torch.mean(torch.relu(energy_change), dim=1), dim=0)

        # COMPUTE LOSS AND ACCURACY
        self.loss = loss.detach().cpu().numpy()
        self.accuracy = torch.mean((myim == self.im_seq[steps,]).float()).numpy()
        self.training_loss.append(self.loss)
        self.training_acc.append(self.accuracy)

        # OPTIMIZE FOR THIS BATCH
        self.optimizer.zero_grad()  # Zero the gradient
        loss.backward()             # Perform backpropagation
        self.optimizer.step()       # Step with optimizer

    def plot(self, fp_results='./results_training'):
        # def plot: Plot information of importance for the current PRIMME object
        #   Inputs--
        #f  p_results: location where to store the results

        if self.num_dims == 2:
            # Plot the next images, predicted and true, together
            fig, axs = plt.subplots(2, self.im_seq.shape[0])
            for k in range(len(self.im_out)):
                axs[0,k].matshow(self.im_out[k][0,0,].cpu().numpy())
                axs[0,k].axis('off')
                axs[1, k].matshow(self.im_seq[k, 0,].cpu().numpy())
                axs[1, k].axis('off')
            plt.savefig('%s/sim_vs_true.png' % fp_results)
            # plt.show()

            # Plot the action distributions, predicted and true, together
            ctr = int((self.act_dim - 1) / 2)   # Center pixel in output
            labels = fs.compute_action_labels(self.im_seq, act_dim=self.act_dim, pad_mode=self.pad_mode).float()
            fig, axs = plt.subplots(2, labels.shape[2])
            for k in range(len(self.prd_out)):
                pred = self.prd_out[k].reshape(-1, self.act_dim, self.act_dim).detach().cpu().numpy()
                p1 = axs[0,k].matshow(np.mean(pred, axis=0))
                #fig.colorbar(p1, ax=axs[0])
                axs[0,k].plot(ctr, ctr, marker='x')
                axs[0,k].axis('off')
                p1 = axs[1, k].matshow(np.mean(labels[:,:,k].numpy(),axis=0).reshape(self.act_dim,self.act_dim))
                #fig.colorbar(p1, ax=axs[0])
                axs[1, k].plot(ctr, ctr, marker='x')
                axs[1, k].axis('off')
            plt.savefig('%s/action_likelihood.png' % fp_results)
            # plt.show()

            # Plot loss and accuracy
            fig, axs = plt.subplots(1, 2)
            axs[0].plot(self.training_loss, '--*', label='Training')
            axs[0].set_title('Loss')
            axs[0].legend()
            axs[1].plot(self.training_acc, '--*', label='Training')
            axs[1].set_title('Accuracy')
            axs[1].legend()
            plt.savefig('%s/train_val_loss_accuracy.png' % fp_results)
            # plt.show()

            plt.close('all')

        if self.num_dims == 3:
            bi = int(self.im_seq.shape[-1] / 2)

            # UPDATE THIS

    def save(self, name):
        # self.model.save(name)
        torch.save(self.state_dict(), name)

def train_primme(trainset, num_eps, obs_dim=17, act_dim=17, lr=5e-5, reg=1, pad_mode="circular", plot_freq=None, if_miso=False, multi_epoch_safe=False):
    
    with h5py.File(trainset, 'r') as f: dims = len(f['ims_id'].shape)-3
    append_name = trainset.split('_kt')[1]
    modelname = "./data/model_dim(%d)_sz(%d_%d)_lr(%.0e)_reg(%d)_ep(%d)_kt%s"%(dims, obs_dim, act_dim, lr, reg, num_eps, append_name)
    agent = PRIMME(obs_dim=obs_dim, act_dim=act_dim, pad_mode=pad_mode, learning_rate=lr, reg=reg, num_dims=dims, if_miso=if_miso).to(device)
    
    # # Code to split into GPUs
    # top_agent = PRIMME(obs_dim=obs_dim, act_dim=act_dim, pad_mode=pad_mode, learning_rate=lr, reg=reg, num_dims=dims, if_miso=if_miso).to(device)
    # top_agent = nn.DataParallel(top_agent)
    
    # # Move model to device
    # top_agent.to(device)
    # agent = top_agent.module
    
    best_validation_loss = 1e9
    
    for i in tqdm(range(num_eps), desc='Epochs', leave=True):
        agent.sample_data(trainset)
        agent.train_model()
        agent.evaluate_model()
        
        val_loss = agent.validation_loss[-1]
        if multi_epoch_safe and i % 200 ==0:
            agent.save(f"{modelname[:-3]}_at_epoch({i}).h5")

        if val_loss<best_validation_loss:
            best_validation_loss = val_loss
            agent.save(modelname)
            
        if plot_freq is not None: 
            if i%plot_freq==0:
                agent.plot()
                
                tmpx = np.stack(agent.logx).T
                tmpy = np.stack(agent.logy).T
                
                plt.figure()
                plt.plot(tmpx)
                plt.plot(tmpy)
                plt.show()
                # tmp0 = np.stack(agent.log0).T
                # tmp1 = np.stack(agent.log1).T
                
                # # print(np.mean(tmp0))
                # # print(np.mean(tmp1))
                
                # plt.figure()
                # plt.plot(tmp0[0], 'C0-') 
                # plt.plot(tmp0[1], 'C0--') 
                # plt.plot(tmp1[0], 'C1-') 
                # plt.plot(tmp1[1], 'C1--')  
                # plt.legend(['Mean Distribution (x)','Mean Distribution (y)','Mean Index (x)','Mean Index (y)'])
                # plt.xlabel('Number of training iterations')
                # plt.ylabel('Num pixels from (0,0)')
                # plt.show()
    
    return modelname


def run_primme(ic, ea, nsteps, modelname, miso_array=None, pad_mode='circular', plot_freq=None, if_miso=False):
    
    # Setup variables
    d = len(ic.shape)
    # Dimensions
    obs_dim, act_dim = np.array(modelname.split("sz(")[1].split(")_lr(")[0].split("_")).astype(int)
    # Code Pre-Split
    agent = PRIMME(num_dims=d, obs_dim=obs_dim, act_dim=act_dim).to(device)
    
    # for windows:
    # agent.load_state_dict(torch.load(modelname))
    # for mac:
    agent.load_state_dict(torch.load(modelname, map_location=torch.device('cpu')))
    
    agent.pad_mode = pad_mode
    im = torch.Tensor(ic).unsqueeze(0).unsqueeze(0).float().to(device)
    if miso_array is None: miso_array = fs.find_misorientation(ea, mem_max=1) 
    miso_matrix = fs.miso_array_to_matrix(torch.from_numpy(miso_array[None,])).to(device)
    size = ic.shape
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
        dset4 = g.create_dataset("miso_matrix", shape=miso_matrix[0].shape)
        dset[0] = im[0].cpu()
        dset2[:] = ea
        dset3[:] = miso_array #radians (does not save the exact "Miso.txt" file values, which are degrees divided by the cutoff angle)
        dset4[:] = miso_matrix[0].cpu() #same values as mis0_array, different format
        
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
                    plt.figure()
                    s = (0,0,slice(None), slice(None),) + (int(im.shape[-1]/2),)*(d-2)
                    plt.imshow(im[s].cpu(), interpolation=None) 
                    plt.show()
                    
                    
                    
                    # tmp0 = np.stack(agent.log0).T
                    # tmp1 = np.stack(agent.log1).T
                    
                    # plt.figure()
                    # plt.plot(tmp0[0], 'C0-') 
                    # plt.plot(tmp0[1], 'C0--') 
                    # plt.plot(tmp1[0], 'C1-') 
                    # plt.plot(tmp1[1], 'C1--')  
                    # plt.legend(['Mean Distribution (x)','Mean Distribution (y)','Mean Index (x)','Mean Index (y)'])
                    # plt.xlabel('Number of Frames')
                    # plt.ylabel('Num pixels from (0,0)')
                    # plt.show()
                    
                    
                    # plt.figure()
                    # tmp0 = np.stack(log0).T
                    # tmp1 = np.stack(log1).T
                    # plt.plot(tmp0[0], tmp0[1], ',') 
                    # plt.plot(tmp1[0], tmp1[1], ',') 
                    
                    # m = np.max([np.max(np.abs(tmp0)), np.max(np.abs(tmp1))])
                    
                    # plt.axis('square')
                    # plt.xlim([-m,m])
                    # plt.ylim([-m,m])
                    # plt.legend(['CoM of mean distribution','Mean chosen index'])
                    # plt.show()
                        
    return fp_save

'''-------------------------------------------------------------------------'''






'''-------------------------------------------------------------------------'''


