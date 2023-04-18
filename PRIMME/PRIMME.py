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


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


class PRIMME(nn.Module):
    def __init__(self, obs_dim=17, act_dim=17, pad_mode='circular', learning_rate=5e-5, reg=1, num_dims=2, batch_dims=[46, 46, 46], cfg='./cfg/dqn_setup.json'):
        super(PRIMME, self).__init__()
        
        # self.device = device
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.pad_mode = pad_mode
        self.learning_rate = learning_rate
        self.reg = reg
        self.num_dims = num_dims
        self.batch_dims = batch_dims
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
        
        
        # n = self.obs_dim ** self.num_dims
        # self.f1 = nn.Linear(n, int(n*0.6))
        # self.f2 = nn.Linear(int(n*0.6), int(n*0.2))
        # self.f3 = nn.Linear(int(n*0.2), int(n*0.6))
        # self.f4 = nn.Linear(int(n*0.6), self.act_dim ** self.num_dims)
        # self.dropout = nn.Dropout(p = 0.25) 
        # self.BatchNorm1 = nn.BatchNorm1d(int(n*0.6))
        # self.BatchNorm2 = nn.BatchNorm1d(int(n*0.2))
        # self.BatchNorm3 = nn.BatchNorm1d(int(n*0.6))
        
        
        # n0 = self.obs_dim ** self.num_dims
        # n = 21*21
        # self.f1 = nn.Linear(n0, int(n*4))
        # self.f11 = nn.Linear(int(n*4), int(n*3))
        # self.f2 = nn.Linear(int(n*3), int(n*2))
        # self.f22 = nn.Linear(int(n*2), int(n*1))
        # self.f3 = nn.Linear(int(n*1), int(n*0.5))
        # self.f33 = nn.Linear(int(n*0.5), int(n*0.5))
        # self.f4 = nn.Linear(int(n*0.5), self.act_dim ** self.num_dims)
        # self.dropout = nn.Dropout(p = 0.25) 
        # self.BatchNorm1 = nn.BatchNorm1d(int(n*4))
        # self.BatchNorm11 = nn.BatchNorm1d(int(n*3))
        # self.BatchNorm2 = nn.BatchNorm1d(int(n*2))
        # self.BatchNorm22 = nn.BatchNorm1d(int(n*1))
        # self.BatchNorm3 = nn.BatchNorm1d(int(n*0.5))
        # self.BatchNorm33 = nn.BatchNorm1d(int(n*0.5))
        
        
        

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
        # y  = torch.sigmoid(self.f4(out))
        y  = torch.relu(self.f4(out))
        
        
        
        # out = F.relu(self.f1(x))
        # out = self.dropout(out)   
        # out = self.BatchNorm1(out)
        # out = F.relu(self.f11(out))
        # out = self.dropout(out)   
        # out = self.BatchNorm11(out)
        # out = F.relu(self.f2(out))
        # out = self.dropout(out)
        # out = self.BatchNorm2(out)
        # out = F.relu(self.f22(out))
        # out = self.dropout(out)
        # out = self.BatchNorm22(out)
        # out = F.relu(self.f3(out))
        # out = self.dropout(out)
        # out = self.BatchNorm3(out)
        # out = F.relu(self.f33(out))
        # out = self.dropout(out)
        # out = self.BatchNorm33(out)
        # # y  = torch.sigmoid(self.f4(out))
        # y  = torch.relu(self.f4(out))
        
        
        
        return y
    
    
    def sample_data(self, h5_path='spparks_data_size257x257_ngrain256-256_nsets200_future4_max100_offset1_kt0.h5'):
        
        with h5py.File(h5_path, 'r') as f:
            i_max = f['ims_id'].shape[0]
            i_split = int(i_max*0.8)
            i_train = np.random.randint(low=0, high=i_split)
            i_val = np.random.randint(low=i_split, high=i_max)
            s = f['ims_id'].shape[3:]
            
            r = [np.random.randint(0, s)[0] for e in s]
            tmp = [str(r[j])+':'+str(r[j]+self.batch_dims[j]) for j in range(len(s))]
            tmp = [str(i_train)]+[':']+[':']+tmp
            slices_txt = tmp = ','.join(tmp)
            batch = fs.wrap_slice(f['ims_id'], slices_txt)[None,]
            miso_array = f['miso_array'][i_train,][None,]
            miso_array = miso_array[:, miso_array[0,]!=0] #cut out zeros, each starts with different number of grains
            
            r = [np.random.randint(0, s)[0] for e in s]
            tmp = [str(r[j])+':'+str(r[j]+self.batch_dims[j]) for j in range(len(s))]
            tmp = [str(i_val)]+[':']+[':']+tmp
            slices_txt = tmp = ','.join(tmp)
            batch_val = fs.wrap_slice(f['ims_id'], slices_txt)[None,]
            miso_array_val = f['miso_array'][i_val,][None,]
            miso_array_val = miso_array_val[:, miso_array_val[0,]!=0]
            
        # with h5py.File(h5_path, 'r') as f:
        #     i_max = f['ims_id'].shape[0]
        #     i_split = int(i_max*0.8)
            
        #     i_train = np.sort(np.random.randint(low=0, high=i_split, size=(1,)))
        #     batch = f['ims_id'][i_train,]
        #     miso_array = f['miso_array'][i_train,] 
        #     miso_array = miso_array[:, miso_array[0,]!=0] #cut out zeros, each starts with different number of grains
            
        #     i_val = np.sort(np.random.randint(low=i_split, high=i_max, size=(1,)))
        #     batch_val = f['ims_id'][i_val,]
        #     miso_array_val = f['miso_array'][i_val,] 
        #     miso_array_val = miso_array_val[:, miso_array_val[0,]!=0] #cut out zeros, each starts with different number of grains
            
        self.im_seq = torch.from_numpy(batch[0,].astype(float)).to(device)
        miso_array = torch.from_numpy(miso_array.astype(float)).to(device)
        self.miso_matrix = fs.miso_array_to_matrix(miso_array)
        
        self.im_seq_val = torch.from_numpy(batch_val[0,].astype(float)).to(device)
        miso_array = torch.from_numpy(miso_array_val.astype(float)).to(device)
        self.miso_matrix_val = fs.miso_array_to_matrix(miso_array)
        
        #Compute features
        # self.features = fs.compute_features(self.im_seq[0:1,], obs_dim=self.obs_dim, pad_mode=self.pad_mode)
        # self.features_val = fs.compute_features(self.im_seq_val[0:1,], obs_dim=self.obs_dim, pad_mode=self.pad_mode)
        
        self.features = fs.compute_features_miso(self.im_seq[0:1,], self.miso_matrix, obs_dim=self.obs_dim, pad_mode=self.pad_mode)
        self.features_val = fs.compute_features_miso(self.im_seq_val[0:1,], self.miso_matrix_val, obs_dim=self.obs_dim, pad_mode=self.pad_mode)
        
        # Compute labels
        self.labels = fs.compute_labels(self.im_seq, obs_dim=self.obs_dim, act_dim=self.act_dim, reg=self.reg, pad_mode=self.pad_mode)
        self.labels_val = fs.compute_labels(self.im_seq_val, obs_dim=self.obs_dim, act_dim=self.act_dim, reg=self.reg, pad_mode=self.pad_mode)
        
        # self.labels = fs.compute_labels_miso(self.im_seq, self.miso_matrix, obs_dim=self.obs_dim, act_dim=self.act_dim, reg=self.reg, pad_mode=self.pad_mode)
        # self.labels_val = fs.compute_labels_miso(self.im_seq_val, self.miso_matrix_val, obs_dim=self.obs_dim, act_dim=self.act_dim, reg=self.reg, pad_mode=self.pad_mode)
        
        
        
        #delete later
        # self.labels = self.labels.reshape(48,48,17,17)[3:-3,3:-3,:,:].reshape(-1,17,17) 
        # self.labels_val = self.labels_val.reshape(48,48,17,17)[3:-3,3:-3,:,:].reshape(-1,17,17)
        
        #NEW
        t = round(self.labels.shape[0]**(1/self.num_dims))
        t0 = (t,)*self.num_dims
        t1 = (self.act_dim,)*self.num_dims
        c = 3
        cc = [slice(c,-c),]*self.num_dims
        
        self.labels = self.labels.reshape(t0+t1)[cc].reshape((-1,)+t1)  
        self.labels_val = self.labels_val.reshape(t0+t1)[cc].reshape((-1,)+t1)  
        
        
    def step(self, im, miso_matrix, evaluate=True):
        
        # self.eval()
        
        # features = fs.compute_features(im, obs_dim=self.obs_dim, pad_mode=self.pad_mode)
        features = fs.compute_features_miso(im, miso_matrix, obs_dim=self.obs_dim, pad_mode=self.pad_mode) #use miso functions
        mid_ix = (np.array(features.shape[1:])/2).astype(int)
        ind = tuple([slice(None)]) + tuple(mid_ix)
        indx_use = torch.nonzero(features[ind])[:,0]
        features = features[indx_use,]
        
        action_features = fs.my_unfoldNd(im, kernel_size=self.act_dim, pad_mode=self.pad_mode)[0,] 
        
        
        
        #delete later
        # action_features = action_features.reshape(-1,48,48)[:,3:-3,3:-3].reshape(-1,42*42)
        
        #NEW
        t = tuple(np.array(im.shape[2:]) - self.obs_dim+1)
        c = 3
        cc = [slice(None),]+[slice(c,-c),]*self.num_dims
        
        action_features = action_features.reshape((-1,)+t)[cc].reshape(self.act_dim**self.num_dims,-1)
        
        
        
        
        
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
        
        
        
        #delete this later
        # self.im_next = im[:,:,11:-11,11:-11].flatten().float() 
        # self.im_next[indx_use] = updated_values.float()
        # self.im_next = self.im_next.reshape(1,1,42,42) 
        # self.indx_use = indx_use
        
        #NEW
        c = int(self.obs_dim/2)+int(7/2)
        cc = [slice(None),]*2+[slice(c,-c),]*self.num_dims
        t = tuple(np.array(im.shape[2:]) - self.obs_dim+1 - 6)
        
        self.im_next = im[cc].flatten().float() 
        self.im_next[indx_use] = updated_values.float()
        self.im_next = self.im_next.reshape((1,1,)+t) 
        self.indx_use = indx_use
        
        return self.im_next


    def evaluate_model(self):
        
        # self.eval()
        
        c = int(self.obs_dim/2)+int(7/2)
        cc = [slice(1,2), slice(None),]+[slice(c,-c),]*self.num_dims
        
        #Training loss and accuracy
        im_next_predicted = self.step(self.im_seq[0:1,], self.miso_matrix)
        
        
        
        #delete this later
        # im_next_actual = self.im_seq[1:2,:,11:-11,11:-11] 
        
        #NEW
        im_next_actual = self.im_seq[cc]
        
        
        
        accuracy = torch.mean((im_next_predicted==im_next_actual).float())
        loss = self.loss_func(self.predictions, self.labels[self.indx_use].reshape(-1, self.act_dim**self.num_dims)).item()
        
        self.training_loss.append(loss)
        self.training_acc.append(accuracy)
        
        #Validation loss and accuracy
        im_next_predicted = self.step(self.im_seq_val[0:1,], self.miso_matrix_val)
        
        
        
        #delete this later
        # im_next_actual = self.im_seq_val[1:2,:,11:-11,11:-11] 
        
        #NEW
        im_next_actual = self.im_seq_val[cc] 
        
        
        
        accuracy = torch.mean((im_next_predicted==im_next_actual).float())
        loss = self.loss_func(self.predictions, self.labels_val[self.indx_use].reshape(-1, self.act_dim**self.num_dims)).item()
        
        self.validation_loss.append(loss)
        self.validation_acc.append(accuracy)
        
    
    def train_model(self):
        
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
        
    
    def plot(self, fp_results='./plots'):
        
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
            
            # i=0
            # plt.imshow(self.labels.cpu().numpy()[i]); plt.show()
            # plt.imshow(pred[i]); plt.show()
            # i+=100
            
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
        
    
    # def load(self, name):
    #     self.model = load_model(name)
    #     self.num_dims = len(self.model.layers[0].get_output_at(0).get_shape().as_list()) - 1
    #     self.obs_dim = self.model.layers[0].get_output_at(0).get_shape().as_list()[1]
    #     model_out_dim = self.model.layers[-1].get_output_at(0).get_shape().as_list()[1]
    #     self.act_dim = int(np.rint(model_out_dim**(1/self.num_dims)))
    #     self.learning_rate = K.eval(self.model.optimizer.lr)


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


def run_primme(ic, ea, nsteps, modelname, miso_array=None, pad_mode='circular', batch_dims=[2000,2000], plot_freq=None):
    
    
    
    # Setup
    agent = PRIMME().to(device)
    agent.load_state_dict(torch.load(modelname))
    agent.pad_mode = pad_mode
    im = torch.Tensor(ic).unsqueeze(0).unsqueeze(0).float().to(device)
    size = ic.shape
    dims = len(size)
    ngrain = len(torch.unique(im))
    tmp = np.array([8,16,32], dtype='uint64')
    dtype = 'uint' + str(tmp[np.sum(ngrain>2**tmp)])
    if np.all(miso_array==None): miso_array = fs.find_misorientation(ea, mem_max=1) 
    miso_matrix = fs.miso_array_to_matrix(torch.from_numpy(miso_array[None,])).to(device)
    append_name = modelname.split('_kt')[1]
    sz_str = ''.join(['%dx'%i for i in size])[:-1]
    fp_save = './data/primme_sz(%s)_ng(%d)_nsteps(%d)_freq(1)_kt%s'%(sz_str,ngrain,nsteps,append_name)
    
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
        
        
    #assumes 2d for now too, just update the "c"s later #!!!
    with h5py.File(fp_save, 'a') as f:
        
        # If file already exists, create another group in the file for this simulaiton
        num_groups = len(f.keys())
        hp_save = 'sim%d'%num_groups
        g = f.create_group(hp_save)
        
        # Save data
        s = list(im.shape); s[0] = nsteps
        dset = g.create_dataset("ims_id", shape=s, dtype=dtype)
        dset2 = g.create_dataset("euler_angles", shape=ea.shape)
        dset3 = g.create_dataset("miso_array", shape=miso_array.shape)
        dset4 = g.create_dataset("miso_matrix", shape=miso_matrix.shape)
        dset2[:] = ea
        dset3[:] = miso_array #radians (does not save the exact "Miso.txt" file values, which are degrees divided by the cutoff angle)
        dset4[:] = miso_matrix.cpu() #same values as mis0_array, different format
        
        batch_dims = [np.clip(b, 0, int(2*size[i]/3)) for i, b in enumerate(batch_dims)]
        aaa = torch.Tensor(size/np.array(batch_dims)).long()+1
        bbb = torch.stack(torch.meshgrid([torch.arange(aa) for aa in aaa])).reshape(2,-1).T.numpy()
        n=8+3
        for i in tqdm(range(nsteps), 'Running PRIMME simulation: '):
            
            im_next = im.clone()
            
            for j in bbb:
                
                mi = j*np.array(batch_dims) - n
                ma = (np.clip((j+1)*np.array(batch_dims), np.zeros(2), size) + n)%size
                slices_txt = ':,:,%d:%d,%d:%d'%(mi[0],ma[0],mi[1],ma[1])
                batch = fs.wrap_slice(im, slices_txt)
                
                batch_p = agent.step(batch.clone(), miso_matrix, evaluate=False)
                
                strs = [str(int(mi[0]+n)),str(int(ma[0]-n)),str(int(mi[1]+n)),str(int(ma[1]-n))]
                for k in range(len(strs)): 
                    if strs[k]=='0': strs[k]=''
                exec('im_next[:,:,%s:%s,%s:%s]=batch_p'%tuple(strs))
            
            im = im_next
            
            #Store
            dset[i,:] = im[0].cpu()
            
            #Plot
            if plot_freq is not None: 
                if i%plot_freq==0:
                    if dims==2: 
                        plt.imshow(im[0,0,].cpu()); plt.show()
            
            
            
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

    return fp_save