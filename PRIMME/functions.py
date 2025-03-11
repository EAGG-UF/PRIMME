#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IF THIS CODE IS USED FOR A RESEARCH PUBLICATION, please cite:
    Yan, W., Melville, J., Yadav, V., Everett, K., Yang, L., Kesler, M. S., ... & Harley, J. B. (2022). A novel physics-regularized interpretable machine learning model for grain growth. Materials & Design, 222, 111032.
"""

### Import functions

import torch
import numpy as np
import os
from tqdm import tqdm
import h5py
import torch.nn.functional as F
import imageio
import matplotlib.pyplot as plt
from unfoldNd import unfoldNd 
#from PRIMME import PRIMME
import matplotlib.colors as mcolors
import pickle
from pathlib import Path
### Script

fp = './data/'
if not os.path.exists(fp): os.makedirs(fp)

fp = './plots/'
if not os.path.exists(fp): os.makedirs(fp)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

### General

def check_exist(fps):
    for fp in fps:
        if not os.path.exists(fp):
            raise Exception('File does not exist: %s'% fp)
            

def check_exist_h5(hps, gps, dts, if_bool=False):
    #Raises an exception if something doesn't exist
    
    for hp in hps:
        if not os.path.exists(hp):
            if if_bool: return False
            else: raise Exception('File does not exist: %s'%hp)
    
    for i in range(len(hps)):
        with h5py.File(hps[i], 'r') as f:
            print(f.keys())
            print(f['sim0'].keys())
            if not gps[i] in f.keys():
                if if_bool: return False
                else: raise Exception('Group does not exist: %s/%s'%(hps[i], gps[i]))
            
            for d in dts:
                if not d in f[gps[i]].keys():
                    if if_bool: return False
                    else: raise Exception('Dataset does not exist: %s/%s/%s'%(hps[i], gps[i], d))
                    
    if if_bool: return True

def save_picke_files(save_dir, filename_save, dataset):

    save_dir.mkdir(parents = True, exist_ok = True)
    path_save = save_dir.joinpath(filename_save)
    
    with open(path_save, 'wb') as handle:
        pickle.dump(dataset, handle, protocol = pickle.HIGHEST_PROTOCOL)        
    print(path_save, "has been created\n")
    
def load_picke_files(load_dir, filename_save):

    path_load = load_dir.joinpath(filename_save)

    if os.path.isfile(str(path_load)):  
        print(path_load, "start to be loaded\n")
        with open(path_load , 'rb') as handle:
            dataset = pickle.load(handle)         
        print(path_load, "has been created\n") 
    else:
        print("please create " + str(path_load)) 
    
    return dataset

### Create initial conditions

def generate_random_grain_centers(size=[128, 64, 32], ngrain=512):
    grain_centers = torch.rand(ngrain, len(size))*torch.Tensor(size)
    return grain_centers


def generate_circleIC(size=[512,512], r=64):
    c = (torch.Tensor(size)-1)/2
    a1 = torch.arange(size[0]).unsqueeze(1).repeat(1, size[1])
    a2 = torch.arange(size[1]).unsqueeze(0).repeat(size[0], 1)
    img = (torch.sqrt((c[0]-a1)**2+(c[1]-a2)**2)<r).float()
    euler_angles = np.pi*torch.rand((2,3))*torch.Tensor([2,0.5,2])
    return img.numpy(), euler_angles.numpy()


def generate_sphereIC(size=[512,512,512], r=200):
    c = (torch.Tensor(size)-1)/2
    a1 = torch.arange(size[0])[:,None,None].repeat(1, size[1], size[2])
    a2 = torch.arange(size[1])[None,:,None].repeat(size[0], 1, size[2])
    a3 = torch.arange(size[1])[None,None,:].repeat(size[0], size[1], 1)
    img = (torch.sqrt((c[0]-a1)**2+(c[1]-a2)**2+(c[2]-a3)**2)<r).float()
    euler_angles = np.pi*torch.rand((2,3))*torch.Tensor([2,0.5,2])
    return img.numpy(), euler_angles.numpy()

def generate_SquareIC(size=[512,512], r=64):
    c = (torch.Tensor(size)-1)/2
    a1 = torch.arange(size[0]).unsqueeze(1).repeat(1, size[1])
    a2 = torch.arange(size[1]).unsqueeze(0).repeat(size[0], 1)
    img = ((np.abs(c[0]-a1) < r) & (np.abs(c[1]-a2) <r)).float()
    euler_angles = np.pi*torch.rand((2,3))*torch.Tensor([2,0.5,2])
    return img.numpy(), euler_angles.numpy()

def generate_3grainIC(size=[512,512], h=350):
    img = torch.ones(size)
    img[size[0]-h:,int(size[0]/2):] = 0
    img[size[1]-h:,:int(size[1]/2)] = 2
    euler_angles = np.pi*torch.rand((3,3))*torch.Tensor([2,0.5,2])
    return img.numpy(), euler_angles.numpy()


def generate_hex_grain_centers(dim=512, dim_ngrain=8):
    #Generates grain centers that can be used to generate a voronoi tesselation of hexagonal grains
    #"dim" is the dimension of one side length, the other is calculated to fit the same number of grains in that direction
    #"dim_ngrain" is the number of grains along one one dimension, it is the same for both dimensions
    mid_length = dim/dim_ngrain #length between two flat sides of the hexagon
    side_length = mid_length/np.sqrt(3) #side length of hexagon
    size = [int(dim*np.sqrt(3)/2), dim] #image size
    
    r1 = torch.arange(1.5*side_length, size[0], 3*side_length).float() #row coordinates of first column
    r2 = torch.arange(0, size[0], 3*side_length).float() #row coordinates of second column
    c1 = torch.arange(0, size[1], mid_length).float() #column coordinates of first row
    c2 = torch.arange(mid_length/2, size[1], mid_length).float() #column coordinates of second row
    
    centers1 = torch.cartesian_prod(r1, c1) #take all combinations of first row and column coordinates
    centers2 = torch.cartesian_prod(r2, c2) #take all combinations of second row and column coordinates
    grain_centers = torch.cat([centers1,centers2], dim=0)[torch.randperm(dim_ngrain**2)]
    return grain_centers, size


def generate_hexIC():
    grain_centers, size = generate_hex_grain_centers(dim=512, dim_ngrain=8)
    ic, ea, _ = voronoi2image(size=size, ngrain=64, center_coords0=grain_centers)
    return ic, ea


def write_grain_centers_txt(center_coords, fp="grains"):
    #Generate a "grains.txt" of grain center locations for use in MOOSE simulations (https://github.com/idaholab/moose/blob/next/modules/phase_field/test/tests/initial_conditions/grains.txt)
    #The file is written to the current directory and can be used for 2D or 3D "size" inputs
    
    if center_coords.shape[1]==2: header = "x y\n"
    else: header = "x y z\n"

    np.savetxt("%s.txt"%fp, center_coords, delimiter=' ', fmt='%.5f')
    
    with open("%s.txt"%fp, 'r+') as f:
            content = f.read()
            f.seek(0, 0)
            f.write(header + content)

def read_grain_centers_txt(fp="Case2_grains_centers"):
    with open("%s.txt"%fp) as file:
        lines = file.readlines()
        lines = [x.split() for x in lines]
        grain_centers = torch.Tensor(np.array(lines[1:]).astype(float))
    return grain_centers


def voronoi2image(size=[128, 64, 32], ngrain=512, memory_limit=1e9, p=2, center_coords0=None, device=device):          
    
    #SETUP AND EDIT LOCAL VARIABLES
    dim = len(size)
    
    #GENERATE RENDOM GRAIN CENTERS
    # center_coords = torch.cat([torch.randint(0, size[i], (ngrain,1)) for i in range(dim)], dim=1).float().to(device)
    # center_coords0 = torch.cat([torch.randint(0, size[i], (ngrain,1)) for i in range(dim)], dim=1).float()
    if center_coords0==None: center_coords0 = generate_random_grain_centers(size, ngrain)
    else: ngrain = center_coords0.shape[0]
    center_coords = torch.Tensor([])
    for i in range(3): #include all combinations of dimension shifts to calculate periodic distances
        for j in range(3):
            if len(size)==2:
                center_coords = torch.cat([center_coords, center_coords0 + torch.Tensor(size)*(torch.Tensor([i,j])-1)])
            else: 
                for k in range(3):
                    center_coords = torch.cat([center_coords, center_coords0 + torch.Tensor(size)*(torch.Tensor([i,j,k])-1)])
    center_coords = center_coords.float().to(device)
    
    #CALCULATE THE MEMORY NEEDED TO THE LARGE VARIABLES
    mem_center_coords = float(64*dim*center_coords.shape[0])
    mem_cords = 64*torch.prod(torch.Tensor(size))*dim
    mem_dist = 64*torch.prod(torch.Tensor(size))*center_coords.shape[0]
    mem_ids = 64*torch.prod(torch.Tensor(size))
    available_memory = memory_limit - mem_center_coords - mem_ids
    batch_memory = mem_cords + mem_dist
    
    #CALCULATE THE NUMBER OF BATCHES NEEDED TO STAY UNDER THE "memory_limit"
    num_batches = torch.ceil(batch_memory/available_memory).int()
    num_dim_batch = torch.ceil(num_batches**(1/dim)).int() #how many batches per dimension
    dim_batch_size = torch.ceil(torch.Tensor(size)/num_dim_batch).int() #what's the size of each of the batches (per dimension)
    num_dim_batch = torch.ceil(torch.Tensor(size)/dim_batch_size).int() #the actual number of batches per dimension (needed because of rouning error)
    
    if available_memory>0: #if there is avaiable memory
        #CALCULATE THE ID IMAGE
        all_ids = torch.zeros(size).type(torch.int16)           
        ref = [torch.arange(size[i]).int() for i in range(dim)] #aranges of each dimension length
        tmp = tuple([torch.arange(i).int() for i in num_dim_batch]) #make a tuple to iterate with number of batches for dimension
        for itr in tqdm(torch.cartesian_prod(*tmp), 'Finding voronoi: '): #asterisk allows variable number of inputs as a tuple
            
            start = itr*dim_batch_size #sample start for each dimension
            stop = (itr+1)*dim_batch_size #sample end for each dimension
            stop[stop>=torch.Tensor(size)] = torch.Tensor(size)[stop>=torch.Tensor(size)].int() #reset the stop value to the end of the dimension if it went over
            indicies = [ref[i][start[i]:stop[i]] for i in range(dim)] #sample indicies for each dimension
            
            coords = torch.cartesian_prod(*indicies).float().to(device) #coordinates for each pixel
            dist = torch.cdist(center_coords, coords, p=p) #distance between each pixel and the "center_coords" (grain centers)
            ids = (torch.argmin(dist, dim=0).reshape(tuple(stop-start))%ngrain).int() #a batch of the final ID image (use modulo/remainder quotient to account for periodic grain centers)
            
            if dim==2: all_ids[start[0]:stop[0], start[1]:stop[1]] = ids
            else: all_ids[start[0]:stop[0], start[1]:stop[1], start[2]:stop[2]] = ids
            
        total_memory = batch_memory + mem_center_coords + mem_ids #total memory used by this function
        print("Total Memory: %3.3f GB, Batches: %d"%(total_memory/1e9, num_batches))
        
        #GENERATE RANDOM EULER ANGLES FOR EACH ID
        euler_angles = torch.stack([2*np.pi*torch.rand((ngrain)), \
                              0.5*np.pi*torch.rand((ngrain)), \
                              2*np.pi*torch.rand((ngrain))], 1)
            
        return all_ids.cpu().numpy(), euler_angles.cpu().numpy(), center_coords0.numpy()
            
    else: 
        print("Available Memory: %d - Increase memory limit"%available_memory)
        return None, None, None

def generate_train_init(filename, grain_shape, grain_sizes, device, miso_array = None):
    
    if grain_shape == "circular": 
        ic, ea = generate_circleIC(size = grain_sizes[0], r = grain_sizes[1]) #nsteps=200, pad_mode='circular'
    elif grain_shape == "square":
        ic, ea = generate_SquareIC(size = grain_sizes[0], r = grain_sizes[1]) 
    elif grain_shape == "hex":
        ic, ea = generate_hexIC() #nsteps=500, pad_mode='circular'
    elif grain_shape == "grain":
        if grain_sizes[0][0] * grain_sizes[0][0] > 2024 * 2024:
            device = 'cpu'
        ic, ea, _ = voronoi2image(size = grain_sizes[0], ngrain = grain_sizes[1], device = device) #nsteps=500, pad_mode='circular'
    
    if np.all(miso_array==None): miso_array = find_misorientation(ea, mem_max=1) 
    miso_matrix = miso_conversion(torch.from_numpy(miso_array[None,]))[0]    
    
    data_dict = {"ic": ic, "ea": ea, "miso_array": miso_array, "miso_matrix": miso_matrix}
    save_picke_files(save_dir = Path('./data'), filename_save = filename, dataset = data_dict)
    
    return ic, ea, miso_array, miso_matrix 

### Run and read SPPARKS

def image2init(img, EulerAngles, fp=None):
    '''
    Takes an image of grain IDs (and euler angles assigned to each ID) and writes it to an init file for a SPPARKS simulation
    The initial condition file is written to the 2D or 3D file based on the dimension of 'img'
    
    Inputs:
        img (numpy, integers): pixels indicate the grain ID of the grain it belongs to
        EulerAngles (numpy): number of grains by three Euler angles
    '''
    # Set local variables
    size = img.shape
    dim = len(img.shape)
    if fp==None: fp = r"./spparks_simulations/spparks.init"
    IC = [0]*(np.prod(size)+3)
    
    # Write the information in the SPPARKS format and save the file
    IC[0] = '# This line is ignored\n'
    IC[1] = 'Values\n'
    IC[2] = '\n'
    k=0
    
    if dim==3: 
        for i in range(0,size[2]):
            for j in range(0,size[1]):
                for h in range(0,size[0]):
                    SiteID = int(img[h,j,i])
                    IC[k+3] = str(k+1) + ' ' + str(int(SiteID+1)) + ' ' + str(EulerAngles[SiteID,0]) + ' ' + str(EulerAngles[SiteID,1]) + ' ' + str(EulerAngles[SiteID,2]) + '\n'
                    k = k + 1
    
    else:
        for i in range(0,size[1]):
            for j in range(0,size[0]):
                SiteID = int(img[j,i])
                IC[k+3] = str(k+1) + ' ' + str(int(SiteID+1)) + ' ' + str(EulerAngles[SiteID,0]) + ' ' + str(EulerAngles[SiteID,1]) + ' ' + str(EulerAngles[SiteID,2]) + '\n'
                k = k + 1
            
    with open(fp, 'w') as file:
        file.writelines(IC)
        
    # Completion message
    print("NEW IC WRITTEN TO FILE: %s"%fp)


def count_tags(fp = r"./edit_files/agg_poly_edit.in"):
    '''
    Returns and print ths number of tags (##<counting numbers>##) found in a given file.

    Parameters
    ----------
    fp : string, optional
        Path to text file to be read in. The default is r"./output/agg_poly_edit.in".
fs
    Returns
    -------
    num_tags : int
        Number of tags found in the file.

    '''    

    # Read the file into a string
    with open(fp, 'r') as f:
        f_str = f.read()

    # Count the number of tags
    num_tags = 0;
    while 1: 
        if "##%d##"%(num_tags+1) in f_str:
            num_tags += 1
        else: 
            print("There are %d tags in '%s'"%(num_tags, fp))
            return num_tags
        
        
def replace_tags(fp_in = r"./edit_files/spparks_2d.in", 
                      replacement_text = ['45684', '512', '511.5', '511.5', '1', '10', '50', '500', "agg"], 
                      fp_out = r"../SPPARKS/examples/agg/2d_sim/agg_poly.in",
                      print_chars = [0,0]):
    '''
    This function takes the txt file at file_path, replaces markers in the
    text (##<counting numbers>##) with the strings provided in
    replacement_text. (Markers need to be placed in the target txt file by
    the user ahead of time.) 
    
    Variables:
        fp_in (*.txt): path to text file to be read in
        replacement_text (list of strings): text to replace each marker with
        fp_out (*.txt): path to text file to be written to
        print_chars (list of two integers): the first and last character to print in the file
    '''
    
    # Read the file into a string
    with open(fp_in, 'r') as f:
        f_str = f.read()
    
    # Print some lines from the file before substitution
    if sum(print_chars) != 0: print(f_str[int(print_chars[0]):int(print_chars[1])])
    
    # Replace tags with the text replacement_text
    for i, rt in enumerate(replacement_text):
        f_str = f_str.replace("##%d##"%(i+1),rt)
        
    # Print some lines from the file after substitution
    if sum(print_chars) != 0: print(f_str[int(print_chars[0]):int(print_chars[1])])
    
    # Write string to a file
    with open(fp_out, 'w') as f:
        f.write(f_str)
        
    # Completion Message
    print("TAGS REPLACED - CREATED: %s"%fp_out)
    

def calc_MisoEnergy(fp=r"../SPPARKS/examples/agg/2d_sim/"):
    # Caclulates and writes MisoEnergy.txt from Miso.txt in the given file path 'fp'
    with open(fp + "Miso.txt", 'r') as f: f_str = f.read()
    miso = np.asarray(f_str.split('\n')[0:-1], dtype=float)
    
    theta = miso;
    theta = theta*(theta<1)+(theta>1);
    gamma = theta*(1-np.log(theta));
    
    tmp =[]
    for i in gamma: tmp.append('%1.6f'%i); tmp.append('\n')
    with open(fp + "MisoEnergy.txt", 'w') as file: file.writelines(tmp)


def run_spparks(ic, ea, nsteps=500, kt=0.66, cut=25.0, freq=(1,1), rseed=None, miso_array=None, which_sim='agg', bcs=['p','p','p'], save_sim=True, del_sim=False, path_sim=None):
    '''
    **DEPRECATED FOR THIS VERSION OF PRIMME, THE USER MUST FIND A SPPARKS TRAINSET MODEL TO USE**
    Runs one simulation and returns the file path where the simulation was run 
    
    Input:
        rseed: random seed for the simulation (the same rseed and IC will grow the same)
        freq_stat: how many steps between printing stats
        freq_dump: how many steps between recording the structure
        nsteps: number of simulation steps
        dims: square dimension of the structure
        ngrain: number of grains
        which_sim ('agg' or 'eng'): dictates which simulator to use where eng is the latest and allows the use of multiple cores 
    Output:
        path_sim
    '''
    
    # Find a simulation path that doesn't already exist (if not told exactly where to run the simulation)
    if path_sim==None:
        for i in range(100): #assume I will not do more than 100 simulations at a time
            path_sim = r"./spparks_simulation_%d/"%i
            if not os.path.exists(path_sim): break
    
    # Create the simulation folder, or clean it out
    if not os.path.exists(path_sim): 
        os.makedirs(path_sim)
    else:
        os.system(r"rm %s/*"%path_sim) #remove all files in the simulations folder to prepare for a new simulation
    
    # Set and edit local variables
    size = list(ic.shape) + [1,]*(3-len(ic.shape)) #add ones until length is three
    bcs = bcs + ['p',]*(3-len(bcs)) #add 'p' until length is three
    dim = np.sum(np.array(size)!=1) #the number of dimensions larger than 1
    if dim==2: bcs[-1] = 'p' #if 2D, ensure last boundary condition is periodic
    ngrain = len(np.unique(ic))
    num_processors = 1 #does not affect agg, agg can only do 1
    if rseed==None: rseed = np.random.randint(10000) #changes get different growth from the same initial condition
    freq_stat = freq[0]
    freq_dump = freq[1]
    # path_sim = r"./spparks_simulations/"
    path_home = r"../"
    path_edit_in = r"./spparks_files/spparks.in"
    path_edit_sh = r"./spparks_files/spparks.sh"
    
    # Edit size for boundary conditions
    size = size.copy()
    size = (np.array(size)-0.5*(np.array(bcs)=='n')*(np.array(size)!=1)).tolist() #subtract 0.5 from all dimensions that are nonperiodic
    
    # Set lattice parameters
    if dim==2: lat = 'sq/8n'
    else: lat = 'sc/26n'
    
    # Setup simulation file parameters
    if which_sim=='eng': #run agg only once if eng is the simulator we are using (to calculate Miso.txt and Energy.txt files)
        replacement_text_agg_in = [str(rseed), str(ngrain), str(size[0]), str(size[1]), str(size[2]), str(freq_stat), str(freq_dump), str(0), 'agg', str(float(kt)), str(float(cut)), *bcs, str(dim), lat]
    else: 
        replacement_text_agg_in = [str(rseed), str(ngrain), str(size[0]), str(size[1]), str(size[2]), str(freq_stat), str(freq_dump), str(nsteps), 'agg', str(float(kt)), str(float(cut)), *bcs, str(dim), lat]
    replacement_text_agg_sh = [str(1), 'agg']
    replacement_text_eng_in = [str(rseed), str(ngrain), str(size[0]), str(size[1]), str(size[2]), str(freq_stat), str(freq_dump), str(nsteps), 'eng', str(float(kt)), str(float(cut)), *bcs, str(dim), lat]
    replacement_text_eng_sh = [str(num_processors), 'eng']
    
    # Write simulation files ('spparks.init', 'spparks.in', 'spparks.sh', 'Miso.txt')
    image2init(ic, ea, r"%s/spparks.init"%path_sim) #write initial condition
    if np.all(miso_array==None): miso_array = find_misorientation(ea, mem_max=1) 
    np.savetxt('%s/Miso.txt'%path_sim, miso_array/np.pi*180/cut) #convert to degrees and divide by cutoff angle
    print('MISO WRITTEN TO FILE: %s/Miso.txt'%path_sim)
    replace_tags(path_edit_in, replacement_text_agg_in, path_sim + "agg.in")
    replace_tags(path_edit_sh, replacement_text_agg_sh, path_sim + "agg.sh")
    replace_tags(path_edit_in, replacement_text_eng_in, path_sim + "eng.in")
    replace_tags(path_edit_sh, replacement_text_eng_sh, path_sim + "eng.sh")
    
    # Run simulation
    print("\nRUNNING SIMULATION \n")
    os.chdir(path_sim)
    os.system('chmod +x agg.sh')
    os.system('chmod +x eng.sh')
    os.system('./agg.sh')
    if which_sim=='eng': 
        calc_MisoEnergy(r"./")
        os.system('./eng.sh')
    os.chdir(path_home)
    print("\nSIMULATION COMPLETE \nSIMULATION PATH: %s\n"%path_sim)
    
    # Save Simulation
    if save_sim==True:
        
        # Create miso_matrix
        miso_matrix = miso_conversion(torch.from_numpy(miso_array[None,]))[0].numpy()
        
        # Read dump
        fp_save = './data/spparks_sz(%dx%d)_ng(%d)_nsteps(%d)_freq(%d)_kt(%.2f)_cut(%d).h5'%(np.ceil(size[0]),np.ceil(size[1]),ngrain,nsteps,freq[1],kt,cut)
        ims_id, _, ims_energy = process_dump('%s/spparks.dump'%path_sim)
        tmp = np.array([8,16,32], dtype='uint64')
        dtype = 'uint' + str(tmp[np.sum(ngrain>2**tmp)])
        
        with h5py.File(fp_save, 'a') as f:
            
            # If file already exists, create another group in the file for this simulaiton
            num_groups = len(f.keys())
            hp_save = 'sim%d'%num_groups
            g = f.create_group(hp_save)
            
            # Save data
            dset = g.create_dataset("ims_id", shape=ims_id.shape, dtype=dtype)
            dset1 = g.create_dataset("ims_energy", shape=ims_energy.shape)
            dset2 = g.create_dataset("euler_angles", shape=ea.shape)
            dset3 = g.create_dataset("miso_array", shape=miso_array.shape)
            dset4 = g.create_dataset("miso_matrix", shape=miso_matrix.shape)
            dset[:] = ims_id
            dset1[:] = ims_energy
            dset2[:] = ea
            dset3[:] = miso_array #radians (does not save the exact "Miso.txt" file values, which are degrees divided by the cutoff angle)
            dset4[:] = miso_matrix #same values as mis0_array, different format
            
        return ims_id, fp_save
    
    if del_sim: os.system(r"rm -r %s"%path_sim) #remove entire folder
    
    return None, None


def read_dump(path_to_dump='./spparks_simulations/spparks.dump'):
    
    with open(path_to_dump) as file: 
        item_names = []
        item_data = []
        item_index = 0
        data_lines = []
        
        # line = file.readline()
        # while line!=None:
        for line in file.readlines(): 
            if 'ITEM:' in line:
                
                # Stack data for previous item
                if len(data_lines)>0: 
                    data = np.stack(data_lines)
                    item_data[item_index].append(data)
                    if item_index==0: print('Read step: %d'%item_data[0][-1][0,-1])
                    data_lines = []
                    
                # Find item index or add it to the list
                item = line[6:].replace('\n', '')
                if item in item_names:
                    item_index = np.where(np.array(item_names)==item)[0][0]
                else:
                    item_names.append(item)
                    item_data.append([])
                    item_index = len(item_names) - 1
    
            else:
                data_line = np.array(line.split()).astype(float)
                data_lines.append(data_line)
                
            # line = file.readline()
    
        # Stack data for previous item one last time
        if len(data_lines)>0: 
            data = np.stack(data_lines)
            item_data[item_index].append(data)
            if item_index==0: print('Read step: %d'%item_data[0][-1][0,-1])
            data_lines = []
    
    # Stack each item into a single tensor 
    for i in range(len(item_data)):
        item_data[i] = np.stack(item_data[i])
        
    return item_names, item_data


def process_dump(path_to_dump='./spparks_simulations/spparks.dump'):
    
    # Read dump
    item_names, item_data = read_dump(path_to_dump)
    
    # Find simulation dimensions
    dims = np.flip(np.ceil(item_data[2][0,:,-1]).astype(int))
    
    # Arrange ID images
    ims_id = item_data[3][...,1].reshape((-1,)+tuple(dims)).transpose([0,3,2,1]).squeeze()[:,None,]-1
    
    # Arrange energy images
    ims_energy = item_data[3][...,-1].reshape((-1,)+tuple(dims)).transpose([0,3,2,1]).squeeze()[:,None,]
    
    # Find euler angles per ID
    num_grains = int(np.max(item_data[3][0,:,1]))
    euler_angles = np.zeros([num_grains,3])
    for i in range(np.prod(dims)):
        ii = int(item_data[3][0,i,1])-1
        ea = item_data[3][0,i,2:-1]
        euler_angles[ii] = ea
    
    return ims_id, euler_angles, ims_energy


def create_SPPARKS_dataset(size=[257,257], ngrains_rng=[256, 256], kt=0.66, cutoff=25.0, nsets=200, max_steps=100, offset_steps=1, future_steps=4, del_sim=False):
    
    # DEPRECATED FOR THIS VERSION OF PRIMME, THE USER MUST FIND A SPPARKS TRAINSET MODEL TO USE
    # SET SIMULATION PATH
    path_sim = './spparks_simulation_trainset/'
        
    # NAMING CONVENTION   
    fp = './data/trainset_spparks_sz(%dx%d)_ng(%d-%d)_nsets(%d)_future(%d)_max(%d)_kt(%.2f)_cut(%d).h5'%(size[0],size[1],ngrains_rng[0],ngrains_rng[1],nsets,future_steps,max_steps,kt,cutoff)

    # DETERMINE THE SMALLEST POSSIBLE DATA TYPE POSSIBLE
    m = np.max(ngrains_rng)
    tmp = np.array([8,16,32], dtype='uint64')
    dtype = 'uint' + str(tmp[np.sum(m>2**tmp)])
    
    h5_shape = (nsets, future_steps+1, 1) + tuple(size)
    h5_shape2 = (nsets, m, 3)
    h5_shape3 = (nsets, int(m*(m-1)/2))

    with h5py.File(fp, 'w') as f:
        dset = f.create_dataset("ims_id", shape=h5_shape, dtype=dtype)
        dset1 = f.create_dataset("ims_energy", shape=h5_shape)
        dset2 = f.create_dataset("euler_angles", shape=h5_shape2)
        dset3 = f.create_dataset("miso_array", shape=h5_shape3)
        for i in tqdm(range(nsets)):
            
            # SET PARAMETERS
            ngrains = np.random.randint(ngrains_rng[0], ngrains_rng[1]+1) #number of grains
            nsteps = np.random.randint(offset_steps+future_steps, max_steps+1) #SPPARKS steps to run
            freq = (1,1) #how often to report stats on the simulation, how often to dump an image (record)
            rseed = np.random.randint(10000) #change to get different growth from teh same initial condition
            
            # RUN SIMULATION
            im, ea, _ = voronoi2image(size, ngrains) #generate initial condition
            miso_array = find_misorientation(ea, mem_max=1) 
            run_spparks(im, ea, nsteps, kt, cutoff, freq, rseed, miso_array=miso_array, save_sim=False, del_sim=False, path_sim=path_sim)
            grain_ID_images, grain_euler_angles, ims_energy = process_dump('%s/spparks.dump'%path_sim)
            # miso = np.loadtxt('%s/Miso.txt'%path_sim)*cutoff/180*np.pi #convert to radians
            
            # WRITE TO FILE
            dset[i,] = grain_ID_images[-(future_steps+1):,] 
            dset1[i,] = ims_energy[-(future_steps+1):,] 
            dset2[i,:ngrains,] = grain_euler_angles
            dset3[i:int(ngrains*(ngrains-1)/2),] = miso_array
            
    if del_sim: os.system(r"rm -r %s"%path_sim) #remove entire folder
    
    return fp


def init2euler(f_init='Case4.init', num_grains=20000):
    # Extracts euler angles for each grain ID from a SPPARKS ".init" file
    # "f_init" - string of the location of the ".init" file
    
    with open(f_init) as file: 
        f_lines = file.readlines()[3:]
    
    ea = np.zeros([1, num_grains, 3])
    for l in f_lines: 
        tmp = np.array(l.replace('\n', '').split(' ')).astype(float)
        ea[:,int(tmp[1]-1),:] = tmp[2:]
    
    if np.sum(ea==0)>0: 
        print("Some euler angles are zero in value. Something might have gone wrong.")
    
    return ea


def append_h5(fp, hp, var_names, var_list):
    with h5py.File(fp, 'a') as f:
        for i in range(len(var_names)):
            f[hp + '/' + var_names[i]] = var_list[i]
            

def extract_spparks_logfile_energy(logfile_path="32c20000grs2400stskT050_cut25.logfile"):
    #From Lin
    delta = 0
    step = 0
    start_point = 0
    with open(logfile_path) as f:
        for i, line in enumerate(f):
            if len(line.split()) > 1:
                 if (line.split())[0] == "stats": delta = float(line.split()[1])
                 if (line.split())[0] == "run": step = float(line.split()[1])
            if line == '      Time    Naccept    Nreject    Nsweeps        CPU     Energy\n':
                start_point = i + 1
    
    timestep = np.zeros(int(step / delta) + 1)
    energy = np.zeros(int(step / delta) + 1)
    end_point = start_point + int(step / delta) + 1
    with open(logfile_path) as f:
        for i, line in enumerate(f):
            if i >= start_point and i < end_point:
                timestep[i - start_point] = int(line.split()[0])
                energy[i-start_point] = float(line.split()[5])
            
    return energy

### Find misorientations
#Code from Lin, optimized by Joseph Melville

def symquat(index, Osym):
    """Convert one(index) symmetric matrix into a quaternion """

    q = np.zeros(4)

    if Osym==24:
        SYM = np.array([[ 1, 0, 0,  0, 1, 0,  0, 0, 1],
                        [ 1, 0, 0,  0,-1, 0,  0, 0,-1],
                        [ 1, 0, 0,  0, 0,-1,  0, 1, 0],
                        [ 1, 0, 0,  0, 0, 1,  0,-1, 0],
                        [-1, 0, 0,  0, 1, 0,  0, 0,-1],
                        [-1, 0, 0,  0,-1, 0,  0, 0, 1],
                        [-1, 0, 0,  0, 0,-1,  0,-1, 0],
                        [-1, 0, 0,  0, 0, 1,  0, 1, 0],
                        [ 0, 1, 0, -1, 0, 0,  0, 0, 1],
                        [ 0, 1, 0,  0, 0,-1, -1, 0, 0],
                        [ 0, 1, 0,  1, 0, 0,  0, 0,-1],
                        [ 0, 1, 0,  0, 0, 1,  1, 0, 0],
                        [ 0,-1, 0,  1, 0, 0,  0, 0, 1],
                        [ 0,-1, 0,  0, 0,-1,  1, 0, 0],
                        [ 0,-1, 0, -1, 0, 0,  0, 0,-1],
                        [ 0,-1, 0,  0, 0, 1, -1, 0, 0],
                        [ 0, 0, 1,  0, 1, 0, -1, 0, 0],
                        [ 0, 0, 1,  1, 0, 0,  0, 1, 0],
                        [ 0, 0, 1,  0,-1, 0,  1, 0, 0],
                        [ 0, 0, 1, -1, 0, 0,  0,-1, 0],
                        [ 0, 0,-1,  0, 1, 0,  1, 0, 0],
                        [ 0, 0,-1, -1, 0, 0,  0, 1, 0],
                        [ 0, 0,-1,  0,-1, 0, -1, 0, 0],
                        [ 0, 0,-1,  1, 0, 0,  0,-1, 0] ])
    elif Osym==12:
        a = np.sqrt(3)/2
        SYM = np.array([ [   1,  0, 0,  0,   1, 0,  0, 0,  1],
                         [-0.5,  a, 0, -a,-0.5, 0,  0, 0,  1],
                         [-0.5, -a, 0,  a,-0.5, 0,  0, 0,  1],
                         [ 0.5,  a, 0, -a, 0.5, 0,  0, 0,  1],
                         [  -1,  0, 0,  0,  -1, 0,  0, 0,  1],
                         [ 0.5, -a, 0,  a, 0.5, 0,  0, 0,  1],
                         [-0.5, -a, 0, -a, 0.5, 0,  0, 0, -1],
                         [   1,  0, 0,  0,  -1, 0,  0, 0, -1],
                         [-0.5,  a, 0,  a, 0.5, 0,  0, 0, -1],
                         [ 0.5,  a, 0,  a,-0.5, 0,  0, 0, -1],
                         [  -1,  0, 0,  0,   1, 0,  0, 0, -1],
                         [ 0.5, -a, 0, -a,-0.5, 0,  0, 0, -1] ])

    if (1+SYM[index,0]+SYM[index,4]+SYM[index,8]) > 0:
        q4 = np.sqrt(1+SYM[index,0]+SYM[index,4]+SYM[index,8])/2
        q[0] = q4
        q[1] = (SYM[index,7]-SYM[index,5])/(4*q4)
        q[2] = (SYM[index,2]-SYM[index,6])/(4*q4)
        q[3] = (SYM[index,3]-SYM[index,1])/(4*q4)
    elif (1+SYM[index,0]-SYM[index,4]-SYM[index,8]) > 0:
        q4 = np.sqrt(1+SYM[index,0]-SYM[index,4]-SYM[index,8])/2
        q[0] = (SYM[index,7]-SYM[index,5])/(4*q4)
        q[1] = q4
        q[2] = (SYM[index,3]+SYM[index,1])/(4*q4)
        q[3] = (SYM[index,2]+SYM[index,6])/(4*q4)
    elif (1-SYM[index,0]+SYM[index,4]-SYM[index,8]) > 0:
        q4 = np.sqrt(1-SYM[index,0]+SYM[index,4]-SYM[index,8])/2
        q[0] = (SYM[index,2]-SYM[index,6])/(4*q4)
        q[1] = (SYM[index,3]+SYM[index,1])/(4*q4)
        q[2] = q4
        q[3] = (SYM[index,7]+SYM[index,5])/(4*q4)
    elif (1-SYM[index,0]-SYM[index,4]+SYM[index,8]) > 0:
        q4 = np.sqrt(1-SYM[index,0]-SYM[index,4]+SYM[index,8])/2
        q[0] = (SYM[index,3]-SYM[index,1])/(4*q4)
        q[1] = (SYM[index,2]+SYM[index,6])/(4*q4)
        q[2] = (SYM[index,7]+SYM[index,5])/(4*q4)
        q[3] = q4

    return q


def get_line(i,j):
    """Get the row order of grain i and grain j in MisoEnergy.txt (i < j)"""
    
    assert type(i)!=torch.Tensor #because torch introduces rounding errors somehow
    num_equal = np.sum(i==j)
    assert num_equal==0
    
    ii = np.where(i>j)[0] #find locations of all indicies that don't follow the i<j rule
    i_tmp = np.copy(i)
    i[ii] = np.copy(j[ii])
    j[ii] = np.copy(i_tmp[ii])
    
    return i+(j-1)*((j)/2)


def symetric_quaternions(Osym=24):
    # Find the quaterians that are all symetritcally equivalent given Osym (24 for cubic or 12 for haxagonal?)
    symm2quat_matrix = np.zeros((Osym,4))
    for i in range(0,Osym):
        symm2quat_matrix[i,:] = symquat(i, Osym)
    
    return symm2quat_matrix


def euler2quaternion(euler_angles):
    """Convert euler angles into quaternions"""
    # 'euler_angles' - numpy array of shape=(number of grains, 3)
    
    yaw = euler_angles[:,0]
    pitch = euler_angles[:,1]
    roll = euler_angles[:,2]

    qx = torch.cos(pitch/2.)*torch.cos((yaw+roll)/2.)
    qy = torch.sin(pitch/2.)*torch.cos((yaw-roll)/2.)
    qz = torch.sin(pitch/2.)*torch.sin((yaw-roll)/2.)
    qw = torch.cos(pitch/2.)*torch.sin((yaw+roll)/2.)

    return torch.stack([qx, qy, qz, qw]).T


def quat_Multi(q1, q2):
    """Return the product of two quaternion"""

    tmp = []
    tmp.append(q1[...,0]*q2[...,0] - q1[...,1]*q2[...,1] - q1[...,2]*q2[...,2] - q1[...,3]*q2[...,3])
    tmp.append(q1[...,0]*q2[...,1] + q1[...,1]*q2[...,0] + q1[...,2]*q2[...,3] - q1[...,3]*q2[...,2])
    tmp.append(q1[...,0]*q2[...,2] - q1[...,1]*q2[...,3] + q1[...,2]*q2[...,0] + q1[...,3]*q2[...,1])
    tmp.append(q1[...,0]*q2[...,3] + q1[...,1]*q2[...,2] - q1[...,2]*q2[...,1] + q1[...,3]*q2[...,0])

    return torch.stack(tmp).transpose(2,0)


def find_misorientation(angles, mem_max=1, if_quat=False, device=device):
    # 'angles' - numpy, shape=(number of grains, 3), euler (yaw, pitch, roll) is default, quaternion is "if_quat==True"
    # 'mem_max' - total memory that can be used by the function in GB
    
    angles = torch.from_numpy(angles).to(device)
    num_grains = angles.shape[0]
    
    # Create and expand symmetry quaternions   (assumes cubic symmetry)
    sym = torch.from_numpy(symetric_quaternions()).to(device)
    tmp = torch.arange(24)
    i0, j0 = list(torch.cartesian_prod(tmp,tmp).T) #pair indicies for expanding the symmetry orientations
    symi = sym[i0,:].unsqueeze(0) 
    symj = sym[j0,:].unsqueeze(0)
    
    # Convert grain euler angles to quaternions, then expand in chunks for processing
    if if_quat: q = angles #if quaternions given
    else: q = euler2quaternion(angles) #if euler angles given
    tmp = torch.arange(num_grains)
    i1, j1 = list(torch.cartesian_prod(tmp,tmp).T) #pair indicies for expanding the grain orientations
    
    # Rearrange indicies to match SPPARKS 'Miso.txt' format
    i_tmp = i1[i1<j1]
    j_tmp = j1[i1<j1]
    i1 = i_tmp
    j1 = j_tmp

    ii = torch.from_numpy(get_line(i1.numpy(),j1.numpy())).long()#.to(device)

    i_tmp = i1.clone()
    i_tmp[ii] = i1
    i1 = i_tmp

    j_tmp = j1.clone()
    j_tmp[ii] = j1
    j1 = j_tmp
    
    # Break quaternion expansion into chunks limited by memory
    mem_per_indx = 24**2*4*64/1e9 #GB per index
    size_chunks = int(mem_max/mem_per_indx)
    num_chunks = int(np.ceil(len(i1)/size_chunks))
    tmp = torch.arange(len(i1))
    i_chunks = torch.chunk(tmp, num_chunks, dim=0)
    
    # Find angle/axis values for each misorientation
    angles = []
    # axis = []
    for ii in tqdm(i_chunks, "Finding misorientations"): 
        
        # Find the chunked indices for expansion
        i2 = i1[ii] 
        j2 = j1[ii]
        qi = q[i2,:].unsqueeze(1)
        qj = q[j2,:].unsqueeze(1)
        
        # Multiply all pairs of symmetry orientations with all pairs of grain orientations (in this chunk)
        q1 = quat_Multi(symi, qi)
        q2 = quat_Multi(symj, qj)
        
        # Find the rotations between all pairs of orientations
        q2[...,1:] = -q2[...,1:]
        qq = quat_Multi(q1, q2).transpose(0,1)
        
        # Find the roation that gives the minimum angle
        angle0 = 2*torch.acos(qq[...,0])
        angle0[angle0>np.pi] = torch.abs(angle0[angle0>np.pi] - 2*np.pi)
        angle_tmp = torch.min(angle0, axis=0)[0]
        angles.append(angle_tmp.cpu().numpy())

    return np.hstack(angles) #misorientation is the angle, radians


### Statistical functions
#Written by Kristien Everett, code optimized and added to by Joseph Melville 

def pad_mixed(ims, pad, pad_mode="circular"):
    #Allows for padding of "ims" with different padding modes per dimension
    #ims: shape = (num images, num channels, dim1, dim2, dim3(optional))
    #pad: e.g. pad = (1, 1, 2, 2) - pad last dim by (1, 1) and 2nd to last by (2, 2)
    #pad_mode: Same pad modes as "F.pad", but can be a list to pad each dimension differently
    #e.g. pad_mode = ["circular", "reflect"] - periodic boundary condition on last dimension, Neumann (zero flux) on 2nd to last
    
    if type(pad_mode)==list: 
        dims = len(ims.shape)-2
        pad_mode = pad_mode + [pad_mode[-1]]*(dims-len(pad_mode)) #copy last dimension if needed
        ims_padded = ims
        for i in range(dims):
            pad_1d = (0,)*i*2 + pad[i*2:(i+1)*2] + (0,)*(dims-i-1)*2
            ims_padded = F.pad(ims_padded.float(), pad_1d, pad_mode[i])
    else:
        ims_padded = F.pad(ims.float(), pad, pad_mode) #if "pad_mode"!=list, pad dimensions simultaneously
    return ims_padded


def find_grain_areas(im, max_id=19999): 
    #"im" is a torch.Tensor grain id image of shape=(1,1,dim1,dim2) (only one image at a time)
    #'max_id' defines which grain id neighbors should be returned -> range(0,max_id+1)
    #Outputs are of length 'max_id'+1 where each element corresponds to the respective grain id
    
    search_ids = torch.arange(max_id+1).to(im.device) #these are the ids being serach, should include every id possibly in the image
    im2 = torch.hstack([im.flatten(), search_ids]) #ensures the torch.unique results has a count for every id
    areas = torch.unique(im2, return_counts=True)[1]-1 #minus 1 to counteract the above concatenation
    return areas


def find_grain_num_neighbors(im, max_id=19999, if_AW=False):
    #"im" is a torch.Tensor grain id image of shape=(1,1,dim1,dim2) (only one image at a time)
    #'max_id' defines which grain id neighbors should be returned -> range(0,max_id+1)
    #Outputs are of length 'max_id'+1 where each element corresponds to the respective grain id
    
    #Pad the images to define how pairs are made along the edges
    im_pad = pad_mixed(im, [1,1,1,1], pad_mode="circular")
    
    #Find all the unique id nieghbors pairs in the image
    pl = torch.stack([im[0,0,].flatten(), im_pad[0,0,1:-1,0:-2].flatten()]) #left pairs
    pr = torch.stack([im[0,0,].flatten(), im_pad[0,0,1:-1,2:].flatten()]) #right pairs
    pu = torch.stack([im[0,0,].flatten(), im_pad[0,0,0:-2,1:-1].flatten()]) #up pairs
    pd = torch.stack([im[0,0,].flatten(), im_pad[0,0,2:,1:-1].flatten()]) #down pairs
    pairs = torch.hstack([pl,pr,pu,pd]) #list of all possible four neighbor pixel pairs in the image
    pairs_sort, _ = torch.sort(pairs, dim=0) #makes pair order not matter
    pairs_unique = torch.unique(pairs_sort, dim=1) #these pairs define every grain boundary uniquely (plus self pairs like [0,0]
    pairs_unique = pairs_unique[:, pairs_unique[0,:]!=pairs_unique[1,:]] #remove self pairs
    
    #Find how many pairs are associated with each grain id
    search_ids = torch.arange(max_id+1).to(im.device) #these are the ids being serach, should include every id possibly in the image
    pairs_unique2 = torch.hstack([pairs_unique.flatten(), search_ids]) #ensures the torch.unique results has a count for every id
    num_neighbors = torch.unique(pairs_unique2, return_counts=True)[1]-1 #minus 1 to counteract the above concatenation
    
    if if_AW==True:
        l = []
        for ids in tqdm(range(max_id+1)):
            if num_neighbors[ids]==0: l.append(0)
            else:
                i = (torch.sum(pairs_unique[:,torch.sum(pairs_unique==ids, dim=0)==1], dim=0)-ids).long()
                l.append(torch.mean(num_neighbors[i].float()))
        AW = torch.Tensor(l).to(im.device)
        return num_neighbors, AW
    else: 
        return num_neighbors


def my_unfoldNd(ims, kernel_size=3, pad_mode='circular'):
    #Pads "ims" before unfolding
    dims = len(ims.shape)-2
    if type(kernel_size)!=list: kernel_size = [kernel_size] #convert to "list" if it isn't
    kernel_size = kernel_size + [kernel_size[-1]]*(dims-len(kernel_size)) #copy last dimension if needed
    pad = tuple((torch.Tensor(kernel_size).repeat_interleave(2)/2).int().numpy()) #calculate padding needed based on kernel_size
    if pad_mode!=None: ims = pad_mixed(ims, pad, pad_mode) #pad "ims" to maintain dimensions after unfolding
    ims_unfold = unfoldNd(ims, kernel_size=kernel_size) #shape = [N, product(kernel_size), dim1*dim2*dim3]
    return ims_unfold


def miso_conversion(miso_arrays):
    #'miso_arrays' - torch, shapr=(num_ims, num_miso_elements)
    # Index 0 of miso_arrays refers to the smallest grain ID found in the initial condition of the sequence of images (which is 1 for SPPARKS)
    
    num_lines = miso_arrays.shape[1] #number of lines in miso array
    num_grains = int((1+np.sqrt(1+4*2*num_lines))/2) #number of grains
    
    # Find indices for turning a misorientation array to a misorientation matrix
    tmp = torch.arange(num_grains)
    tmp = torch.cartesian_prod(tmp, tmp)
    i = tmp[:,0]
    j = tmp[:,1]
    del tmp
    
    b1 = torch.where(i>j)[0] 
    b2 = torch.where(i==j)[0]
    
    tmp = i[b1] #ensure i<j
    i[b1] = j[b1]
    j[b1] = tmp
    k = (i+(j-1)*(j)/2).long() #calculate indices
    k[b2] = num_lines #wherever i==j, replace with the biggest index possible plus 1
    del i, j, b1, b2, tmp
    
    tmp = torch.zeros([miso_arrays.shape[0], 1]).to(miso_arrays.device) 
    miso_arrays = torch.hstack([miso_arrays, tmp]) #add zero to end for 
    del tmp
    
    miso_matrices = miso_arrays[:,k].reshape(-1, num_grains, num_grains)
    
    return miso_matrices


def mean_wo_zeros(a):
    return torch.sum(a)/torch.sum(a!=0)


def iterate_function(array, func, args=[], device=device):
    
    #Iterate through the first dimension in "array" and apply "func" using "args"
    log = []
    for i in tqdm(range(array.shape[0]), 'In progress: %s'%func.__name__):
        im = torch.from_numpy(array[i:i+1,][:].astype('float')).to(device)
        tmp = func(im, *args).cpu().numpy()
        log.append(tmp)
    return np.stack(log)

def compute_grain_stats(hps, gps='sim0', device=device):
    
    #Make 'hps' and 'gps' a list if it isn't already
    if type(hps)!=list: hps = [hps]
    if type(gps)!=list: gps = [gps]
    
    #Make sure the files needed actually exist
    dts = ['ims_id', 'euler_angles', 'miso_matrix']
    check_exist_h5(hps, gps, dts, if_bool=False)
    
    for i in range(len(hps)):
        hp = hps[i]
        gp = gps[i]
        print('Calculating statistics for: %s/%s'%(hp,gp))
            
        with h5py.File(hp, 'a') as f:
            
            # Setup
            g = f[gp]
            d = g['ims_id']
            max_id = g['euler_angles'].shape[0] - 1
            
            # Find number of pixels per grain
            if 'grain_areas' not in g.keys():
                args = [max_id]
                func = find_grain_areas
                grain_areas = iterate_function(d, func, args)
                g['grain_areas'] = grain_areas
                print('Calculated: grain_areas')
            else: grain_areas = None
                
            # Find average number of pixels per grain
            if 'grain_areas_avg' not in g.keys():
                if np.all(grain_areas==None): grain_areas = g['grain_areas']
                func = mean_wo_zeros
                grain_areas_avg = iterate_function(grain_areas, func, args=[])
                g['grain_areas_avg'] = grain_areas_avg
                print('Calculated: grain_areas_avg')
            
            # Find number of neighbors per grain
            if 'grain_sides' not in g.keys():
                args = [max_id]
                func = find_grain_num_neighbors
                grain_sides = iterate_function(d, func, args)
                g['grain_sides'] = grain_sides
                print('Calculated: grain_sides')
            else: grain_sides = None
                
            # Find average number of neighbors per grain
            if 'grain_sides_avg' not in g.keys():
                if np.all(grain_sides==None): grain_sides = g['grain_sides']
                func = mean_wo_zeros
                grain_sides_avg = iterate_function(grain_sides, func, args=[])
                g['grain_sides_avg'] = grain_sides_avg
                print('Calculated: grain_sides_avg')

def make_videos(hps, ic_shape, sub_folder="", gps='sim0'):
    # Run "compute_grain_stats" before this function
    
    #Make 'hps' and 'gps' a list if it isn't already
    if type(hps)!=list: hps = [hps]
    if type(gps)!=list: gps = [gps]
    
    # Make sure all needed datasets exist
    #dts=['ims_id', 'ims_miso', 'ims_miso_spparks']
    #check_exist_h5(hps, gps, dts)  
    if sub_folder:
        for i in tqdm(range(len(hps)), "Making videos"):
            with h5py.File(hps[i], 'a') as f:
                g = f[gps[i]]
                ims = g['ims_id'][:,0]
                ims = (255/np.max(ims)*ims).astype(np.uint8)
                imageio.mimsave('./plots/%s/%s_ims_id%d.mp4'%(sub_folder, ic_shape, i), ims)
                imageio.mimsave('./plots/%s/%s_ims_id%d.gif'%(sub_folder, ic_shape, i), ims)
    else:
        for i in tqdm(range(len(hps)), "Making videos"):
            with h5py.File(hps[i], 'a') as f:
                g = f[gps[i]]
                ims = g['ims_id'][:,0]
                ims = (255/np.max(ims)*ims).astype(np.uint8)
                imageio.mimsave('./plots/%s_ims_id%d.mp4'%(ic_shape, i), ims)
                imageio.mimsave('./plots/%s_ims_id%d.gif'%(ic_shape, i), ims)

def make_time_plots(hps, ic_shape, sub_folder="", legend = [], gps='last', scale_ngrains_ratio=0.05, cr=None, if_plot=True):
    # Run "compute_grain_stats" before this function
    
    #Make 'hps' and 'gps' a list if it isn't already, and set default 'gps'
    if type(hps)!=list: hps = [hps]
    
    if gps=='last':
        gps = []
        for hp in hps:
            with h5py.File(hp, 'r') as f:
                gps.append(list(f.keys())[-1])
                print(f.keys())
        print('Last groups in each h5 file chosen:')
        #print(gps)        
    else:
        if type(gps)!=list: gps = [gps]
    
    # Establish color table
    c = [mcolors.TABLEAU_COLORS[n] for n in list(mcolors.TABLEAU_COLORS)]
    if np.all(cr!=None): #repeat the the color labels using "cr"
        tmp = []
        for i, e in enumerate(c[:len(cr)]): tmp += cr[i]*[e]
        c = tmp
    
    # Make sure all needed datasets exist
    #dts=['grain_areas', 'grain_sides', 'ims_miso', 'ims_miso_spparks']
    #check_exist_h5(hps, gps, dts)  
    
    # Calculate scale limit
    with h5py.File(hps[0], 'r') as f:
        g = f[gps[0]]
        #print(g.keys())
        total_area = np.prod(g['ims_id'].shape[1:])
        ngrains = g['grain_areas'].shape[1]
        lim = total_area/(ngrains*scale_ngrains_ratio)
    
    # Plot average grain area through time and find linear slopes
    log = []
    ys = []
    ps = []
    rs = []
    for i in tqdm(range(len(hps)),'Calculating avg grain areas'):
        with h5py.File(hps[i], 'r') as f: 
            grain_areas_avg = f[gps[i]+'/grain_areas_avg'][:]
        log.append(grain_areas_avg)
        
        x = np.arange(len(grain_areas_avg))
        p = np.polyfit(x, grain_areas_avg, 1)
        ps.append(p)
        
        fit_line = np.sum(np.array([p[j]*x*(len(p)-j-1) for j in range(len(p))]), axis=0)
        ys.append(fit_line)
        
        r = np.corrcoef(grain_areas_avg, fit_line)[0,1]**2
        rs.append(r)
    
    plt.figure()
    for i in range(len(hps)):
        plt.plot(log[i], c=c[i%len(c)])
        legend.append('Slope: %.3f | R2: %.3f'%(ps[i][0], rs[i]))
    plt.title('Average grain area')
    plt.xlabel('Number of frames')
    plt.ylabel('Average area (pixels)')
    if legend!= []: plt.legend(legend)
    plt.savefig('./plots/%s/%s_avg_grain_area_time'%(sub_folder, ic_shape), dpi=300)
    if if_plot:
        plt.show()
        plt.close()

    # Plot scaled average grain area through time and find linear slopes
    ys = []
    ps = []
    rs = []
    si = []
    xs = []
    for i in range(len(hps)):
        grain_areas_avg = log[i]
        ii = np.argmin(np.abs(grain_areas_avg-lim))
        si.append(ii)
        
        x = np.arange(len(grain_areas_avg))
        p = np.polyfit(x, grain_areas_avg, 1)
        ps.append(p)
        
        fit_line = np.sum(np.array([p[j]*x*(len(p)-j-1) for j in range(len(p))]), axis=0)
        ys.append(fit_line)
        
        r = np.corrcoef(grain_areas_avg, fit_line)[0,1]**2
        rs.append(r)
        
        xx = np.linspace(ngrains,int(ngrains*scale_ngrains_ratio),ii)
        xs.append(xx)
    
    plt.figure()
    for i in range(len(hps)):
        plt.plot(xs[i], log[i][:len(xs[i])], c=c[i%len(c)])
        plt.xlim([np.max(xs[i]), np.min(xs[i])])
        legend.append('Slope: %.3f | R2: %.3f'%(ps[i][0], rs[i]))
    plt.title('Average grain area (scaled)')
    plt.xlabel('Number of grains')
    plt.ylabel('Average area (pixels)')
    if legend!=[]: plt.legend(legend)
    plt.savefig('./plots/%s/%s_avg_grain_area_time_scaled'%(sub_folder, ic_shape), dpi=300)
    if if_plot:
        plt.show()
        plt.close()
    
    # Plot average grain sides through time
    log = []
    for i in tqdm(range(len(hps)),'Plotting avg grain sides'):
        with h5py.File(hps[i], 'r') as f: 
            grain_sides_avg = f[gps[i]+'/grain_sides_avg'][:]
        log.append(grain_sides_avg)
    
    plt.figure()
    for i in range(len(hps)):
        plt.plot(log[i], c=c[i%len(c)])
        legend.append('')
    plt.title('Average number of grain sides')
    plt.xlabel('Number of frames')
    plt.ylabel('Average number of sides')
    if legend!=[]: plt.legend(legend)
    plt.savefig('./plots/%s/%s_avg_grain_sides_time'%(sub_folder, ic_shape), dpi=300)
    if if_plot:
        plt.show()
        plt.close()
        
    # Plot scaled average grain sides through time
    plt.figure()
    for i in range(len(hps)):
        plt.plot(xs[i], log[i][:len(xs[i])], c=c[i%len(c)])
        plt.xlim([np.max(xs[i]), np.min(xs[i])])
        legend.append('')
    plt.title('Average number of grain sides (scaled)')
    plt.xlabel('Number of grains')
    plt.ylabel('Average number of sides')
    if legend!=[]: plt.legend(legend)
    plt.savefig('./plots/%s/%s_avg_grain_sides_time_scaled'%(sub_folder, ic_shape), dpi=300)
    if if_plot:
        plt.show()
        plt.close()
        
    # Plot grain size distribution
    plt.figure()
    frac = 0.25
    for i in tqdm(range(len(hps)),'Calculating normalized radius distribution'):
        with h5py.File(hps[i], 'r') as f: 
            grain_areas = f[gps[i]+'/grain_areas'][:]
        tg = (grain_areas.shape[1])*frac
        ng = (grain_areas!=0).sum(1)
        ii = (ng<tg).argmax()
        ga = grain_areas[ii]
        gr = np.sqrt(ga/np.pi)
        bins=np.linspace(0,3,10)
        gr_dist, _ = np.histogram(gr[gr!=0]/gr[gr!=0].mean(), bins)
        plt.plot(bins[:-1], gr_dist/gr_dist.sum()/bins[1])
    plt.title('Normalized radius distribution (%d%% grains remaining)'%(100*frac))
    plt.xlabel('R/<R>')
    plt.ylabel('Frequency')
    if legend!=[]: plt.legend(legend)
    plt.savefig('./plots/%s/%s_normalized_radius_distribution'%(sub_folder, ic_shape), dpi=300)
    if if_plot:
        plt.show()
        plt.close()
    
    # Plot number of sides distribution
    plt.figure()
    frac = 0.25
    for i in tqdm(range(len(hps)),'Calculating number of sides distribution'):
        with h5py.File(hps[i], 'r') as f: 
            grain_areas = f[gps[i]+'/grain_areas'][:]
            grain_sides = f[gps[i]+'/grain_sides'][:]
        tg = (grain_areas.shape[1])*frac
        ng = (grain_areas!=0).sum(1)
        ii = (ng<tg).argmax()
        gs = grain_sides[ii]
        bins=np.arange(3,9)+0.5
        gs_dist, _ = np.histogram(gs[gs!=0], bins)
        plt.plot(bins[1:]-0.5, gs_dist/gs_dist.sum())
    plt.title('Number of sides distribution (%d%% grains remaining)'%(100*frac))
    plt.xlabel('Number of sides')
    plt.ylabel('Frequency')
    if legend!=[]: plt.legend(legend)
    plt.savefig('./plots/%s/%s_number_sides_distribution'%(sub_folder, ic_shape), dpi=300)
    if if_plot:
        plt.show()
        plt.close()

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def num_diff_neighbors(ims, window_size=3, pad_mode='circular'): 
    #ims - torch.Tensor of shape [# of images, 1, dim1, dim2, dim3(optional)]
    #window_size - the patch around each pixel that constitutes its neighbors
    #May need to add memory management through batches for large tensors in the future
    
    if type(window_size)==int: window_size = [window_size] #convert to "list" if "int" is given
    
    ims_unfold = my_unfoldNd(ims, kernel_size=window_size, pad_mode=pad_mode)
    center_pxl_ind = int(ims_unfold.shape[1]/2)
    ims_diff_unfold = torch.sum(ims_unfold[:,center_pxl_ind,] != ims_unfold.transpose(0,1), dim=0) #shape = [N, dim1*dim2*dim3]
    
    if pad_mode==None: s = ims.shape[:2]+tuple(np.array(ims.shape[2:])-window_size+1)
    else: s = ims.shape
    return ims_diff_unfold.reshape(s) #reshape to orignal image shape


def num_diff_neighbors_inline(ims_unfold): 
    #ims_unfold - torch tensor of shape = [N, product(kernel_size), dim1*dim2] from [N, 1, dim1, dim2] using "torch.nn.Unfold" object
    #Addtiional dimensions to ims_unfold could be included at the end
    center_pxl_ind = int(ims_unfold.shape[1]/2)
    return torch.sum(ims_unfold[:,center_pxl_ind,] != ims_unfold.transpose(0,1), dim=0) #shape = [N, dim1*dim2]


def compute_action_energy_change(im, im_next, energy_dim=3, act_dim=9, pad_mode="circular"):
    #Calculate the energy change introduced by actions in each "im" action window
    #Energy is calculated as the number of different neighbors for each observation window
    #Find the current energy at each site in "im" observational windows
    #Finds the energy of "im_next" using observational windows with center pixels replaced with possible actions
    #The difference is the energy change
    #FUTURE WORK -> If I change how the num-neighbors function works, I could probably use expand instead of repeat
    
    num_dims = len(im.shape)-2
    
    windows_curr_obs = my_unfoldNd(im_next, kernel_size=energy_dim, pad_mode=pad_mode) 
    current_energy = num_diff_neighbors_inline(windows_curr_obs)
    windows_curr_act = my_unfoldNd(im, kernel_size=act_dim, pad_mode=pad_mode)
    windows_next_obs = my_unfoldNd(im_next, kernel_size=energy_dim, pad_mode=pad_mode)
    
    ll = []
    for i in range(windows_curr_act.shape[1]):
        windows_next_obs[:,int(energy_dim**num_dims/2),:] = windows_curr_act[:,i,:]
        ll.append(num_diff_neighbors_inline(windows_next_obs))
    action_energy = torch.cat(ll)[...,None]
    
    energy_change = (current_energy.transpose(0,1)-action_energy)/(energy_dim**num_dims-1)
    
    return energy_change
    

def compute_energy_labels(im_seq, act_dim=9, pad_mode="circular"):
    #Compute the action energy change between the each image and the one immediately following
    #MAYBE CHANGE IT TO THIS IN THE FUTURE -> Compute the action energy change between the first image and all following
    #The total energy label is a decay sum of those action energy changes
    
    # CALCULATE ALL THE ACTION ENERGY CHANGES
    size = im_seq.shape[1:]
    energy_changes = []
    for i in range(im_seq.shape[0]-1):
        ims_curr = im_seq[i].unsqueeze(0)
        ims_next = im_seq[i+1].unsqueeze(0)
        energy_change = compute_action_energy_change(ims_curr, ims_next, act_dim=act_dim, pad_mode=pad_mode)
        energy_changes.append(energy_change)
    
    # COMBINE THEM USING A DECAY SUM
    energy_change = torch.cat(energy_changes, dim=2)
    decay_rate = 1/2
    decay = decay_rate**torch.arange(1,im_seq.shape[0]).reshape(1,1,-1).to(im_seq.device)
    energy_labels = torch.sum(energy_change*decay, dim=2).transpose(0,1).reshape((np.prod(size),)+(act_dim,)*(len(size)-1))
    
    return energy_labels


def compute_action_labels(im_seq, act_dim=9, pad_mode="circular"):
    #Label which actions in each action window were actually taken between the first image and all following
    #The total energy label is a decay sum of those action labels

    size = im_seq.shape[1:]
    im = im_seq[0:1,]
    ims_next = im_seq[1:]
    
    # CALCULATE ACTION LABELS
    window_act = my_unfoldNd(im, kernel_size=act_dim, pad_mode=pad_mode)[0]
    ims_next_flat = ims_next.view(ims_next.shape[0], -1)
    
    actions_marked = window_act.unsqueeze(0).expand(4,-1,-1)==ims_next_flat.unsqueeze(1) #Mark the actions that matches each future image (the "action taken")
    decay_rate = 1/2
    decay = decay_rate**torch.arange(1,im_seq.shape[0]).reshape(-1,1,1).to(im.device)
    action_labels = torch.sum(actions_marked*decay, dim=0).transpose(0,1).reshape((np.prod(size),)+(act_dim,)*(len(size)-1))
    
    return action_labels


def compute_labels(im_seq, obs_dim=9, act_dim=9, reg=1, pad_mode="circular"):
    
    energy_labels = compute_energy_labels(im_seq, act_dim=act_dim, pad_mode=pad_mode)
    action_labels = compute_action_labels(im_seq, act_dim=act_dim, pad_mode=pad_mode)
    labels = action_labels + reg*energy_labels
    
    return labels


def compute_features(im, obs_dim=9, pad_mode='circular'):
    size = im.shape[1:]
    local_energy = num_diff_neighbors(im, window_size=7, pad_mode=pad_mode)
    features = my_unfoldNd(local_energy.float(), obs_dim, pad_mode=pad_mode).T.reshape((np.prod(size),)+(obs_dim,)*(len(size)-1))
    return features




