#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DESCRIPTION:
    These functions are referenced by both the PRIMME and SPPARKS classes
    They are also referenced by "validate_model_script.py" for generating various initial conditions
    In this file are functions that interface with SPPARKS files and commands through the system command line (windows is assumed)
    There are functions used for calculating micostructure features and labels
    There are also functions for custom padding and unfolding of data as well as functions for the statistical analysis of microstructures used for validation
    
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
import math
from tqdm import tqdm
import os
import h5py
import torch
import torch.nn.functional as F
from unfoldNd import unfoldNd 
import pynvml
from scipy.stats import skew, kurtosis



# SETUP NEEDED PATH
if not os.path.exists(r"./spparks_simulations/"): os.makedirs(r"./spparks_simulations/")



### FUNCTIONS


def show_gpus():
    pynvml.nvmlInit()
    count = torch.cuda.device_count()
    for i in range(count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        mem_free = mem_info.free / 1024 ** 3
        device_name = torch.cuda.get_device_name(device=i)
        print("%d: Memory Free - %2.2f \tGB (%s)"%(i, mem_free, device_name))
        #torch.cuda.empty_cache()
        
        
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


def generate_circleIC(size=[512,512], r=64):
    c = (torch.Tensor(size)-1)/2
    a1 = torch.arange(size[0]).unsqueeze(1).repeat(1, size[1])
    a2 = torch.arange(size[1]).unsqueeze(0).repeat(size[0], 1)
    img = (torch.sqrt((c[0]-a1)**2+(c[1]-a2)**2)<r).float()
    euler_angles = math.pi*torch.rand((2,3))*torch.Tensor([2,0.5,2])
    return img.numpy(), euler_angles.numpy()


def generate_3grainIC(size=[512,512], h=350):
    img = torch.ones(512, 512)
    img[size[0]-h:,256:] = 0
    img[size[1]-h:,:256] = 2
    euler_angles = math.pi*torch.rand((3,3))*torch.Tensor([2,0.5,2])
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
    return grain_centers


def generate_random_grain_centers(size=[128, 64, 32], ngrain=512):
    grain_centers = torch.rand(ngrain, len(size))*torch.Tensor(size)
    return grain_centers


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


def voronoi2image(size=[128, 64, 32], ngrain=512, memory_limit=1e9, p=2, center_coords0=None):          
    
    #SETUP AND EDIT LOCAL VARIABLES
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
        for itr in tqdm(torch.cartesian_prod(*tmp)): #asterisk allows variable number of inputs as a tuple
            
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
        euler_angles = torch.stack([2*math.pi*torch.rand((ngrain)), \
                              0.5*math.pi*torch.rand((ngrain)), \
                              2*math.pi*torch.rand((ngrain))], 1)
            
        return all_ids.cpu().numpy(), euler_angles.cpu().numpy(), center_coords0.numpy()
            
    else: 
        print("Available Memory: %d - Increase memory limit"%available_memory)
        return None, None, None
    
    
    
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
    if fp==None: fp = r"./spparks_simulations/PolyIC.init"
    IC = [0]*(np.product(size)+3)
    
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
    

def pad_mixed(ims, pad, pad_mode="reflect"):
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


def my_unfoldNd(ims, kernel_size=3, pad_mode='reflect'):
    #Pads "ims" before unfolding
    dims = len(ims.shape)-2
    if type(kernel_size)!=list: kernel_size = [kernel_size] #convert to "list" if it isn't
    kernel_size = kernel_size + [kernel_size[-1]]*(dims-len(kernel_size)) #copy last dimension if needed
    pad = tuple((torch.Tensor(kernel_size).repeat_interleave(2)/2).int().numpy()) #calculate padding needed based on kernel_size
    ims_padded = pad_mixed(ims, pad, pad_mode) #pad "ims" to maintain dimensions after unfolding
    ims_unfold = unfoldNd(ims_padded, kernel_size=kernel_size) #shape = [N, product(kernel_size), dim1*dim2*dim3]
    return ims_unfold

    
def num_diff_neighbors(ims, window_size=3, pad_mode='reflect'): 
    #ims - torch.Tensor of shape [# of images, 1, dim1, dim2, dim3(optional)]
    #window_size - the patch around each pixel that constitutes its neighbors
    #May need to add memory management through batches for large tensors in the future
    
    if type(window_size)==int: window_size = [window_size] #convert to "list" if "int" is given
    
    ims_unfold = my_unfoldNd(ims, kernel_size=window_size, pad_mode=pad_mode)
    center_pxl_ind = int(ims_unfold.shape[1]/2)
    ims_diff_unfold = torch.sum(ims_unfold[:,center_pxl_ind,] != ims_unfold.transpose(0,1), dim=0) #shape = [N, dim1*dim2*dim3]
    
    # window_size = window_size + [window_size[-1]]*(len(ims.shape)-2-len(window_size)) #copy last dimension of window_size if needed
    # pad = tuple((torch.Tensor(window_size).repeat_interleave(2)/2).int().numpy()) #calculate padding needed based in window_size
    # ims_unfold = unfoldNd(F.pad(ims, pad, pad_mode), kernel_size=window_size) #shape = [N, product(window_size), dim1*dim2*dim3]
    # center_pxl_ind = int(ims_unfold.shape[1]/2)
    # ims_diff_unfold = torch.sum(ims_unfold[:,center_pxl_ind,] != ims_unfold.transpose(0,1), dim=0) #shape = [N, dim1*dim2*dim3]
    return ims_diff_unfold.reshape(ims.shape) #reshape to orignal image shape


def num_diff_neighbors_inline(ims_unfold): 
    #ims_unfold - torch tensor of shape = [N, product(kernel_size), dim1*dim2] from [N, 1, dim1, dim2] using "torch.nn.Unfold" object
    #Addtiional dimensions to ims_unfold could be included at the end
    center_pxl_ind = int(ims_unfold.shape[1]/2)
    return torch.sum(ims_unfold[:,center_pxl_ind,] != ims_unfold.transpose(0,1), dim=0) #shape = [N, dim1*dim2]


def run_spparks(size=[512,512], ngrain=512, nsteps=500, freq_dump=1, freq_stat=1, rseed=45684, which_sim='agg', del_files=False):
    '''
    Runs one simulation and returns the file path where the simulation was run
    
    Input:
        rseed: random seed for the simulation (the same rseed and IC will grow the same)
        freq_stat: how many steps between printing stats
        freq_dump: how many steps between recording the structure
        nsteps: number of simulation steps
        dims: square dimension of the structure
        ngrain: number of grains
        which_sim ('agg' or 'eng'): dictates which simulator to use where eng is the latest and allows the use of multiple cores 
        del_files: if True, deletes Miso.txt and Energy.txt. files and allows agg to calculate new files
    Output:
        path_sim
    '''

    # Set and edit local variables
    num_processors = 1 #does not affect agg, agg can only do 1
    dim = len(size)
    path_sim = r"./spparks_simulations/"
    path_home = r"../"
    path_edit_in = r"./spparks_files/spparks_%sd.in"%str(dim)
    path_edit_sh = r"./spparks_files/spparks.sh"
    
    
    # Setup simulation file parameters
    size = size.copy()
    if len(size)==2: size.append(1)
    size[:dim] = (np.array(size[:dim]) - 0.5).tolist()
    if which_sim=='eng': #run agg only once if eng is the simulator we are using (to calculate Miso.txt and Energy.txt files)
        replacement_text_agg_in = [str(rseed), str(ngrain), str(size[0]), str(size[1]), str(size[2]), str(freq_stat), str(freq_dump), str(0), 'agg']
    else: 
        replacement_text_agg_in = [str(rseed), str(ngrain), str(size[0]), str(size[1]), str(size[2]), str(freq_stat), str(freq_dump), str(nsteps), 'agg']
    replacement_text_agg_sh = [str(1), 'agg']
    replacement_text_eng_in = [str(rseed), str(ngrain), str(size[0]), str(size[1]), str(size[2]), str(freq_stat), str(freq_dump), str(nsteps), 'eng']
    replacement_text_eng_sh = [str(num_processors), 'eng']
    
    # Write simulation files 'spparks.in and spparks.sh files'
    replace_tags(path_edit_in, replacement_text_agg_in, path_sim + "agg.in")
    replace_tags(path_edit_sh, replacement_text_agg_sh, path_sim + "agg.sh")
    replace_tags(path_edit_in, replacement_text_eng_in, path_sim + "eng.in")
    replace_tags(path_edit_sh, replacement_text_eng_sh, path_sim + "eng.sh")
    
    # Clean up some files (if they are there)
    if del_files:
        os.system("rm " + path_sim + "Miso.txt") 
        os.system("rm " + path_sim + "Energy.txt") 
    
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
    
    return path_sim


def extract_spparks_dump(dim=2):
    '''
    Extracts the information from a spparks.dump file containing euler angles (dump  1 text 1.0 ${fileBase}.dump id site d1 d2 d3)
    Placed information in Numpy variables.
    Works for both 2D and 3D dump files
    
    Parameters
    ----------
    dim : int
        relative path to spparks.dump file
    Returns
    -------
    euler_angle_images : numpy array
        dimensions of [number of images, euler angles in 3 channels, dimx, dimy, dimz (0 for 2D)]
    sim_steps : numpy array
        the monte carlo step for the image of the same index in euler_angle_images
    grain_euler_angles : numpy array
        the euler angles for each grain ID
    grain_ID_images: numpy array
    energy_images: numpy array or site energy
    '''

    sim_steps = []
    euler_angle_images = []
    grain_ID_images = []
    energy_images = []
    num_grains = 0
    # path_to_dump = r"../SPPARKS/examples/agg/%sd_sim/spparks.dump"%str(dim)
    
    path_to_dump = r"./spparks_simulations/spparks.dump"
    
    with  open(path_to_dump) as file: 
        print('Loaded')
        for i, l in enumerate(file.readlines()):
            
            t = l.split(' ')
            
            #First time through
            if i == 1:
                sim_step = int(t[-11]) #capture simulation step
                print('Capture sim step: %d'%sim_step)
            elif i == 5: dimx = int(np.ceil(float(t[1]))) #find image dimensions 
            elif i == 6: dimy = int(np.ceil(float(t[1])))
            elif i == 7: 
                dimz = int(np.ceil(float(t[1])))
                num_elements = dimx*dimy*dimz
                image = np.zeros([3, num_elements]) #create image to hold element orientations at this growth step
                ID_image = np.zeros([1, num_elements]) #create image to hold element grain IDs at this growth step
                energy_image = np.zeros([1, num_elements]) #create image to hold element energy at this growth step
                print('Dimensions: [%d, %d, %d]'%(dimx, dimy, dimz))
            
            elif i > 7: 
                [q, r] = np.divmod(i,num_elements+9) #what line are we on in this simulation step
            
                if q==0: #find highest labeled grain
                    if i > 8: 
                        if int(t[1]) > num_grains: num_grains = int(t[1])
                        if i==num_elements+8: 
                            grain_euler_angles = np.zeros([num_grains, 3])
                            print('Number of grains: %d'%num_grains)
                            
                if q==1: #record euler angles for each grain on second pass
                    if r > 8: grain_euler_angles[int(t[1])-1, :] = [float(t[2]), float(t[3]), float(t[4])] 
                
                if r == 0: 
                    image = np.zeros([3, num_elements]) #create image to hold element orientations at this growth step
                    ID_image = np.zeros([1, num_elements]) #create image to hold element grain IDs at this growth step
                    energy_image = np.zeros([1, num_elements]) #create image to hold element energy at this growth step
                elif r == 1:
                    sim_step = int(float(t[-1]))  #capture simulation step
                    print('Capture sim step: %d'%sim_step)
                elif r > 8: 
                    image[:,int(t[0])-1] =  [float(t[2]), float(t[3]), float(t[4])] #record this element's orientation
                    ID_image[:,int(t[0])-1] = [int(t[1])-1] #'-1' to start from 0 instead of 1
                    energy_image[:,int(t[0])-1] = [float(t[5])] #record this element's energy
                
                if r==num_elements+8: #add sim_step and euler_angle_image to the master lists
                    sim_steps.append(sim_step)
                    if dimz==1: 
                        euler_angle_images.append(image.reshape([3, dimy, dimx]).transpose([0,2,1]))
                        grain_ID_images.append(ID_image.reshape([1, dimy, dimx]).transpose([0,2,1]))
                        energy_images.append(energy_image.reshape([1, dimy, dimx]).transpose([0,2,1]))
                    else: 
                        euler_angle_images.append(image.reshape([3, dimz, dimy, dimx]).transpose(0,3,2,1))
                        grain_ID_images.append(ID_image.reshape([1, dimz, dimy, dimx]).transpose(0,3,2,1))
                        energy_images.append(energy_image.reshape([1, dimz, dimy, dimx]).transpose(0,3,2,1))
    
    #Convert to numpy
    sim_steps = np.array(sim_steps)     
    euler_angle_images = np.array(euler_angle_images)  
    grain_ID_images = np.array(grain_ID_images)    
    energy_images = np.array(energy_images)  
    
    return euler_angle_images, sim_steps, grain_euler_angles, grain_ID_images, energy_images


def dump_to_hdf5(path_to_dump="Circle_512by512", path_to_hdf5="Circle_512by512", num_steps=None):
    #A more general purpose extract dump file - reads lines directly to an hdf5 file and saves header names
    #The lines can then be extracted one-by-one from the hdf5 file and converted to an image
    #"num_steps" is a guess at how many entries there are in the dump file to report how long it will take
    
    with open(path_to_dump+".dump") as file:
        bounds = np.zeros([3,2])
        for i, line in enumerate(file): #extract the number of atoms, bounds, and variable names from the first few lines
            if i==3: num_atoms = int(line) 
            if i==5: bounds[0,:] = np.array(line.split(), dtype=float)
            if i==6: bounds[1,:] = np.array(line.split(), dtype=float)
            if i==7: bounds[2,:] = np.array(line.split(), dtype=float)
            if i==8: var_names = line.split()[2:]
            if i>8: break
    bounds = np.ceil(bounds).astype(int) #reformat bounds
    entry_length = num_atoms+9 #there are 9 header lines in each entry
    
    if num_steps!=None: total_lines = num_steps*entry_length
    else: total_lines=None
    
    time_steps = []
    with h5py.File(path_to_hdf5+".hdf5", 'w') as f:
        f["bounds"] = bounds #metadata
        f["variable_names"] = [x.encode() for x in var_names] #metadata
        dset = f.create_dataset("dump_extract", shape=(1,num_atoms,len(var_names)), maxshape=(None,num_atoms,len(var_names)))#, chunks=True)
        with open(path_to_dump+".dump") as file:
            for i, line in tqdm(enumerate(file), "EXTRACTING SPPARKS DUMP (%s.dump)"%path_to_dump, total=total_lines):
                [entry_num, line_num] = np.divmod(i,entry_length) #what entry number and entry line number does this line number indicate
                if line_num==0: entry = np.zeros([num_atoms, len(var_names)]) #reset the entry values at the beginning of each entry
                if line_num==1: time_steps.append(int(line.split()[-1])) #log the time step
                atom_num = line_num-9 #track which atom line we're on
                if atom_num>0 and atom_num<num_atoms: entry[atom_num,] = np.array(line.split(), dtype=float) #record valid atom lines
                if line_num==entry_length-1: 
                    dset[-1,:,:] = entry #save this entry before going to the next
                    dset.resize(dset.shape[0]+1, axis=0) #make more room in the hdf5 dataset
        dset.resize(dset.shape[0]-1, axis=0) #remove the extra room that wasn't used
        time_steps = np.array(time_steps) #reformat time_steps
        f["time_steps"] = time_steps #metadata
        
    return var_names, time_steps, bounds
    

def cumsum_sample(arrays):
    #"array" - shape=(number of arrays, array elements)
    #Chooses an index from each row in "array" by sampling from it's cumsum
    arrays_cumsum = torch.cumsum(arrays, dim=1)/torch.sum(arrays, dim=1).unsqueeze(1)
    sample_values = torch.rand(arrays_cumsum.shape[0]).to(arrays.device)
    sample_indices = torch.argmax((arrays_cumsum>sample_values.unsqueeze(1)).float(), dim=1)
    return sample_indices


def rand_argmax(arrays, dim=1):
    #"array" - shape=(number of arrays, array elements)
    #Chooses an index from each row in "array" that is the max value or a random max value indicie if there are multiples
    arrays_max, _ = torch.max(arrays, dim=dim)
    arrays_marked = arrays==arrays_max.unsqueeze(1)
    samples_indices = cumsum_sample(arrays_marked)
    return samples_indices


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]



### Written by Kristien Everett, code optimized by Joseph Melville 
def grain_size(im, max_id=19999): 
    #"im" is a torch.Tensor grain id image of shape=(1,1,dim1,dim2) (only one image at a time)
    #'max_id' defines which grain id neighbors should be returned -> range(0,max_id+1)
    #Outputs are of length 'max_id'+1 where each element corresponds to the respective grain id
    
    search_ids = torch.arange(max_id+1).to(im.device) #these are the ids being serach, should include every id possibly in the image
    im2 = torch.hstack([im.flatten(), search_ids]) #ensures the torch.unique results has a count for every id
    areas = torch.unique(im2, return_counts=True)[1]-1 #minus 1 to counteract the above concatenation
    sizes = 2*torch.sqrt(areas/np.pi) #assumes grain size equals the diameter of a circular area - i.e. d = 2 * (A/pi)^(1/2)

    return sizes


def grain_num_neighbors(im, max_id=19999, if_AW=False):
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
    
    #Find how many pairs are associated with each grain id
    search_ids = torch.arange(max_id+1).to(im.device) #these are the ids being serach, should include every id possibly in the image
    pairs_unique2 = torch.hstack([pairs_unique.flatten(), search_ids]) #ensures the torch.unique results has a count for every id
    num_neighbors = torch.unique(pairs_unique2, return_counts=True)[1]-3 #minus 1 to counteract the above concatenation and 2 for the self pairs (e.g. [0,0])
    
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


def metric_stats(array):
    #'array' is a 1d numpy array
    
    array = array[array!=0] #remove all zero values
    mn = np.mean(array)
    std = np.std(array)
    skw = skew(array)
    kurt = kurtosis(array, fisher=True)
    stats = np.array([mn, std, skw, kurt])

    return stats


def apply_grain_func(h5_path, func, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    #'h5_path' is the full path from current folder to h5 file (assumes a single dataset called 'images')
    #'func' is a function that inputs a torch.Tensor of shape=(1,1,dim1,dim2) and outputs a constant sized 1d tensor array
    
    l = []
    l_stats = []
    with h5py.File(h5_path, "r") as f:
        num_images = f['images'].shape[0]
        max_id = np.max(f['images'][0])
        for i in tqdm(range(num_images)): #for all the images
            im = torch.from_numpy(f['images'][i].astype(np.int16)).unsqueeze(0).unsqueeze(0).to(device) #convert to tensor of correct shape
            array = func(im, max_id=max_id).cpu().numpy() #rungiven function
            l.append(array) #store results
            l_stats.append(metric_stats(array)) #run and store stats
         
    arrays = np.stack(l)
    array_stats = np.stack(l_stats)
    
    return arrays, array_stats



def create_SPPARKS_dataset(fp, size=[257,257], ngrains_range=[256, 256], nsets=200, future_steps=4, max_steps=100, offset_steps=1):
    
    # DETERMINE THE SMALLEST POSSIBLE DATA TYPE POSSIBLE
    m = np.max(ngrains_range)
    tmp = np.array([8,16,32], dtype='uint64')
    dtype = 'uint' + str(tmp[np.sum(m>2**tmp)])

    with h5py.File(fp, 'w') as f:
        dset = f.create_dataset("dataset", shape=(1, future_steps+1, 1, size[0], size[1]) , maxshape=(None, future_steps+1, 1, size[0], size[1]), dtype=dtype)
        for _ in tqdm(range(nsets)):
            
            # SET PARAMETERS
            ngrains = np.random.randint(ngrains_range[0], ngrains_range[1]+1) #number of grains
            nsteps = np.random.randint(offset_steps+future_steps, max_steps) #SPPARKS steps to run
            freq_dump = 1 #how offten to dump an image (record)
            freq_stat = 1 #how often to report stats on the simulation
            rseed = np.random.randint(10000) #change to get different growth from teh same initial condition
            
            # RUN SIMULATION
            img, EulerAngles, center_coords0 = voronoi2image(size, ngrains) #generate initial condition
            image2init(img, EulerAngles) #write initial condition
            size_tmp = (np.array(size)+np.array([0.5, 0.5])).tolist()
            _ = run_spparks(size_tmp, ngrains, nsteps, freq_dump, freq_stat, rseed) #run simulation
            _, _, _, grain_ID_images, _ = extract_spparks_dump(dim=len(size)) #extract SPPARKS dump data to python
                
            # WRITE TO THE DATASET
            dset[-1,:,:] = grain_ID_images[-(future_steps+1):,] 
            dset.resize(dset.shape[0]+1, axis=0) 
        dset.resize(dset.shape[0]-1, axis=0) #Remove the final dimension that has nothing written to it






