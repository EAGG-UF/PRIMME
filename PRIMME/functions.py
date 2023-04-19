# !/usr/bin/env python3
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
import matplotlib.colors as mcolors
# from uvw import RectilinearGrid, DataArray





### Script

fp = './data/'
if not os.path.exists(fp): os.makedirs(fp)

fp = './plots/'
if not os.path.exists(fp): os.makedirs(fp)

device=torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# device=torch.device("cpu")





### General

def check_exist(fps):
    for fp in fps:
        if not os.path.exists(fp):
            raise Exception('File does not exist: %s'%fp)
            

def check_exist_h5(hps, gps, dts, if_bool=False):
    #Raises an exception if something doesn't exist
    
    for hp in hps:
        if not os.path.exists(hp):
            if if_bool: return False
            else: raise Exception('File does not exist: %s'%hp)
    
    for i in range(len(hps)):
        with h5py.File(hps[i], 'r') as f:
            
            if not gps[i] in f.keys():
                if if_bool: return False
                else: raise Exception('Group does not exist: %s/%s'%(hps[i], gps[i]))
            
            for d in dts:
                if not d in f[gps[i]].keys():
                    if if_bool: return False
                    else: raise Exception('Dataset does not exist: %s/%s/%s'%(hps[i], gps[i], d))
                    
    if if_bool: return True


def my_batch(data, func, batch_sz=100):
    #'data' is broken into a list of "batch_sz" data along dim=0
    #"func" is then applied along dim=1 and concatenated back together
    data_split = data.split(batch_sz, dim=0)
    data_list = [func(d, dim=1) for d in data_split]
    return torch.cat(data_list)  
    
    
def batch_where(data, batch_sz=100):
    #'data' is broken into a list of "batch_sz" data along dim=0
    #"func" is then applied along dim=1 and concatenated back together
    data_split = data.split(batch_sz, dim=0)
    data_list = [torch.where(d) for d in data_split]
    data_list2 = [torch.stack((i+k*batch_sz,j)) for k, (i, j) in enumerate(data_list)]
    return torch.cat(data_list2, 1)


def wrap_slice(data, slices_txt):
    #Slices 'data' using 'slices_txt' (i.e. data[slices_txt])
    #Except - Slices can wrap around boundaries using this function
    #Note - 'data' can be an h5 dataset link
    
    # Setup
    sz = np.array(data.shape)
    
    # Parse 'slices_txt' 
    dims_remove = []
    slices = slices_txt.replace(' ','').split(',')
    for i, s in enumerate(slices): 
        if ':' not in s: #if only int given, keep as is, but keep the dimension until the end
            plus_one = int(s)+1
            if plus_one==0: slices[i] = [s+':'] 
            else: slices[i] = [s+':'+str(plus_one)] 
            dims_remove.append(i)
        else:
            ss = s.split(':')
            for j in range(2): 
                if ss[j]!='': ss[j] = int(ss[j])%sz[i] #wrap all indices
            if '' not in ss and ss[0]>=ss[1]:
                slices[i] = [str(ss[0])+':', ':'+str(ss[1])] #split indices when wraping
            else:
                slices[i] = [str(ss[0])+':'+str(ss[1])] #don't split when not wrapping
    
    # Find number of ranges needed for each dimension
    num_iter = torch.Tensor([len(e) for e in slices]).long() 

    # Create a nested list to hold blocks of data
    log = []
    for t in num_iter.flip(0): log = [log for _ in range(t)]
    log = eval(str(log)) #ensure all the lists are unique objects
    
    # Extract, nest, and then block the data
    i = torch.stack(torch.meshgrid([torch.arange(e) for e in num_iter])).reshape(len(num_iter), -1).T
    for j in i: 
        list_index = ''.join(['[%d]'%jj for jj in j])
        slices_select = tuple([slices[k][j[k]] for k in range(len(slices))])
        slices_select = ''.join(['%s,',]*len(slices))[:-1]%slices_select
        
        if type(data)==torch.Tensor:
            exec('log%s = data[%s].cpu().numpy()'%(list_index, slices_select))
        else:
            exec('log%s = data[%s]'%(list_index, slices_select))
      
    tmp = np.block(log).squeeze(tuple(dims_remove))
    if type(data)==torch.Tensor: 
        return torch.Tensor(tmp).to(data.device)
    else:
        return tmp


def shape_indices(i, sz):
    #'i' can be used to index the flattened Tensor 
    #'i_shaped' can index the same values in the same Tensor of shape 'sz'
    i_shaped = []
    for s in sz.flip(0): 
        i_shaped.append((i%s).long())
        # i = torch.floor_divide(i,s)
        i = torch.div(i, s, rounding_mode='floor')
    i_shaped.reverse()
    return i_shaped


def flatten_indices(indices, sz):
    #'indices' can index the same values in the same Tensor of shape 'sz'
    #'index' can be used to index the flattened Tensor 
    index = indices[-1].long()
    for i in range(1, len(indices)):
        index += (indices[-i-1]*torch.prod(sz[-i:])).long()
    return index


def unfold_in_batches(im, batch_sz, kernel_sz, stride, if_shuffle=False):
    #Create an unfolded view of 'im' (torch.Tensor) 
    #Given 'kernel_sz' (tuple) and 'stride' (tuple)
    #Yield 'batch_sz' (int) portions of the view at a time
    #Shuffles output if 'if_shuffle', but yields each kernal once each
    
    dm = im.dim()
    sz = tuple(im.size())
    sz_new = torch.Tensor(sz)-(torch.Tensor(kernel_sz)-1)
    num_kernels = int(torch.prod(sz_new))
    
    im_unfolded = im.unfold(0,kernel_sz[0],stride[0])
    for i in range(dm-1): im_unfolded = im_unfolded.unfold(i+1,kernel_sz[i+1],stride[i+1])
    
    if if_shuffle is True: i = torch.randperm(num_kernels)
    else: i = torch.arange(num_kernels)
    i_split = torch.split(i,batch_sz)
    
    for j in i_split:
        indices = shape_indices(j, sz_new)
        yield im_unfolded[indices]





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
    if center_coords0 is None: center_coords0 = generate_random_grain_centers(size, ngrain)
    else: 
        center_coords0 = torch.Tensor(center_coords0)
        ngrain = center_coords0.shape[0]
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
    num_dim_batch = torch.ceil(torch.Tensor(size)/dim_batch_size).int() #the actual number of batches per dimension (needed because of rounding error)
    
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


def run_spparks(ic, ea, nsteps=500, kt=0.66, cut=25.0, freq=(1,1), rseed=None, miso_array=None, which_sim='eng', num_processors=1, bcs=['p','p','p'], save_sim=True, del_sim=False, path_sim=None):
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
        num_processors: does not affect agg, agg can only do 1
    Output:
        path_sim
    '''
    
    # Find a simulation path that doesn't already exist (or if not told exactly where to run the simulation)
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
    if which_sim=='eng': 
        calc_MisoEnergy(r"./")
        os.system('./eng.sh')
    else: 
        os.system('./agg.sh')
    os.chdir(path_home)
    print("\nSIMULATION COMPLETE \nSIMULATION PATH: %s\n"%path_sim)
    
    # Save Simulation
    if save_sim==True:
        
        # Create miso_matrix
        miso_matrix = miso_array_to_matrix(torch.from_numpy(miso_array[None,]))[0].numpy()
        
        # Read dump
        size = ic.shape
        sz_str = ''.join(['%dx'%i for i in size])[:-1]
        fp_save = './data/spparks_sz(%s)_ng(%d)_nsteps(%d)_freq(%.1f)_kt(%.2f)_cut(%d).h5'%(sz_str,ngrain,nsteps,freq[1],kt,cut)
        tmp = np.array([8,16,32], dtype='uint64')
        dtype = 'uint' + str(tmp[np.sum(ngrain>2**tmp)])
        
        with h5py.File(fp_save, 'a') as f:
            
            # If file already exists, create another group in the file for this simulaiton
            num_groups = len(f.keys())
            hp_save = 'sim%d'%num_groups
            g = f.create_group(hp_save)
            
            # Save data
            nsteps_tot = int(nsteps/freq_dump)
            dset = g.create_dataset("ims_id", shape=(nsteps_tot+1, 1,)+size, dtype=dtype)
            dset1 = g.create_dataset("ims_energy", shape=(nsteps_tot+1, 1,)+size)
            dset2 = g.create_dataset("euler_angles", shape=ea.shape)
            dset3 = g.create_dataset("miso_array", shape=miso_array.shape)
            dset4 = g.create_dataset("miso_matrix", shape=miso_matrix.shape)
            dset2[:] = ea
            dset3[:] = miso_array #radians (does not save the exact "Miso.txt" file values, which are degrees divided by the cutoff angle)
            dset4[:] = miso_matrix #same values as mis0_array, different format
            
            item_itr = process_dump_item('%s/spparks.dump'%path_sim)
            for i, (im_id, _, im_energy) in enumerate(tqdm(item_itr, 'Reading dump', total=nsteps_tot+1)):
                dset[i,0] = im_id
                dset1[i,0] = im_energy
            
        return fp_save
    
    if del_sim: os.system(r"rm -r %s"%path_sim) #remove entire folder
    
    return None


def read_dump_item(path_to_dump):
    #Can't seek through file reliably, item IDs different length and inconsistent
    
    with open(path_to_dump) as file: 
        
        line = file.readline()
        item = line[6:].replace('\n', '')
        log = []
        # for line in file.readlines():
        while line != '':
            line = file.readline()
            if 'ITEM:' in line or line=='':
                data = np.stack(log)
                yield item, data
                item = line[6:].replace('\n', '')
                log = []
            else:
                data_line = np.array(line.split()).astype(float)
                log.append(data_line)
            
            
def process_dump_item(path_to_dump):
    for i, (item, data) in enumerate(read_dump_item(path_to_dump)): 
    
        # if 'TIMESTEP' in item: print('\rReading step: %f'%data[0,-1], end="\r")
        if 'BOX BOUNDS' in item: dims = np.ceil(data[:,-1]).astype(int) #z,y,x
        if 'ATOMS id type d1 d2 d3 energy' in item:
            
            if i==3: #first pass through atoms
                # Find indicies to sort data by "ATOMS: id", which is not linear when running on multiple cores
                i_sort = np.argsort(data[:,0]) #find the sort indices for the first step (assume each step is the same order)
                
                # Find euler angles
                num_grains = int(np.max(data[:,1]))
                euler_angles = np.zeros([num_grains,3])
                j = data[:,1].astype(int)-1 #subtract 1 so min ID is 0 and not 1
                euler_angles[j,:] = data[:,2:-1]
            
            # Sort data
            data = data[i_sort,:]
            
            # Arrange ID images
            im_id = data[:,1].reshape(tuple(np.flip(dims))).transpose([2,1,0]).squeeze()-1 #subtract 1 so min ID is 0 and not 1
            
            # Arrange energy images
            im_energy = data[...,-1].reshape(tuple(np.flip(dims))).transpose([2,1,0]).squeeze()
            
            yield im_id, euler_angles, im_energy


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
    
    # Sort by "ATOMS id", which are not linear when running on multiple cores
    i = np.argsort(item_data[3][0,:,0]) #find the sort indices for the first step (assume each step is the same order)
    item_data[3] = item_data[3][:,i,:]
    
    # Find simulation dimensions
    dims = np.flip(np.ceil(item_data[2][0,:,-1]).astype(int))
    
    # Arrange ID images
    ims_id = item_data[3][...,1].reshape((-1,)+tuple(dims)).transpose([0,3,2,1]).squeeze()[:,None,]-1
    
    # Arrange energy images
    ims_energy = item_data[3][...,-1].reshape((-1,)+tuple(dims)).transpose([0,3,2,1]).squeeze()[:,None,]
    
    # Find euler angles per ID
    num_grains = int(np.max(item_data[3][0,:,1]))
    euler_angles = np.zeros([num_grains,3])
    for i in range(np.product(dims)):
        ii = int(item_data[3][0,i,1])-1
        ea = item_data[3][0,i,2:-1]
        euler_angles[ii] = ea
    
    return ims_id, euler_angles, ims_energy


def create_SPPARKS_dataset(size=[257,257], ngrains_rng=[256, 256], kt=0.66, cutoff=25.0, nsets=200, max_steps=100, offset_steps=1, future_steps=4, freq = (1,1), del_sim=False):
    #'freq' - [how often to report stats on the simulation, how often to dump an image or record]
            
    # SET SIMULATION PATH
    path_sim = './spparks_simulation_trainset/'
        
    # NAMING CONVENTION   
    sz_str = ''.join(['%dx'%i for i in size])[:-1]
    fp = './data/trainset_spparks_sz(%s)_ng(%d-%d)_nsets(%d)_future(%d)_max(%d)_kt(%.2f)_freq(%.1f)_cut(%d).h5'%(sz_str,ngrains_rng[0],ngrains_rng[1],nsets,future_steps,max_steps,kt,freq[0],cutoff)

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
            rseed = np.random.randint(10000) #change to get different growth from teh same initial condition
            
            # RUN SIMULATION
            im, ea, _ = voronoi2image(size, ngrains) #generate initial condition
            miso_array = find_misorientation(ea, mem_max=1) 
            run_spparks(im, ea, nsteps, kt, cutoff, freq, rseed, miso_array=miso_array, save_sim=False, del_sim=del_sim, path_sim=path_sim, num_processors=32)
            grain_ID_images, grain_euler_angles, ims_energy = process_dump('%s/spparks.dump'%path_sim)
            # miso = np.loadtxt('%s/Miso.txt'%path_sim)*cutoff/180*np.pi #convert to radians
            
            # WRITE TO FILE
            dset[i,] = grain_ID_images[-(future_steps+1):,] 
            dset1[i,] = ims_energy[-(future_steps+1):,] 
            dset2[i,:ngrains,] = grain_euler_angles
            dset3[i,:int(ngrains*(ngrains-1)/2),] = miso_array
            
    if del_sim: os.system(r"rm -r %s"%path_sim) #remove entire folder
    
    return fp


def create_SPPARKS_dataset_circles(size=[512,512], radius_rng=[64,200], kt=0.66, cutoff=0.0, nsets=200, max_steps=10, offset_steps=1, future_steps=4, freq = (1,1), del_sim=False):
    #'freq' - [how often to report stats on the simulation, how often to dump an image or record]
            
    # SET SIMULATION PATH
    path_sim = './spparks_simulation_trainset/'
        
    # NAMING CONVENTION   
    sz_str = ''.join(['%dx'%i for i in size])[:-1]
    fp = './data/trainset_spparks_sz(%s)_r(%d-%d)_nsets(%d)_future(%d)_max(%d)_kt(%.2f)_freq(%.1f)_cut(%d).h5'%(sz_str,radius_rng[0],radius_rng[1],nsets,future_steps,max_steps,kt,freq[0],cutoff)

    # DETERMINE THE SMALLEST POSSIBLE DATA TYPE POSSIBLE
    m = 2 #number of grains
    dtype = 'uint8'
    
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
            r = np.random.randint(radius_rng[0], radius_rng[1]+1) #number of grains
            nsteps = np.random.randint(offset_steps+future_steps, max_steps+1) #SPPARKS steps to run
            rseed = np.random.randint(10000) #change to get different growth from teh same initial condition
            
            # RUN SIMULATION
            im, ea = generate_circleIC(size, r)
            miso_array = find_misorientation(ea, mem_max=1) 
            run_spparks(im, ea, nsteps, kt, cutoff, freq, rseed, miso_array=miso_array, save_sim=False, del_sim=del_sim, path_sim=path_sim, num_processors=32)
            grain_ID_images, grain_euler_angles, ims_energy = process_dump('%s/spparks.dump'%path_sim)
            # miso = np.loadtxt('%s/Miso.txt'%path_sim)*cutoff/180*np.pi #convert to radians
            
            # WRITE TO FILE
            dset[i,] = grain_ID_images[-(future_steps+1):,] 
            dset1[i,] = ims_energy[-(future_steps+1):,] 
            dset2[i,:] = grain_euler_angles
            dset3[i,:,] = miso_array
            
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


def num_features(ims, window_size=17, pad_mode='circular'):
    edges = num_diff_neighbors(ims, window_size, pad_mode)
    edges_flat = edges.reshape([ims.shape[0], -1])
    return torch.sum(edges_flat!=0, dim=1)


def trainset_calcNumFeatures(fp, window_size, if_plot=False):
    g = 'num_features_%d'%window_size
    
    if_exist = check_exist_h5([fp], [g], [], if_bool=True)
    
    with h5py.File(fp, 'a') as f:
        
        if not if_exist: 
            
            ni = f['ims_id'].shape[0]
            log = []
            for i in tqdm(range(ni), 'Calculating number of features for training:'):
                im = torch.from_numpy(f['ims_id'][i,0][None].astype(int)).to(device)
                nf = num_features(im, window_size, pad_mode='circular')
                log.append(nf)
            nf = torch.cat(log).cpu().numpy()
            
            f[g] = nf
        else:
            nf = f[g][:]
    
    if if_plot:
        cs = np.cumsum(nf)
        plt.figure()
        plt.plot(cs)
        plt.title('Cumulative number of features')
        plt.xlabel('Number of sets')
        plt.ylabel('Number of features')
        plt.show()
    
    return nf


def trainset_cutNumFeatures(fp, window_size, cut_f):

    nf = trainset_calcNumFeatures(fp, window_size)
    cs = np.cumsum(nf)
    i = np.argmin((np.abs(cs-cut_f)).astype(int))+1
    
    tmp0 = fp.split('nsets(')[0]
    tmp1 = fp.split(')_future')[1]
    fp_new = tmp0+'nsets(%d_%df)_future'%(i,cs[i-1])+tmp1
    
    with h5py.File(fp, 'r') as f:
        with h5py.File(fp_new, 'w') as fn:
            fn['ims_id'] = f['ims_id'][:i]
            fn['euler_angles'] = f['euler_angles'][:]
            fn['miso_array'] = f['miso_array'][:]
            




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
        
        # # Find the roation that gives the minimum angle
        # angle0 = 2*torch.acos(qq[...,0])
        
        # angle0[angle0>np.pi] = angle0[angle0>np.pi] - 2*np.pi
        
        # i_min = torch.argmin(torch.abs(angle0), axis=0)
        # qqmin = qq[i_min,torch.arange(len(ii))]
        
        # # Find and store the angle/axis values for this minimum angle rotations
        # angle_tmp = angle0[i_min, torch.arange(len(ii))][:,None] 
        # axis_tmp = qqmin[...,1:]/torch.sin(angle_tmp/2) #the axis might be different than spparks code, with the same misorientation angle
        # angles.append(torch.abs(angle_tmp).cpu().numpy())
        # axis.append(axis_tmp.cpu().numpy())

    return np.hstack(angles) #misorientation is the angle, radians





### Statistical functions
#Written by Kristien Everett, code optimized and added to by Joseph Melville 

# def find_frame_num_grains(grain_areas, num_grains=7500, min_pix=50):
    # tmp = np.sum(grain_areas>min_pix, axis=1)
    # i = np.argmin(np.abs(tmp-num_grains))
    # return i
    
    
def plotly_micro(im):
    #Plot a 3D image of the microstructure "im"
    #im - shape=(dim0, dim1, dim2)
    
    import plotly.graph_objects as go
    import plotly.io as pio
    pio.renderers.default = 'svg' #svg, browser
    
    sz = im.shape
    X, Y, Z = np.mgrid[0:sz[0], 0:sz[1], 0:sz[2]]
    fig = go.Figure(data=go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=im.flatten(),
        opacity=0.1, # needs to be small to see through all surfaces
        surface_count=5, # needs to be a large number for good volume rendering
        ))
    fig.show()


# def create_3D_paraview_vtr(ims, fp='micro_grid.vtr'):
#     #ims.shape = (d1, d2, d3), numpy
#     cells_coords = [np.arange(s+1)-int(s/2) for s in ims.shape]
#     grid = RectilinearGrid(fp, cells_coords, compression=True)
#     grid.addCellData(DataArray(ims, range(3), 'grains'))
#     grid.write()
    

def find_frame_num_grains(h5_group, num_grains=7500, min_pix=50):
    grain_areas = h5_group['grain_areas'][:]
    tmp = np.sum(grain_areas>min_pix, axis=1)
    i = np.argmin(np.abs(tmp-num_grains))
    # h5_group['frame_at_%d_grains'%num_grains] = np.array([i])
    return i


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
    # sizes = 2*torch.sqrt(areas/np.pi) #assumes grain size equals the diameter of a circular area - i.e. d = 2 * (A/pi)^(1/2)

    return areas


def find_grain_num_neighbors(im, max_id=19999, if_AW=False):
    #"im" is a torch.Tensor grain id image of shape=(1,1,dim1,dim2) (only one image at a time)
    #'max_id' defines which grain id neighbors should be returned -> range(0,max_id+1)
    #Outputs are of length 'max_id'+1 where each element corresponds to the respective grain id
    
    im = im[0,0,]
    d = im.dim()
    sz = torch.Tensor([im.shape]).T.to(im.device)
    
    # Find coordinates for pixels and define shifts
    ii = [torch.arange(sz[i,0]).to(im.device) for i in range(d)]
    coords = torch.cartesian_prod(*ii).float().transpose(0,1).reshape(d, -1)
    shifts = torch.eye(d).to(im.device)
    
    # Find all neighbor pairs possible
    p = []
    for i in range(d):
        s = shifts[i,][:,None] #coordintions of shift
        c0 = [*coords.long()]
        cu = [*((coords+s)%sz).long()]
        cd = [*((coords-s)%sz).long()]
        p.append(torch.stack([im[c0], im[cu]]))
        p.append(torch.stack([im[c0], im[cd]]))
    pairs = torch.hstack(p) 
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
    
    
def find_grain_aspect(ii):
    
    tmp = torch.from_numpy(np.cov(ii.cpu().numpy())).to(ii.device)
    
    a0, a1 = torch.linalg.eig(tmp)
    a0, a1 = a0.real, a1.real
    i_maj = torch.argmax(a0)
    i_min = torch.argmin(a0)
    v_maj = a1[:,i_maj] #major axis
    v_min = a1[:,i_min] #minor axis
    tmp = torch.sqrt(a0)
    r_maj = tmp[i_maj] #radius
    r_min = tmp[i_min] #radius
    return r_maj, r_min, v_maj, v_min


def find_aspect_ratios(im, max_id=19999, min_pix=5):
    if torch.min(im)!=0: im = im-1
    im = im[0,0,]
    aspects = torch.zeros(6,max_id+1)
    grains, counts = torch.unique(im, return_counts=True)
    grains = grains[counts>min_pix]
    for g in grains:
        ii = torch.stack(torch.where(im==g))
        r_maj, r_min, v_maj, v_min = find_grain_aspect(ii)
        aspects[:,g.long()] = torch.hstack([r_maj, r_min, v_maj, v_min])
    return aspects


def my_unfoldNd(ims, kernel_size=3, pad_mode='circular'):
    #Pads "ims" before unfolding
    dims = len(ims.shape)-2
    if type(kernel_size)!=list: kernel_size = [kernel_size] #convert to "list" if it isn't
    kernel_size = kernel_size + [kernel_size[-1]]*(dims-len(kernel_size)) #copy last dimension if needed
    pad = tuple((torch.Tensor(kernel_size).repeat_interleave(2)/2).int().numpy()) #calculate padding needed based on kernel_size
    if pad_mode!=None: ims = pad_mixed(ims, pad, pad_mode) #pad "ims" to maintain dimensions after unfolding
    ims_unfold = unfoldNd(ims, kernel_size=kernel_size) #shape = [N, product(kernel_size), dim1*dim2*dim3]
    return ims_unfold


def miso_array_to_matrix(miso_arrays):
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


def miso_matrix_to_array(miso_matrix):
    i, j = np.tril_indices(miso_matrix.shape[0], -1)
    return miso_matrix[i,j]


def gid_to_miso(ims_unfold, miso_matrices):
    # Convert each grain id value into a misorientation between each center and neighbor
    # ims_unfold: grain ids, shape=(num_ims, num_neighbors, num_image_elements)
    # miso_matrices: grain id misorientations, shape=(num_images, dim1, dim2)
    # ims_unfold_miso: misorientations, shape=(num_ims, num_neighbors, num_image_elements)
    
    num_ims = ims_unfold.shape[0]
    num_neigh = ims_unfold.shape[1]
    num_elem = ims_unfold.shape[2] 
    center_pxl_ind = int(num_neigh/2)
    
    tmp = ims_unfold[:,center_pxl_ind:center_pxl_ind+1,] 
    ims_repeat = tmp.repeat(1, num_neigh, 1) #repeat the centers to compare against all neighbors
    
    # Define the indicies to search the miso_matrices for corresponding misorientations
    i = (ims_unfold.flatten()).long() 
    j = (ims_repeat.flatten()).long() 
    tmp = torch.zeros([1,num_neigh*num_elem])
    tmp2 = [tmp+h for h in range(num_ims)]
    k = torch.cat(tmp2).long().flatten().to(ims_unfold.device)
    
    ims_unfold_miso = miso_matrices[k,i,j].reshape(ims_unfold.shape)
    return ims_unfold_miso


def neighborhood_miso(ims, miso_matrices, window_size=3, pad_mode='circular'): 
    # ims - torch.Tensor of shape [# of images, 1, dim1, dim2, dim3(optional)]
    # miso_matrices: grain id misorientations in radians, shape=(num_images, dim1, dim2)
    # window_size - the patch around each pixel that constitutes its neighbors
    # May need to add memory management through batches for large tensors in the future
    
    if type(window_size)==int: window_size = [window_size] #convert to "list" if "int" is given
    
    ims_unfold = my_unfoldNd(ims, kernel_size=window_size, pad_mode=pad_mode)
    # miso_matrices = miso_array_to_matrix(miso_arrays) #indicies to convert miso array to matrix
    # del miso_arrays
    ims_unfold_miso = gid_to_miso(ims_unfold, miso_matrices)
    del miso_matrices
    
    if pad_mode==None: s = ims.shape[:2]+tuple(np.array(ims.shape[2:])-window_size+1)
    else: s = ims.shape
    ims_miso = torch.sum(ims_unfold_miso, axis=1).reshape(s) #misorientation image
    return ims_miso #reshape to orignal image shape


def neighborhood_miso_spparks(ims, miso_matrices, cut=25, window_size=3, pad_mode='circular'): 
    #ims - torch.Tensor of shape [# of images, 1, dim1, dim2, dim3(optional)]
    #'miso_matrices' - grain id misorientations in radians
    #window_size - the patch around each pixel that constitutes its neighbors
    #May need to add memory management through batches for large tensors in the future
    #Calculated the same as in spparks
    
    if type(window_size)==int: window_size = [window_size] #convert to "list" if "int" is given
    
    ims_unfold = my_unfoldNd(ims, kernel_size=window_size, pad_mode=pad_mode)
    # miso_matrices = miso_array_to_matrix(miso_arrays) #indicies to convert miso array to matrix
    # del miso_arrays
    ims_unfold_miso = gid_to_miso(ims_unfold, miso_matrices)
    del miso_matrices
    
    if pad_mode==None: s = ims.shape[:2]+tuple(np.array(ims.shape[2:])-window_size+1)
    else: s = ims.shape
    
    ims_unfold_miso = ims_unfold_miso/np.pi*180 #convert to degrees
    r = ims_unfold_miso/cut
    tmp = r*(1-torch.log(r))
    tmp[torch.isnan(tmp)] = 0
    tmp[ims_unfold_miso>cut] = 1
    
    ims_miso = torch.sum(tmp, axis=1).reshape(s) #misorientation image
    
    return ims_miso #reshape to orignal image shape


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


def compute_grain_stats(hps, gps='last', device=device):
    
    #Make 'hps' and 'gps' a list if it isn't already, and set default 'gps'
    if type(hps)!=list: hps = [hps]
    
    if gps=='last':
        gps = []
        for hp in hps:
            with h5py.File(hp, 'r') as f:
                gps.append(list(f.keys())[-1])
        print('Last groups in each h5 file chosen:')
        print(gps)
    else:
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
            miso_matrix = torch.from_numpy(g['miso_matrix'][:]).to(device)
            
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
            
            # Find misorientation images
            if 'ims_miso' not in g.keys():
                args = [miso_matrix[None,]]
                func = neighborhood_miso
                ims_miso = iterate_function(d, func, args)[:,0]
                g['ims_miso'] = ims_miso
                print('Calculated: ims_miso')
            else: ims_miso = None
                
            # Find average misorientation per boundary pixel
            if 'ims_miso_avg' not in g.keys():
                if np.all(ims_miso==None): ims_miso = g['ims_miso']
                func = mean_wo_zeros
                ims_miso_avg = iterate_function(ims_miso, func, args=[])
                g['ims_miso_avg'] = ims_miso_avg
                print('Calculated: ims_miso_avg')
            
            # Find misorientation images using the SPPARKS method
            if 'ims_miso_spparks' not in g.keys():
                args = [miso_matrix[None,]]
                func = neighborhood_miso_spparks
                ims_miso_spparks = iterate_function(d, func, args)[:,0]
                g['ims_miso_spparks'] = ims_miso_spparks
                print('Calculated: ims_miso_spparks')
            else: ims_miso_spparks = None
                
            # Find average misorientation per boundary pixel using the SPPARKS method
            if 'ims_miso_spparks_avg' not in g.keys():
                if np.all(ims_miso_spparks==None): ims_miso_spparks = g['ims_miso_spparks']
                func = mean_wo_zeros
                ims_miso_spparks_avg = iterate_function(ims_miso_spparks, func, args=[])
                g['ims_miso_spparks_avg'] = ims_miso_spparks_avg
                print('Calculated: ims_miso_spparks_avg')
                
            # # Find dihedral angle standard deviation
            # if 'dihedral_std' not in g.keys():
            #     func = find_dihedral_stats
            #     dihedral_std = iterate_function(d, func)
            #     g['dihedral_std'] = dihedral_std
            #     print('Calculated: dihedral_std')
            
            # # Find grain aspect ratios
            # if 'aspects' not in g.keys():
            #     args = [max_id]
            #     func = find_aspect_ratios
            #     aspects = iterate_function(d, func, args)
            #     g['aspects'] = aspects
            
            # # Find inclination of boundary pixels
            # if 'ims_inclination' not in g.keys():
            #     func = fs2.find_inclination
            #     ims_inclination = fs2.iterate_function(d[0:1,], func, args=[])[:,0]
            #     g['ims_inclination'] = ims_inclination
            
            # # Setup arguments
            # ea = g['euler_angles'][:]
            # ims_inclination = g['ims_inclination'][:]
            
            # # Find angle between inclination and orientation at each boundary pixel
            # if 'ims_inc_angle' not in g.keys():
            #     args = [ea, ims_inclination]
            #     func = find_inclination_orientation_angle
            #     ims_inc_angle = iterate_function(d, func, args)[:,0]
            #     g['ims_inc_angle'] = ims_inc_angle
            
            #curvature


def make_videos(hps, gps='last'):
    # Run "compute_grain_stats" before this function
    
    #Make 'hps' and 'gps' a list if it isn't already
    if type(hps)!=list: hps = [hps]
    
    # Set default 'gps'
    if gps=='last':
        gps = []
        for hp in hps:
            with h5py.File(hp, 'r') as f:
                gps.append(list(f.keys())[-1])
        print('Last groups in each h5 file chosen:')
        print(gps)
    else:
        if type(gps)!=list: gps = [gps]
    
    # Make sure all needed datasets exist
    dts=['ims_id', 'ims_miso', 'ims_miso_spparks']
    check_exist_h5(hps, gps, dts)  
    
    for i in tqdm(range(len(hps)), "Making videos"):
        with h5py.File(hps[i], 'a') as f:
            
            g = f[gps[i]]
            
            # If 3D, split down the middle of the first axis
            sz = g['ims_id'].shape[2:]
            dim = len(sz)
            mid = int(sz[0]/2)
            if dim==2: j = np.arange(sz[0])
            elif dim==3: j = mid
            
            ims = g['ims_id'][:,0,j]
            ims = (255/np.max(ims)*ims).astype(np.uint8)
            imageio.mimsave('./plots/ims_id%d.mp4'%(i), ims)
            imageio.mimsave('./plots/ims_id%d.gif'%(i), ims)
            
            ims = g['ims_miso'][:,0,j]
            ims = (255/np.max(ims)*ims).astype(np.uint8)
            imageio.mimsave('./plots/ims_miso%d.mp4'%(i), ims)
            imageio.mimsave('./plots/ims_miso%d.gif'%(i), ims)
            
            ims = g['ims_miso_spparks'][:,0,j]
            ims = (255/np.max(ims)*ims).astype(np.uint8)
            imageio.mimsave('./plots/ims_miso_spparks%d.mp4'%(i), ims)
            imageio.mimsave('./plots/ims_miso_spparks%d.gif'%(i), ims)

        
def make_time_plots(hps, gps='last', scale_ngrains_ratio=0.05, cr=None, legend=True, if_show=True):
    # Run "compute_grain_stats" before this function
    
    #Make 'hps' and 'gps' lists if they aren't already, and set default 'gps'
    if type(hps)!=list: hps = [hps]
    
    if gps=='last':
        gps = []
        for hp in hps:
            with h5py.File(hp, 'r') as f:
                gps.append(list(f.keys())[-1])
        print('Last groups in each h5 file chosen:')
        print(gps)
    else:
        if type(gps)!=list: gps = [gps]
    
    # Establish color table
    c = [mcolors.TABLEAU_COLORS[n] for n in list(mcolors.TABLEAU_COLORS)]
    if np.all(cr!=None): #repeat the the color labels using "cr"
        tmp = []
        for i, e in enumerate(c[:len(cr)]): tmp += cr[i]*[e]
        c = tmp
    
    # Make sure all needed datasets exist
    # dts=['grain_areas', 'grain_sides', 'ims_miso', 'ims_miso_spparks']
    # check_exist_h5(hps, gps, dts)  
    
    # Calculate scale limit
    with h5py.File(hps[0], 'r') as f:
        g = f[gps[0]]
        total_area = np.product(g['ims_id'].shape[1:])
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
    legend = []
    for i in range(len(hps)):
        plt.plot(log[i], c=c[i%len(c)])
        legend.append('Slope: %.3f | R2: %.3f'%(ps[i][0], rs[i]))
    plt.title('Average grain area')
    plt.xlabel('Number of frames')
    plt.ylabel('Average area (pixels)')
    if legend==True: plt.legend(legend)
    plt.savefig('./plots/avg_grain_area_time', dpi=300)
    if if_show: plt.show()

    # Plot scaled average grain area through time and find linear slopes
    ys = []
    ps = []
    rs = []
    si = []
    xs = []
    for i in range(len(hps)):
        grain_areas_avg = log[i]
        ii = len(grain_areas_avg) - 1 - np.argmin(np.flip(np.abs(grain_areas_avg-lim)))
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
    legend = []
    for i in range(len(hps)):
        plt.plot(xs[i], log[i][:len(xs[i])], c=c[i%len(c)])
        plt.xlim([np.max(xs[i]), np.min(xs[i])])
        legend.append('Slope: %.3f | R2: %.3f'%(ps[i][0], rs[i]))
    plt.title('Average grain area (scaled)')
    plt.xlabel('Number of grains')
    plt.ylabel('Average area (pixels)')
    if legend==True: plt.legend(legend)
    plt.savefig('./plots/avg_grain_area_time_scaled', dpi=300)
    if if_show: plt.show()
    
    # Plot average grain sides through time
    log = []
    for i in tqdm(range(len(hps)),'Plotting avg grain sides'):
        with h5py.File(hps[i], 'r') as f: 
            grain_sides_avg = f[gps[i]+'/grain_sides_avg'][:]
        log.append(grain_sides_avg)
    
    plt.figure()
    legend = []
    for i in range(len(hps)):
        plt.plot(log[i], c=c[i%len(c)])
        legend.append('')
    plt.title('Average number of grain sides')
    plt.xlabel('Number of frames')
    plt.ylabel('Average number of sides')
    if legend==True: plt.legend(legend)
    plt.savefig('./plots/avg_grain_sides_time', dpi=300)
    if if_show: plt.show()
    
    # Plot scaled average grain sides through time
    plt.figure()
    legend = []
    for i in range(len(hps)):
        plt.plot(xs[i], log[i][:len(xs[i])], c=c[i%len(c)])
        plt.xlim([np.max(xs[i]), np.min(xs[i])])
        legend.append('')
    plt.title('Average number of grain sides (scaled)')
    plt.xlabel('Number of grains')
    plt.ylabel('Average number of sides')
    if legend==True: plt.legend(legend)
    plt.savefig('./plots/avg_grain_sides_time_scaled', dpi=300)
    if if_show: plt.show()
    
    # Plot grain radius distribution
    plt.figure()
    frac = 0.25
    for i in tqdm(range(len(hps)),'Calculating normalized radius distribution'):
        with h5py.File(hps[i], 'r') as f: 
            grain_areas = f[gps[i]+'/grain_areas'][:]
        tg = (grain_areas.shape[1])*frac
        ng = (grain_areas!=0).sum(1)
        j = (ng<tg).argmax()
        ga = grain_areas[j]
        gr = np.sqrt(ga/np.pi)
        bins=np.linspace(0,3,20)
        gr_dist, _ = np.histogram(gr[gr!=0]/gr[gr!=0].mean(), bins)
        plt.plot(bins[1:]-0.5*(bins[1]-bins[0]), gr_dist/gr_dist.sum()/bins[1])
    plt.title('Normalized radius distribution (%d%% grains remaining)'%(100*frac))
    plt.xlabel('R/<R>')
    plt.ylabel('Frequency')
    if legend==True: plt.legend(legend)
    plt.savefig('./plots/normalized_radius_distribution', dpi=300)
    if if_show: plt.show()
    
    # Plot number of sides distribution
    plt.figure()
    frac = 0.25
    for i in tqdm(range(len(hps)),'Calculating number of sides distribution'):
        with h5py.File(hps[i], 'r') as f: 
            grain_areas = f[gps[i]+'/grain_areas'][:]
            grain_sides = f[gps[i]+'/grain_sides'][:]
        tg = (grain_areas.shape[1])*frac
        ng = (grain_areas!=0).sum(1)
        j = (ng<tg).argmax()
        gs = grain_sides[j]
        
        if i==0: #set histogram limits base on first set
            low = np.min(gs[gs!=0])-1
            high = np.max(gs[gs!=0])+1
            if high>30: high=30
        
        bins=np.arange(low,high+2)-0.5
        gs_dist, _ = np.histogram(gs[gs!=0], bins)
        plt.plot(bins[1:]-0.5, gs_dist)
        
    plt.title('Number of sides distribution (%d%% grains remaining)'%(100*frac))
    plt.xlabel('Number of sides')
    plt.ylabel('Frequency')
    if legend==True: plt.legend(legend)
    plt.savefig('./plots/number_sides_distribution', dpi=300)
    if if_show: plt.show()
        
    # Plot average misorientation per bounday pixel
    log = []
    for i in tqdm(range(len(hps)),'Plotting average miso'):
        with h5py.File(hps[i], 'r') as f: 
            ims_miso_avg = f[gps[i]+'/ims_miso_avg'][:]
        log.append(ims_miso_avg)
    
    plt.figure()
    legend = []
    for i in range(len(hps)):
        plt.plot(log[i], c=c[i%len(c)])
        legend.append('')
    plt.title('Average miso per boundary pixel')
    plt.xlabel('Number of frames')
    plt.ylabel('Average miso per boundary pixel')
    if legend==True: plt.legend(legend)
    plt.savefig('./plots/avg_miso_time', dpi=300)
    if if_show: plt.show()
    
    # Plot scaled average misorientation per bounday pixel
    plt.figure()
    legend = []
    for i in range(len(hps)):
        plt.plot(xs[i], log[i][:len(xs[i])], c=c[i%len(c)])
        plt.xlim([np.max(xs[i]), np.min(xs[i])])
        legend.append('')
    plt.title('Average miso per boundary pixel (scaled)')
    plt.xlabel('Number of grains')
    plt.ylabel('Average miso per boundary pixel')
    if legend==True: plt.legend(legend)
    plt.savefig('./plots/avg_miso_time_scaled', dpi=300)
    if if_show: plt.show()
    
    # Plot average misorientation per bounday pixel (SPPARKS)
    log = []
    for i in tqdm(range(len(hps)),'Plotting average spparks miso'):
        with h5py.File(hps[i], 'r') as f: 
            ims_miso_spparks_avg = f[gps[i]+'/ims_miso_spparks_avg'][:]
        log.append(ims_miso_spparks_avg)
    
    plt.figure()
    legend = []
    for i in range(len(hps)):
        plt.plot(log[i], c=c[i%len(c)])
        legend.append('')
    plt.title('Average miso per boundary pixel (SPPARKS)')
    plt.xlabel('Number of frames')
    plt.ylabel('Average miso per boundary pixel')
    if legend==True: plt.legend(legend)
    plt.savefig('./plots/avg_miso_spparks_time', dpi=300)
    if if_show: plt.show()
    
    # Plot scaled average misorientation per bounday pixel (SPPARKS)
    plt.figure()
    legend = []
    for i in range(len(hps)):
        plt.plot(xs[i], log[i][:len(xs[i])], c=c[i%len(c)])
        plt.xlim([np.max(xs[i]), np.min(xs[i])])
        legend.append('')
    plt.title('Average miso per boundary pixel (SPPARKS, scaled)')
    plt.xlabel('Number of grains')
    plt.ylabel('Average miso per boundary pixel')
    if legend==True: plt.legend(legend)
    plt.savefig('./plots/avg_miso_spparks_time_scaled', dpi=300)
    if if_show: plt.show()
    
    # # Plot dihedral angle distribution standard deviation over time
    # log = []
    # for i in tqdm(range(len(hps)),'Plotting dihedral angle STD'):
    #     with h5py.File(hps[i], 'r') as f: 
    #         dihedral_std = f[gps[i]+'/dihedral_std'][:]
    #     log.append(dihedral_std)
    
    # plt.figure()
    # legend = []
    # for i in range(len(hps)):
    #     plt.plot(log[i], c=c[i%len(c)])
    #     legend.append('')
    # plt.title('Dihedral angle distribution STD')
    # plt.xlabel('Number of frames')
    # plt.ylabel('STD (degrees)')
    # if legend==True: plt.legend(legend)
    # plt.savefig('./plots/dihedral_std', dpi=300)
    # if if_show: plt.show()
    
    # # Plot scaled dihedral angle distribution standard deviation over time
    # plt.figure()
    # legend = []
    # for i in range(len(hps)):
    #     plt.plot(xs[i], log[i][:len(xs[i])], c=c[i%len(c)])
    #     plt.xlim([np.max(xs[i]), np.min(xs[i])])
    #     legend.append('')
    # plt.title('Dihedral angle distribution STD (scaled)')
    # plt.xlabel('Number of grains')
    # plt.ylabel('STD (degrees)')
    # if legend==True: plt.legend(legend)
    # plt.savefig('./plots/dihedral_std_scaled', dpi=300)
    # if if_show: plt.show()
    
    # #vizualize the relationship between area change and number of sides
    # i=1
    # with h5py.File(hps[i], 'r') as f: 
    #     grain_areas = f[gps[i]+'/grain_areas'][:]
    #     grain_sides = f[gps[i]+'/grain_sides'][:]
    # grain_areas = grain_areas.astype(float)
    # grain_areas[grain_areas==0] = np.nan
    
    # for k in range(1001):
    #     a0 = grain_areas[:,k]
    #     a1 = grain_sides[:,k]
    #     b1 = a1>=6
        
    #     i = np.where(b1)[0]
    #     j = np.where(~b1)[0]
        
    #     plt.plot(i,a0[i],'tab:blue')
    #     plt.plot(j,a0[j],'tab:orange')
    
    # plt.legend(['>6 sides','<6 sides'])
    # plt.ylim([0,10000])
    # plt.title('Area and number of sides through time')
    # plt.xlabel('Frame')
    # plt.ylabel('Number of pixels')
    
    print(si)
    if not if_show: plt.close('all')
    

def circle_stats(fp):

    with h5py.File(fp, 'a') as f:
        ims = f['sim0/ims_id'][:]
        areas = ims.sum(1).sum(1).sum(1)
        f['sim0/circle_area'] = areas
        
        
def circle_videos(fp):
    
    with h5py.File(fp, 'r') as f:
        areas = f['sim0/circle_area'][:]
        ims = f['sim0/ims_id'][:]
        s = f['sim0/ims_id'].shape[2:]
    
    i_mid_growth = int(np.argmin(areas)/2)
    im_mid_growth = ims[i_mid_growth,0]
    plt.figure()
    plt.imshow(im_mid_growth)
    plt.title('Frame: %d'%i_mid_growth)
    plt.savefig('./plots/circle_mid_growth_%dx%d'%s, dpi=300)
    
    ims = (255/np.max(ims)*ims).astype(np.uint8)
    imageio.mimsave('./plots/ims_circle_%dx%d.mp4'%s, ims[:,0])
    imageio.mimsave('./plots/ims_circle_%dx%d.gif'%s, ims[:,0])
        

def circle_plots(fps):
    
    if type(fps) is not list: fps = [fps]
    
    plt.figure()
    for fp in fps:
        with h5py.File(fp, 'r') as f:
            areas = f['sim0/circle_area'][:]
            s = f['sim0/ims_id'].shape[2:]
        plt.plot(areas)
    plt.title('Circle area over time')
    plt.xlabel('Frames')
    plt.ylabel('Circle Aera')
    plt.savefig('./plots/circle_area_over_time_%dx%d'%s, dpi=300)





### Calculate dihedral angles

def cartesian_prod(xi):
    mi = list(torch.meshgrid(*xi))
    return torch.stack(mi).reshape(len(xi),-1)


def find_im_indices(size=[64,64,64]):
    xi = [torch.arange(s) for s in size]
    return cartesian_prod(xi)


def find_ncombo(im, n=3):
    #n=2 returns all pairs that can be made between a center pixel ID and neighbors of different values
    #n=3 is the same, but find triplets composed of the center and two nieghbors that all have different IDs
    #'ncombo_diff': shape=(n+d, number of combinations)
    #The first values are the IDs of the grains in the combination (size n)
    #The final values in the first dimension are the indices of the center value (size d)
    
    d = (im.dim()-2) #dimensions of the image
    if d==2: ni = np.array([1,3,5,7]) #indices of VonNeuman neighbors (N=1)
    else: ni = np.array([4, 10, 12, 14, 16, 22]) #for 3D
    nn = len(ni) #number of neighbors
    
    # Find all possible neighbor combinations
    im_unfold = my_unfoldNd(im)[0,ni]
    i = torch.combinations(torch.arange(nn), r=n-1, with_replacement=False).T.to(im.device) #find all combinations of neighbors that can make an N-combination with the center ID
    if n==3: i = i[:, i.sum(0)!=(nn-1)] #if finding junctions, remove cross combinations (top/bottom, left/right, in/out)
    i_center = find_im_indices(im.shape[2:]).to(im.device) #find the indicies at the center
    tmp = torch.stack([im.flatten(), *i_center])[:,None,].repeat(1,i.shape[1],1) #stack center indices with center values
    v_neighbors = [im_unfold[j,] for j in i] #values of neighbors in different combinations
    ncombo = torch.stack([*v_neighbors, *tmp]).reshape(n+d, -1) #then stack on neighbor values
    
    # Reduce to only combinations with all different IDs
    i = torch.combinations(torch.arange(n), r=2, with_replacement=False).T.to(im.device) #find all possible comparisons between the "n" ID values
    j = torch.sum(ncombo[i[0]]==ncombo[i[1]], dim=0)==0 #make all comparisions and only keep those that have all different ID values
    ncombo_diff = ncombo[:,j]
    
    #Sort the IDs so different combinations of the same numbers are the same
    ids_sort = torch.sort(ncombo_diff[:n], dim=0)[0]
    ncombo_diff[:n] = ids_sort
    
    return ncombo_diff


def find_ncombo_avg(ncombo, sz):
    #Find the average of indices with the same ID combinations
    #"ncombo" - shape=(n+2, num ID sets), sets of n combinations neighbors
    #For 2D and 3D
    #Assumes triplets only
    
    ids = ncombo[:3,][None,] #retrieve only the grain ID sets
    matches = torch.all(ids==ids.T, dim=1) #contruct a matching matrix (find which ID sets are equal)
    
    #Find the mean without zeros of all matching ID sets - when there is a "0" and a "256" in the locations, wrap "256" to "-1"
    nmatch = torch.sum(matches, dim=1) 
    
    tmp = []
    for i in range(len(sz)): 
        num_hi = torch.sum(matches*ncombo[i+3,:]==sz[i]-1, dim=1)
        has0 = torch.sum(matches*ncombo[i+3,:]!=0, dim=1)!=nmatch
        tmpi = torch.sum(matches*ncombo[i+3,:], dim=1)-(sz[i]*num_hi*has0)
        tmp.append(tmpi/nmatch) #mean without zero
        
    ncombo_avg = torch.stack([*ids[0], *tmp]) #add IDs back in
    ncombo_avg = torch.unique(ncombo_avg, dim=1) #remove duplicates
    
    if len(ncombo_avg)==0: ncombo_avg = torch.zeros([5,0]).to(ncombo.device)
    
    # num_hi = torch.sum(matches*ncombo[-2,:]==sz[0]-1, dim=1)
    # has0 = torch.sum(matches*ncombo[-2,:]!=0, dim=1)!=nmatch
    # tmpi = torch.sum(matches*ncombo[-2,:], dim=1)-(sz[0]*num_hi*has0)
    # i = tmpi/nmatch #mean without zero for x direction
    
    # num_hi = torch.sum(matches*ncombo[-1,:]==sz[1]-1, dim=1)
    # has0 = torch.sum(matches*ncombo[-1,:]!=0, dim=1)!=nmatch
    # tmpj = torch.sum(matches*ncombo[-1,:], dim=1)-(sz[1]*num_hi*has0)
    # j = tmpj/nmatch #mean without zero for y direction
    
    # ncombo_avg = torch.stack([*ids[0], i, j]) #add IDs back in
    # ncombo_avg = torch.unique(ncombo_avg, dim=1) #remove duplicates
    
    return ncombo_avg #shape=(n+2, num unique ID sets), first n values are the IDs, last two are the location indices


def test_ncombo_avg(im, ncombo_avg):
    #"im" - ID image, shape=(1,1,dim1,dim2)
    #"ncombo_avg" - shape=(n+2, num ID sets), sets of n combinations neighbors
    
    im = im.cpu()
    ncombo_avg = ncombo_avg.cpu()
    
    sz = im.shape[2:]
    
    ii = find_im_indices([3,3]) - int(3/2)
    for i in range(ncombo_avg.shape[1]):
        j = ncombo_avg[:,i].long()
        ids = j[:-2]
        p = im[0, 0, (j[-2]+ii[0])%sz[0], (j[-1]+ii[1])%sz[1]].cpu()
        
        for k in ids: 
            if k not in p: 
                print('Nieghbors and set IDs inconsistant for row: %d'%i)
                break
    
    im0 = num_diff_neighbors(im, window_size=3, pad_mode='circular')
    plt.imshow(im0[0,0,].cpu())
    plt.plot(ncombo_avg[-1],ncombo_avg[-2],'.r', markersize=5)
    plt.show()
    
    
def test_num_junctions_through_time(ims):
    #Finds the number of junctions per grain (should converge on 6 for 2D)
    #'ims' - torch, shape=(num images, 1, dim1, dim2)

    ll = []
    for i in tqdm(range(ims.shape[0])):
        im = ims[i][None,]
        ncombo = find_ncombo(im, n=3)
        ncombo_avg = find_ncombo_avg(ncombo, im.shape[2:]).cpu()
        
        ids = torch.unique(im.cpu())
        l = []
        for i in ids:
            nj = torch.sum(torch.any(ncombo_avg[:-2]==i, dim=0))
            l.append(nj)
            
        ll.append(torch.mean(torch.stack(l).float()))
    plt.plot(ll)
    plt.title('Number of junctions per grain through time')
    plt.xlabel('Frame number')
    plt.ylabel('Number of junctions per grain')
    
    
def find_juntion_neighbors(ncombo_avg):
    #For each junction, find the junctions that share exactly 2 IDs
    
    ids = ncombo_avg[:3,].flatten()[None,]
    adj_tmp = ids==ids.T #compare all ID values
    tmp = ncombo_avg.shape[1]
    adjacent = (adj_tmp.reshape(3,tmp,3,tmp).sum(0).sum(1))==2 #mark junctions that have exactly 2 IDs the same
    return adjacent


def find_angle_between(i,j,deg=False):
    #given angles i and j in degrees
    #outputs absolute value of the angle between i and j in degrees
    if deg: 
        i = i/180*np.pi
        j = j/180*np.pi
    # tmp = torch.arctan((torch.tan(i)-torch.tan(j))/(1+torch.tan(i)*torch.tan(j))) #not absolute value
    tmp = torch.acos(torch.cos(i)*torch.cos(j)+torch.sin(i)*torch.sin(j))
    if deg: return tmp/np.pi*180
    else: return tmp


def calc_dihedral_angles(junction_angles):
    #'edge_directions' - shape=(3, number of junctions), the direction from which the three edges exit junctions
    #output - shape=(3, number of junctions), the angles between the junctions edges
    a0 = find_angle_between(junction_angles[0],junction_angles[1],deg=True)
    a1 = find_angle_between(junction_angles[1],junction_angles[2],deg=True)
    a2 = find_angle_between(junction_angles[0],junction_angles[2],deg=True)
    
    i0 = torch.abs((a0-(a1+a2)))<0.001
    i1 = torch.abs((a1-(a0+a2)))<0.001
    i2 = torch.abs((a2-(a0+a1)))<0.001
    
    a0[i0] = 360-a0[i0]
    a1[i1] = 360-a1[i1]
    a2[i2] = 360-a2[i2]
    
    return torch.stack([a0, a1, a2])


def find_dihedral_angles(im, if_plot=False, num_plot_jct=10):
    #'im' - shape=(1,1,dim1,dim2), microstructureal image in which to find junction digedral angles
    #output - shape=(6, number of junctions), first three numbers are the IDs that define the junction, the last three are the dihedral angles between ID indices 0/1, 1/2, and 0/2    
    
    # Find triplet indices and neighbors 
    ncombo = find_ncombo(im, n=3) #find all indices included in a triplet
    ncombo_avg = find_ncombo_avg(ncombo, im.shape[2:]) #find the average location of those found in the same triplet
    adj = find_juntion_neighbors(ncombo_avg)  #find the neighbors for each triplet (share two of the same IDs)
    
    # Keep only triplets with 3 neighbors (true triplets)
    i = torch.sum(adj, dim=0)==3 
    adj1 = adj[i, :] 
    ncombo_avg1 = ncombo_avg[:, i]
    
    # Find junction pairs (first junction always has 3 neighbors and will be present in the first position exactly three times, the second might not)
    i, j = torch.where(adj1) 
    jpairs = torch.stack([ncombo_avg1[:,i], ncombo_avg[:,j]])
    
    # Find junction pair common IDs (find the two of the three IDs that match)
    tmp = (jpairs[0,:3][None]==jpairs[1,:3][:,None]).sum(0) 
    i, j = torch.where(tmp)
    ii = torch.argsort(j)
    i, j = [i[ii], j[ii]]
    jpair_ids = jpairs[0,i,j].reshape(-1,2).T
    jpair_ids = torch.sort(jpair_ids, dim=0)[0] #always have ascending ids
    
    # Find the edge indices that belong to each junction pair
    ncombo = find_ncombo(im, n=2) #find edge indicies
    jpair_edges = torch.all(jpair_ids.T[:,:,None]==ncombo[:-2][None,], dim=1) #These don't neccesarily include the junctions yet, because junctions are an average of triplets that don't include just these two ids
    
    #Remove all of the jpairs that have any edge that has a length of four or less
    edges_len = my_batch(jpair_edges, torch.sum, 100)
    i = (edges_len>4).reshape(-1,3).all(1)
    j = i[:,None].repeat(1,3).flatten()
    jpairs = jpairs[:,:,j]
    jpair_edges = jpair_edges[j,]
    edges_len = edges_len[j]
    
    if len(jpair_edges)==0: return None
    
    # Create a padded matrix to hold edge indices
    i, j = batch_where(jpair_edges, 100)
    edges_all = ncombo[-2:,j].T
    edges_split = torch.split(edges_all, list(edges_len))
    edges_padded = torch.nn.utils.rnn.pad_sequence(edges_split, padding_value=0)
    
    # Append start and end junction locations (ensure these locations are also sampled from in the next step)
    tmp0 = jpairs[0:1,-2:,:].permute(0,2,1) #start junction location
    tmp1 = jpairs[1:,-2:,:].permute(0,2,1) #end junction location
    edges_tmp = torch.cat([tmp0, tmp1, edges_padded])
    
    # Oversample non-zero values to fill in the padded zero regions
    i = ((edges_len[None,]+2)*torch.rand(edges_tmp.shape[:2]).to(im.device)).long()
    edges = edges_tmp[i, torch.arange(edges_tmp.shape[1]).long(), :]
    
    # Append the start junction location (ensure this is the location that is set to [0,0] for the line fit)
    tmp0 = jpairs[0:1,-2:,:].permute(0,2,1) #start junction location
    edges = torch.cat([tmp0, edges])
    
    # Unwrap edge indices that jump from one boundary to the other
    h = edges.max(0)[0].max(0)[0][None,None]
    j = jpairs[0,-2:,].T[None,]
    tmp = edges-j
    edges = edges - h*torch.sign(tmp)*(torch.abs(tmp)>(h/2))
    
    # Fit lines to all of these sets (do it twice - fit x to y and y to x to avoid infinite slopes)
    points = edges
    points = points - points[0,:,:]
    x = points[...,0].T
    y = points[...,1].T
    A = torch.stack([x, x**2, x**3]).permute(1,2,0)
    B = y[...,None]
    sx, rx0, _, _ = torch.linalg.lstsq(A, B)
    rx = ((torch.matmul(A,sx)-B)[...,0]**2).sum(1)[:,None]
    
    A = torch.stack([y, y**2, y**3]).permute(1,2,0)
    B = x[...,None]
    sy, ry0, _, _ = torch.linalg.lstsq(A, B)
    ry = ((torch.matmul(A,sy)-B)[...,0]**2).sum(1)[:,None]
    
    # Find junction angles and then dihedral angles
    i = (ry<rx)[:,0] #Keep x fit when its "r" value is lower
    
    ang_x = torch.atan(sx[:,0])/np.pi*180%360
    ang_x[x.sum(1)<0] = (ang_x[x.sum(1)<0] + 180)%360 
    ang_x = (360-ang_x+90)%360 #to match the axis and rotation direction for the angles calculated below
    
    ang_y = torch.atan(sy[:,0])/np.pi*180%360
    ang_y[y.sum(1)<0] = (ang_y[y.sum(1)<0] + 180)%360 
    
    ang = ang_x.clone(); ang[i] = ang_y[i]
    
    junction_angles = ang.reshape(-1,3).T
    dihedral_angles = calc_dihedral_angles(junction_angles)
    
    junction_ids = jpairs[0,:3].reshape(3,-1,3)[:,:,0]
    
    # Plot junctions with edge indices and fit lines
    if if_plot:
        
        # Plot the dihedral angle histogram
        plt.hist(dihedral_angles.flatten().cpu().numpy())
        plt.title('Dihedral historgram')
        plt.xlabel('Dihedral angle')
        plt.ylabel('Bin count')
        plt.show()
        
        # Find values for edge fit lines
        if num_plot_jct!=0:
            x = edges[...,0]
            y = edges[...,1]
            x_os = x[0][:,None]
            y_os = y[0][:,None]
            
            x_tmp = points[...,0]
            x_fit = torch.stack([torch.linspace(torch.min(x_tmp[:,k]), torch.max(x_tmp[:,k]), 100) for k in range(x_tmp.shape[1])]).to(im.device)
            ss = sx[...,0]
            y_fit = (ss[:,0:1]*x_fit + ss[:,1:2]*x_fit**2 + ss[:,2:]*x_fit**3 + y_os).T
            x_fit = (x_fit + x_os).T
            
            y_tmp = points[...,1]
            y0 = torch.stack([torch.linspace(torch.min(y_tmp[:,k]), torch.max(y_tmp[:,k]), 100) for k in range(y_tmp.shape[1])]).to(im.device)
            ss = sy[...,0]
            x0 = (ss[:,0:1]*y0 + ss[:,1:2]*y0**2 + ss[:,2:]*y0**3 + x_os).T
            y0 = (y0 + y_os).T
            
            x_fit[:,i] = x0[:,i]
            y_fit[:,i] = y0[:,i]
            
            # Plot a 'plot_num' of junctions
            h = 15 #plot radius around a junction
            num_junctions = int(y.shape[0]/3)
            while if_plot:
            
                if num_plot_jct>num_junctions: 
                    jcts=np.arange(int(y.shape[0]/3))
                else:
                    jcts = np.sort(np.random.choice(np.arange(int(y.shape[1]/3)), num_plot_jct, replace=False))
                
                for i in jcts:
                    plt.imshow(im[0,0].cpu())
                    plt.plot(y[:,i*3:(i+1)*3].cpu(), x[:,i*3:(i+1)*3].cpu(),'.')
                    plt.plot(y_fit[:,i*3:(i+1)*3].cpu(), x_fit[:,i*3:(i+1)*3].cpu(), linewidth=3)
                    plt.xlim([y[0,i*3].cpu()-h, y[0,i*3].cpu()+h+1])
                    plt.ylim([x[0,i*3].cpu()-h, x[0,i*3].cpu()+h+1])
                    plt.title('Junction: %d'%i)
                    plt.show()
                
                tmp = input('Plot more junctions (y/n)?')
                if tmp!='y': if_plot=False
            
    return torch.stack([*junction_ids, *dihedral_angles])


def find_dihedral_stats(ims, if_plot=False):
    #'ims' - shape=(num_ims, 1, dim0, dim1, dim2)
    #Works in 2D and 3D
    #3D images are sliced into 2D and the result are appended together
    
    # ims = torch.from_numpy(ims.astype(int))
    d = ims.dim() - 2 #2D or 3D?
    if d==2: ims = ims[...,None]
    
    num_ims, _, _, _, dim_size = ims.shape
    
    log_std = []
    # log_mean_max = []
    for i in range(num_ims):
        im = ims[i,0]
        log_da = [torch.ones([6,0]).to(im.device)] #preload so all functions work even when no junctions are found
        for j in range(dim_size):
            im_split = im[:,:,j][None,None]
            da = find_dihedral_angles(im_split)
            if da is not None: log_da.append(da)
        dihedral_angles = torch.cat(log_da, dim=1)[3:]
        dihedral_angles = dihedral_angles[:,torch.isnan(dihedral_angles).sum(0)==0] #remove rows with nan values
        log_std.append(torch.std(dihedral_angles.flatten()))
        # log_mean_max.append(torch.mean(dihedral_angles.max(0)[0]))
        
        if if_plot:
            plt.figure()
            bins = np.linspace(0,360,20)
            a, b = np.histogram(dihedral_angles, bins)
            plt.plot(a)
            plt.show()
    
    da_std = torch.stack(log_std)
    # da_mean_max = torch.stack(log_mean_max)
    
    return da_std#, da_mean_max





### Run PRIMME

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
    
    num_dims = im.dim()-2
    
    
    
    
    #delete later
    # windows_curr_obs = my_unfoldNd(im_next, kernel_size=energy_dim, pad_mode=pad_mode) 
    # windows_curr_obs = windows_curr_obs.reshape(1,9,62,62)[:,:,7:-7,7:-7].reshape(1,9,-1) 
    # current_energy = num_diff_neighbors_inline(windows_curr_obs)
    # windows_curr_act = my_unfoldNd(im, kernel_size=act_dim, pad_mode=pad_mode)
    # windows_next_obs = my_unfoldNd(im_next, kernel_size=energy_dim, pad_mode=pad_mode)
    # windows_next_obs = windows_next_obs.reshape(1,9,62,62)[:,:,7:-7,7:-7].reshape(1,9,-1)
    
    #NEW
    t = np.array(im.shape)
    t[1] = int(energy_dim)**num_dims
    t[2:] = t[2:] - int(energy_dim/2)*2
    t = tuple(t)
    c = int(act_dim/2)-int(energy_dim/2)
    cc = [slice(None),]*2 + [slice(c,-c),]*num_dims
    
    windows_curr_obs = my_unfoldNd(im_next, kernel_size=energy_dim, pad_mode=pad_mode) 
    windows_curr_obs = windows_curr_obs.reshape(t)[cc].reshape(t[:2]+(-1,)) 
    current_energy = num_diff_neighbors_inline(windows_curr_obs)
    windows_curr_act = my_unfoldNd(im, kernel_size=act_dim, pad_mode=pad_mode)
    windows_next_obs = my_unfoldNd(im_next, kernel_size=energy_dim, pad_mode=pad_mode)
    windows_next_obs = windows_next_obs.reshape(t)[cc].reshape(t[:2]+(-1,)) 
    
    
    
    
    
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
    energy_labels = torch.sum(energy_change*decay, dim=2).transpose(0,1).reshape((-1,)+(act_dim,)*(im_seq.dim()-2))
    
    return energy_labels


def compute_energy_labels2(im_seq, act_dim=9, energy_dim=3, pad_mode="circular"):
    window_act = my_unfoldNd(im_seq[0:1,], kernel_size=act_dim, pad_mode=pad_mode)[0]
    window_act2 = my_unfoldNd(im_seq[0:1,], kernel_size=energy_dim, pad_mode=pad_mode)[0]
    tmp = (window_act2[None,]==window_act[:,None,]).permute(2,1,0)
    energy_labels = my_batch(tmp, torch.sum, batch_sz=100).reshape(-1,17,17)/(energy_dim**2)
    return energy_labels.to(im_seq.device)


def compute_action_labels(im_seq, act_dim=9, pad_mode="circular"):
    #Label which actions in each action window were actually taken between the first image and all following
    #The total energy label is a decay sum of those action labels

    sz = im_seq.shape
    num_dims = len(sz)-2
    im = im_seq[0:1,]
    ims_next = im_seq[1:]
    
    # CALCULATE ACTION LABELS
    window_act = my_unfoldNd(im, kernel_size=act_dim, pad_mode=pad_mode)[0]
    ims_next_flat = ims_next.view(ims_next.shape[0], -1)
    
    
    
    
    #delete later
    # ims_next_flat = ims_next_flat.reshape(-1,64,64)[:,8:-8,8:-8].reshape(-1,48*48)
    
    #NEW
    c = int(act_dim/2)
    cc = [slice(None),]*1 + [slice(c,-c),]*num_dims
    ims_next_flat = ims_next_flat.reshape((-1,)+sz[2:])[cc].reshape(sz[0]-1,-1) 
    
    
    
    
    actions_marked = window_act.unsqueeze(0).expand(sz[0]-1,-1,-1)==ims_next_flat.unsqueeze(1) #Mark the actions that matches each future image (the "action taken")
    decay_rate = 1/2
    decay = decay_rate**torch.arange(1,im_seq.shape[0]).reshape(-1,1,1).to(im.device)
    action_labels = torch.sum(actions_marked*decay, dim=0).transpose(0,1).reshape((-1,)+(act_dim,)*(im_seq.dim()-2))
    
    return action_labels


def compute_labels(im_seq, obs_dim=9, act_dim=9, reg=1, pad_mode="circular"):
    energy_labels = compute_energy_labels(im_seq, act_dim=act_dim, pad_mode=pad_mode)
    action_labels = compute_action_labels(im_seq, act_dim=act_dim, pad_mode=pad_mode)
    
    # action_labels = my_normalize(action_labels)
   
    labels = action_labels + reg*energy_labels
    
    # labels = (labels+reg)/(reg+1)
    
    # labels = my_normalize(labels)
    
    return labels


def my_normalize(data):
    mi = torch.min(data)
    ma = torch.max(data)
    return (data-mi)/(ma-mi)
    

def compute_features(im, obs_dim=9, pad_mode='circular'):
    local_energy = num_diff_neighbors(im, window_size=7, pad_mode=pad_mode)
    features = my_unfoldNd(local_energy.float(), obs_dim, pad_mode=pad_mode).T.reshape((-1,)+(obs_dim,)*(im.dim()-2))
    return features





def neighborhood_miso_inline(ims_unfold, miso_matrix): 
    # ims - torch.Tensor of shape [# of images, 1, dim1, dim2, dim3(optional)]
    # miso_matrices: grain id misorientations, shape=(num_images, dim1, dim2)
    # window_size - the patch around each pixel that constitutes its neighbors
    # May need to add memory management through batches for large tensors in the future
    
    ims_unfold_miso = gid_to_miso(ims_unfold, miso_matrix)
    ims_miso = torch.sum(ims_unfold_miso, axis=1) #misorientation image
    return ims_miso #reshape to orignal image shape


def compute_action_energy_change_miso(im, im_next, miso_matrix, energy_dim=3, act_dim=9, pad_mode="circular"):
    #Calculate the energy change introduced by actions in each "im" action window
    #Energy is calculated as the number of different neighbors for each observation window
    #Find the current energy at each site in "im" observational windows
    #Finds the energy of "im_next" using observational windows with center pixels replaced with possible actions
    #The difference is the energy change
    #FUTURE WORK -> If I change how the num-neighbors function works, I could probably use expand instead of repeat
    
    num_dims = len(im.shape)-2
    
    windows_curr_obs = my_unfoldNd(im_next, kernel_size=energy_dim, pad_mode=pad_mode) 
    windows_curr_obs = windows_curr_obs.reshape(1,9,62,62)[:,:,7:-7,7:-7].reshape(1,9,-1) #!!! hardcoded
    current_energy = neighborhood_miso_inline(windows_curr_obs, miso_matrix)
    windows_curr_act = my_unfoldNd(im, kernel_size=act_dim, pad_mode=pad_mode)
    windows_next_obs = my_unfoldNd(im_next, kernel_size=energy_dim, pad_mode=pad_mode)
    windows_next_obs = windows_next_obs.reshape(1,9,62,62)[:,:,7:-7,7:-7].reshape(1,9,-1) #!!! hardcoded
    
    ll = []
    for i in range(windows_curr_act.shape[1]):
        windows_next_obs[:,int(energy_dim**num_dims/2),:] = windows_curr_act[:,i,:]
        ll.append(neighborhood_miso_inline(windows_next_obs, miso_matrix))
    action_energy = torch.cat(ll)[...,None]
    
    energy_change = (current_energy.transpose(0,1)-action_energy)/(energy_dim**num_dims*63/180*np.pi)
    
    return energy_change


def compute_energy_labels_miso(im_seq, miso_matrix, act_dim=9, pad_mode="circular"):
    #Compute the action energy change between the each image and the one immediately following
    #MAYBE CHANGE IT TO THIS IN THE FUTURE -> Compute the action energy change between the first image and all following
    #The total energy label is a decay sum of those action energy changes
    
    # CALCULATE ALL THE ACTION ENERGY CHANGES
    energy_changes = []
    for i in range(im_seq.shape[0]-1):
        ims_curr = im_seq[i].unsqueeze(0)
        ims_next = im_seq[i+1].unsqueeze(0)
        energy_change = compute_action_energy_change_miso(ims_curr, ims_next, miso_matrix, act_dim=act_dim, pad_mode=pad_mode)
        energy_changes.append(energy_change)
    
    # COMBINE THEM USING A DECAY SUM
    energy_change = torch.cat(energy_changes, dim=2)
    decay_rate = 1/2
    decay = decay_rate**torch.arange(1,im_seq.shape[0]).reshape(1,1,-1).to(im_seq.device)
    energy_labels = torch.sum(energy_change*decay, dim=2).transpose(0,1).reshape((-1,)+(act_dim,)*(im_seq.dim()-2))
    
    return energy_labels


def compute_labels_miso(im_seq, miso_matrix, obs_dim=9, act_dim=9, reg=1, pad_mode="circular"):
    
    # energy_labels = compute_energy_labels(im_seq, act_dim=act_dim, pad_mode=pad_mode)
    
    energy_labels = compute_energy_labels_miso(im_seq, miso_matrix, act_dim=act_dim, pad_mode=pad_mode)
    
    action_labels = compute_action_labels(im_seq, act_dim=act_dim, pad_mode=pad_mode)
    labels = action_labels + reg*energy_labels
    
    # labels = (labels+reg)/(2+reg) #scale from [-reg, 1+reg] to [0,1]
    
    return labels


def compute_features_miso(im, miso_matrix, obs_dim=9, pad_mode='circular'):
    local_energy = neighborhood_miso(im, miso_matrix, window_size=7, pad_mode=pad_mode)
    # local_energy = neighborhood_miso_spparks(im, miso_matrix, window_size=7, pad_mode=pad_mode)
    features = my_unfoldNd(local_energy.float(), obs_dim, pad_mode=pad_mode).T.reshape((-1,)+(obs_dim,)*(im.dim()-2))
    
    return features