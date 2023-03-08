#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 13:58:08 2023

@author: joseph.melville
"""


import numpy as np
import torch
import matplotlib.pyplot as plt
import functions as fs
import h5py
from tqdm import tqdm


# Functions
def count_occurance(arrays):
    counts = (arrays[:,None,:]==arrays[None,:,:]).sum(0)
    return counts


def count_energy(arrays, miso_matrix, cut):
    
    # Cutoff and normalize misorientation matrix (for cubic symetry, cut=63 degrees effectively just normalizes)
    if cut==0: cut = 1e-10
    cut_rad = cut/180*np.pi
    miso_mat_norm = miso_matrix/cut_rad
    miso_mat_norm[miso_mat_norm>1] = 1
    
    #Mark where neighbor IDs do not match
    diff_matricies = (arrays[:,None,:]!=arrays[None,:,:]).float()

    #Find the indicies of ones
    i, j, k = torch.where(diff_matricies)
    
    #Find the ids of each of those indices
    i2 = arrays[i,k].long()
    j2 = arrays[j,k].long()
    
    #Find the misorientations of the id pairs
    f = miso_mat_norm[i2,j2].float()
    
    #Place misorientations in place of all the ones
    diff_matricies[i,j,k] = f
    
    #Invert and sum
    energy = torch.sum(1-diff_matricies, dim=0)

    return energy


def find_mode(arrays, miso_matrix, cut):
    #Takes the mode of the array using torch.Tensor.cuda
    
    if cut==0: counts = count_occurance(arrays) #counts the number of occurances for each value
    else: counts = count_energy(arrays, miso_matrix, cut) #counts the energy value
    i = torch.argmax(counts, dim=0)[None,] #find the indices of the max
    mode = torch.gather(arrays, dim=0, index=i)[0] #selects those indices
    return mode


def sample_cumsum(arrays):
    #"array" - shape=(array elements, number of arrays)
    #Chooses an index from each column in "array" by sampling from it's cumsum
    arrays_cumsum = torch.cumsum(arrays.T, dim=1)/torch.sum(arrays, dim=0)[:,None]
    sample_values = torch.rand(arrays_cumsum.shape[0]).to(arrays.device)
    sample_indices = torch.argmax((arrays_cumsum>sample_values.unsqueeze(1)).float(), dim=1)
    return sample_indices


def sample_counts(arrays, miso_matrix, cut):
    if cut==0: counts = count_occurance(arrays) #counts the number of occurances for each value
    else: counts = count_energy(arrays, miso_matrix, cut) #counts the energy value
    index = sample_cumsum(counts)[None,] #use this if you want to sample from the counts instead of choosing the max
    return torch.gather(arrays, dim=0, index=index)[0] #selects those indices


def normal_mode_filter(im, miso_matrix, cut=0, cov=25, num_samples=64, bcs='p', memory_limit=1e9):
    
    # Find constants
    s = list(im.shape) #images shape
    d = len(s) #number of dimensions
    e = np.prod(s)
    
    # Create covariance matrix
    if type(cov)==int: cov=torch.eye(d)*cov
    
    # Set boundary conditions [x,y,z]
    if type(bcs)==str: bcs=[bcs]*d
    
    # Create sampler
    cov = cov.to(im.device)
    mean_arr = torch.zeros(d).to(im.device)
    mvn = torch.distributions.MultivariateNormal(mean_arr, cov)
    
    # Calculate the index coords
    ii = [torch.arange(ss).to(im.device) for ss in s]
    coords = torch.cartesian_prod(*ii).float().transpose(0,1).reshape(1, d, -1)
    
    
    # Calculate neighborhood modes by batch
    mem_total = 2*num_samples*d*num_samples*e*64/8
    num_batches = mem_total/memory_limit
    batch_size = int(e/num_batches)
    l = []
    coords_split = torch.split(coords, batch_size, dim=2)
    for c in coords_split: 
        
        # Sample neighborhoods
        samples = mvn.sample((num_samples, c.shape[2])).int().transpose(1,2) #sample separately for each site
        samples = torch.cat([samples, samples*-1], dim=0).to(im.device) #mirror the samples to keep a zero mean
        c = samples + c
        
        # Set bounds for the indices 
        for i in range(d): #remember - periodic x values, makes the y axis periodic, or the y-z plane periodic
            if bcs[i]=='p': c[:,i,:] = c[:,i,:]%s[0] #wrap (periodic)
            else: c[:,i,:] = torch.clamp(c[:,i,:], min=0, max=s[0]-1) #or clamp
        
        #Gather the coord values and take the mode for each pixel (replace this with the matrix approach)   
        ii = [c[:,i,:].long() for i in range(d)]
        im_sampled = im[ii]
        im_next_part = find_mode(im_sampled, miso_matrix, cut)
        l.append(im_next_part)
        
    im_next = torch.hstack(l).reshape(s)
    
    return im_next


def run_mf(ic, ea, nsteps, cut, cov=25, num_samples=64, miso_array=None, if_plot=False, bcs='p', memory_limit=1e9, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    #run mode filter simulation
    
    # Setup
    im = torch.Tensor(ic).float().to(device)
    ngrain = len(torch.unique(im))
    tmp = np.array([8,16,32], dtype='uint64')
    dtype = 'uint' + str(tmp[np.sum(ngrain>2**tmp)])
    if np.all(miso_array==None): miso_array = torch.Tensor(fs.find_misorientation(ea, mem_max=1) )
    miso_matrix = fs.miso_array_to_matrix(miso_array[None,]).to(device)[0]
    if type(cov)==int: cov_rep = cov
    else: cov_rep = cov[0]
    sz_str = ''.join(['%dx'%i for i in ic.shape])[:-1]
    fp_save = './data/mf_sz(%s)_ng(%d)_nsteps(%d)_cov(%d)_numnei(%d)_cut(%d).h5'%(sz_str, ngrain, nsteps, cov_rep, num_samples, cut)
    
    # Run simulation
    log = [im.clone()]
    for i in tqdm(range(nsteps), 'Running MF simulation:'): 
        im = normal_mode_filter(im, miso_matrix, cut, cov, num_samples, bcs, memory_limit=memory_limit)
        log.append(im.clone())
        if if_plot: plt.imshow(im[0,0,].cpu()); plt.show()
    
    ims_id = torch.stack(log)[:,None,].cpu().numpy()
    
    # Save Simulation
    with h5py.File(fp_save, 'a') as f:
        
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
        dset4[:] = miso_matrix.cpu() #same values as mis0_array, different format

    return ims_id, fp_save


def image_covariance_matrix(im, min_max=[-200, 200], num_samples=8, bounds=['wrap','wrap']):
    #Sample and calculate the index coords
    mvn = torch.distributions.Uniform(torch.Tensor([min_max[0]]).to(im.device), torch.Tensor([min_max[1]]).to(im.device))
    samples = mvn.sample((num_samples, 2, im.numel()))[...,0].int()
    
    arr0 = torch.arange(im.shape[0]).to(im.device)
    arr1 = torch.arange(im.shape[1]).to(im.device)
    coords = torch.cartesian_prod(arr0, arr1).float().transpose(0,1).reshape(1, 2, -1)
    coords = samples+coords
    
    #Set bounds for the indices
    if bounds[1]=='wrap': coords[:,0,:] = coords[:,0,:]%im.shape[0]
    else: coords[:,0,:] = torch.clamp(coords[:,0,:], min=0, max=im.shape[0]-1)
    if bounds[0]=='wrap': coords[:,1,:] = coords[:,1,:]%im.shape[1]
    else: coords[:,1,:] = torch.clamp(coords[:,1,:], min=0, max=im.shape[1]-1)
    
    #Flatten indices
    index = (coords[:,1,:]+im.shape[1]*coords[:,0,:]).long()
        
    # #Gather the coord values and take the mode for each pixel      
    # im_expand = im.reshape(-1,1).expand(-1, im.numel())
    # v = torch.gather(im_expand, dim=0, index=index)
    # im_next = torch.mode(v.cpu(), dim=0).values.reshape(im.shape)
    # im_next = im_next.to(device)
    # # im_next = fs.rand_mode(v).reshape(im.shape)
    
    #Find the covariance matrix of just the samples that equal the mode of the samples
    index_mode = im.reshape(-1)[index]==im.reshape(-1)
    samples_mode = samples.transpose(1,2)[index_mode].transpose(1,0).cpu().numpy()
    # plt.plot(samples_mode[0,:1000], samples_mode[1,:1000], '.'); plt.show()
    
    return np.cov(samples_mode)


def find_sample_coords(im, cov=torch.Tensor([[25,0],[0,25]]), num_samples=64, bcs=['p','p']):
    
    #Create sampler
    cov = cov.to(im.device)
    mean_arr = torch.zeros(2).to(im.device)
    mvn = torch.distributions.MultivariateNormal(mean_arr, cov)
    
    #Calculate the index coords
    arr0 = torch.arange(im.shape[0]).to(im.device)
    arr1 = torch.arange(im.shape[1]).to(im.device)
    coords = torch.cartesian_prod(arr0, arr1).float().transpose(0,1).reshape(1, 2, -1)
    
    samples = mvn.sample((num_samples, coords.shape[2])).int().transpose(1,2) #sample separately for each site
    samples = torch.cat([samples, samples*-1], dim=0).to(im.device) #mirror the samples to keep a zero mean
    c = samples + coords #shifted samples
    
    #Set bounds for the indices (wrap or clamp - add a reflect)
    if bcs[1]=='p': c[:,0,:] = c[:,0,:]%im.shape[0] #periodic or wrap
    else: c[:,0,:] = torch.clamp(c[:,0,:], min=0, max=im.shape[0]-1)
    if bcs[0]=='p': c[:,1,:] = c[:,1,:]%im.shape[1] #periodic or wrap
    else: c[:,1,:] = torch.clamp(c[:,1,:], min=0, max=im.shape[1]-1)
    
    #Flatten indices
    index = (c[:,1,:]+im.shape[1]*c[:,0,:]).long()
    
    return coords, samples, index


def find_pf_matrix(id_ratios):
    #finds pair-factor (pf) matrix
    #pf matrix relates ID pair-factors to ID probability of adoption using ID ratios
    
    #Find reference indicies to create the matrix
    num_ids = len(id_ratios)
    num_pf = int(num_ids*(num_ids-1)/2) #number of pair-factors (combinations of IDs)
    i, j = torch.where(torch.triu(torch.ones(num_ids, num_ids), 1))
    k = torch.arange(num_pf)
    
    #Place ID ratios in matrix
    A = torch.zeros(num_ids, num_pf)
    A[i,k] = id_ratios[j]
    A[j,k] = id_ratios[i]
    
    # #Convert from argmin (energy) to argmax (probability)
    # h = torch.arange(num_ids)
    # Ap = (id_ratios[i]+id_ratios[j])[None,].repeat(num_ids, 1)
    # Ap[h[:,None]==i[None,]] = id_ratios[i]
    # Ap[h[:,None]==j[None,]] = id_ratios[j]
    
    #Create B matrix (for normalization)
    B = (id_ratios[i] + id_ratios[j]) / (len(id_ratios)-1)
    
    #Record order of pair-factor location indices
    pf_loc = torch.stack([i,j])
    
    return A, B, pf_loc


def find_radial_max(rs, c, v):
    vt = v-c #transform to new center
    rt = rs-c #transform to new center
    dot = (vt*rt).sum(0)
    mag_mult = torch.sqrt((rt**2).sum(0))*torch.sqrt((vt**2).sum(0))
    tmp = dot/mag_mult
    tmp[tmp>1]=1; tmp[tmp<-1]=-1
    angles = torch.acos(tmp)/np.pi*180
    
    i_max = torch.argmax(angles) #location of the ratio with the largest angles
    a_max = angles[i_max] #largest angle
    
    # a_min = 60
    # #If a_max is less than a_min, then the center is outside the triangle
    # if a_max<a_min:
    #     return np.nan, np.nan #needs to be solved later
    
    #Convert rt_max to a unit vector from c, then add c to return to original space
    rt_max = rt[:,i_max] #ratios with largest angle
    rt_max_unit = rt_max/torch.sqrt(torch.matmul(rt_max, rt_max.T))
    r_max_unit = rt_max_unit + c[:,0]
    
    return r_max_unit, a_max


def estimate_angles(ratios, outcomes):
    #"ratios" - torch.Tensor, shape=(number of grains, number of neighborhoods), ratios found in each neighborhood
    #"outcomes" - torch.Tensor, shape=(number of grains, number of neighborhoods), which ratio was chosen to flip to
    
    d = ratios.shape[0]
    ij = torch.combinations(torch.arange(d), r=2)
    e = (torch.ones(d)/d)[:,None] #intersecting point when all factors are equal
    e = e/torch.norm(e)
    
    midpoints = []
    angles = []
    for i, j in ij: #for each grain ID pair
    
        #New center
        c = torch.zeros(d)[:,None] 
        c[i] = 0.5
        c[j] = 0.5
        
        #Find max angle from grain IDs i to j
        v = torch.zeros(d)[:,None]; v[i] = 1 #find angle from this vector
        rs = ratios[:,outcomes[i]==1] #for these ratios
        rmi, ami = find_radial_max(rs, c, v) #angle max from i
        # rmi = (v-c)*np.cos(ami/180*np.pi) + (e-c)*np.sin(ami/180*np.pi) + c
        
        #Find max angle from grain j to i
        v = torch.zeros(d)[:,None]; v[j] = 1 #find angle from this vector
        rs = ratios[:,outcomes[j]==1] #for these ratios
        rmj, amj = find_radial_max(rs, c, v) #angle max from j
        # rmj = (v-c)*np.cos(amj/180*np.pi) + (e-c)*np.sin(amj/180*np.pi) + c
        
        #Find a point in the dividing line
        midpoints.append((rmi+rmj)/2)
        
        #Average angles
        angles.append((ami+180-amj)/2)
        
    #Convert to Tensors
    angles = torch.stack(angles)[:,None]
    midpoints = torch.stack(midpoints)
    
    if ratios.shape[0]==3:
        #Replace invalid midpoints
        is_valid = is_valid_3grains(ratios, outcomes)
        midpoints = replace_invalid_3grain(midpoints, is_valid, ij)
    else:
        is_valid = torch.ones(midpoints.shape[0])==1
    
    return angles, midpoints, ij, is_valid


def is_valid_3grains(ratios, outcomes):
    #If the center of is outside the triangle, return the invalid edge
    #"ratios" are the neighborhood ratios for each grain - shape=(3,number of neighborhoods)
    #"outcomes" are the grain chosen (lowest energy) - shape=(number of neighborhoods, 3)
    #Currently only works for three grain systems
    ii = (ratios>0.5).sum(0)==1 #for all ratios with one greater than 0.5
    nn = ((ratios[:,ii]>0.5)<outcomes[:,ii]).sum(1) #number of each grain in restricted a region
    is_valid = nn.flip((0))==0 #where 0->[01], 1->[02], 2->[12]
    return is_valid


def replace_invalid_3grain(midpoints, i_valid, ij):
    #"midpoints" - shape=(number of points or ID pairs, number of dimensions)
    #"i_valid" - shape=(number of points or ID pairs)
    #"ij" - shape=(number of points or ID pairs, 2)
    i, j = ij[i_valid,:][0,]
    p00 = torch.zeros(d) 
    p00[i] = 0.5
    p00[j] = 0.5
    
    p01 = midpoints[i_valid,][0,]
    
    i, j = ij[i_valid,:][1,]
    p10 = torch.zeros(d)
    p10[i] = 0.5
    p10[j] = 0.5
    
    p11 = midpoints[i_valid,][1,]
    
    tmp = point_line_intercept(p00,p01,p10,p11)
    
    midpoints[i_valid==False] = tmp

    return midpoints


def affine_projection(r, d0=0, d1=1):
    d = r.shape[0]
    
    n = torch.ones(d,1)
    
    tmp = torch.linalg.inv(torch.matmul(n.T,n))
    P = torch.matmul(n*tmp,n.T)
    
    Q = torch.eye(d) - P
    
    t = torch.zeros(d,1)
    t[0] = 1
    
    q0 = torch.cat([torch.cat([torch.eye(d), t], dim=1), torch.cat([torch.zeros(1,d), torch.ones(1,1)], dim=1)])
    q1 = torch.cat([torch.cat([Q, torch.zeros(d,1)], dim=1), torch.cat([torch.zeros(1,d), torch.ones(1,1)], dim=1)])
    q2 = torch.cat([torch.cat([torch.eye(d), -t], dim=1), torch.cat([torch.zeros(1,d), torch.ones(1,1)], dim=1)])
    QQ = torch.matmul(torch.matmul(q0,q1),q2)
    
    t0 = torch.zeros(d+1); t0[d0]=-1; t0[d1]=1
    t1 = torch.ones(d+1); t1[d]=0; t1[d0]=0; t1[d1]=0
    T = torch.stack([t0,t1])
    
    TQ = torch.matmul(T,QQ)
    
    rhat = torch.matmul(TQ,torch.cat([r,torch.ones(1,r.shape[1])]))
    return rhat


def point_vector_angle(c, p0, p1):
    #all inputs are torch.Tensor
    #"c" is center point - shape=(number of dimenisons)
    #"p0" is the point in the direction of zero degrees - shape=(number of dimenisons)
    #"p1" are the points to find an angle to - shape=(number of dimenisons, number of points)
    #outputs the angle between the vectors in degrees
    if len(p1.shape)<2: p1 = p1[:,None]
    v0 = (p0-c)
    v1 = p1-c[:,None]
    tmp = torch.linalg.multi_dot([v0,v1])/(torch.norm(v0)*torch.norm(v1,dim=0))
    return torch.acos(tmp)/np.pi*180


def point_line_intercept(p00,p01,p10,p11):
    #finds the intercept between the line p00 to p01 and the line p10 to p11
    #all inputs are torch.Tensor - shape=(number of dimensions)
    #can only find one intercept right now
    b = (p00-p10)[:,None]
    A = torch.stack([p01-p00, p10-p11]).T
    A_inv = torch.linalg.pinv(A)
    s = torch.matmul(A_inv,b)[0]
    return (p00-p01)*s+p00


def find_intersect_lines(midpoints):
    #'midpoints' - list of poin on the line seperating the cooresponding ID pairs indicated in 'ij'
    #'intersections' - a matrix of lines that divide the ID pairs indicated in 'ij'
    #Works for arbitrarily large grain systems (may not be true actually, sometimes it's weird with more than three grains)
    #Can be optimized somewhat by only generating the two rows needed from 'find_pf_matrix'
    
    d = midpoints.shape[1]
    ij = torch.combinations(torch.arange(d), r=2)
    
    l = []
    for k, (i, j) in enumerate(ij):
        A, _, _ = find_pf_matrix(midpoints[k]) #find the system of equations for this ratio set
        tmp = A[j,:]-A[i,:] #find the difference of the rows indicated in 'ij'
        l.append(tmp)
    intersections = torch.stack(l)
    
    return intersections
    
    
def find_pair_factor_ratios(intersections):
    #Sets the first pair factor to one and solves for the rest
    #The exact pair factors are not neccesarily found, but the ratios between them are the same
    
    b = -intersections[:,0]
    A = intersections[:,1:]
    x_partial = torch.matmul(torch.linalg.pinv(A),b)
    
    x = torch.ones(intersections.shape[0])
    x[1:] = x_partial
    
    return x


def plot_ratios(rr, p, i=0, j=1):
    #'rr' - sets of ratios, torch.Tensor, shape=(num_IDs, num neighborhood)
    #'p' - ID labels for each ratio, torch.Tensor, num neighborhood
    #'i' and 'j' are the IDs of the projection plane when more than three IDs
    
    if rr.shape[0]==3:
        i = p[0]==1
        j = p[1]==1
        k = p[2]==1
        i0, j0, k0 = rr[:,i]
        i1, j1, k1 = rr[:,j]
        i2, j2, k2 = rr[:,k]
        
        ax = plt.axes(projection='3d')
        ax.scatter3D(i0, j0, k0, depthshade=0, cmap='Blues')
        ax.scatter3D(i1, j1, k1, depthshade=0, cmap='Oranges')
        ax.scatter3D(i2, j2, k2, depthshade=0, cmap='Greens')
        ax.set_xlim([1,0])
        ax.set_ylim([0,1])
        ax.set_zlim([1,0])
        ax.set_xlabel('r0')
        ax.set_ylabel('r1')
        ax.set_zlabel('r2')
    else:
        x, y = affine_projection(rr, i, j)
        for i in [i,j]:#range(p.shape[1]):
            ii = p[i]==1
            plt.scatter(x[ii], y[ii])
            plt.xlim([-1,1])
            plt.ylim([0,1])
         

def find_avg_intersect(midpoints, is_valid):
    #Find thes intercepts between the lines defined by 'midpoints', ignoring points that are not 'is_valid'
    #'midpoints' - torch.Tensor, shape=(num points, num IDs)
    #'is_valid' - torch.Tensor, shape=(num points), which points are valid

    d = midpoints.shape[1]
    ij = torch.combinations(torch.arange(d), r=2)
    ij = ij[is_valid]
    midpoints = midpoints[is_valid]
    
    #Find new origins for each ID pair
    l = []
    for i,j in ij:
        tmp = torch.zeros(d) 
        tmp[i] = 0.5
        tmp[j] = 0.5
        l.append(tmp)
    origins = torch.stack(l)
    
    #Find intercepts for each line pair
    hk = torch.combinations(torch.arange(ij.shape[0]), r=2)
    l = []
    for h,k in hk:
        l.append(point_line_intercept(origins[h], midpoints[h], origins[k], midpoints[k]))
    
    #Take average
    avg_intercept = sum(l)/len(l)
    
    return avg_intercept



### MAIN


fp = '../primme_share/PRIMME/data/primme_sz(128x128x128)_ng(8192)_nsteps(1000)_freq(1)_kt(0.66)_cut(0).h5'
with h5py.File(fp, 'r') as f:
    ic = f['sim0/ims_id'][0,0].astype('int')
    ea = f['sim0/euler_angles'][:].astype('int')

ims, fp_save = run_mf(ic, ea, nsteps=1000, cut=0, cov=3, num_samples=64)
fs.compute_grain_stats(fp_save)
fs.make_time_plots(fp_save)

fs.make_time_plots(fp)




hps = ['./data/mf_sz(128x128x128)_ng(4096)_nsteps(1000)_cov(25)_numnei(64)_cut(0).h5',
       './data/mf_sz(128x128x128)_ng(4096)_nsteps(1000)_cov(25)_numnei(64)_cut(25).h5']








plotly_micro(im)



with h5py.File(hps[0], 'r') as f:
    print(f.keys())
    print(f['sim0'].keys())
    im = f['sim0/ims_id'][0,0,].astype('int')
    ic = f['sim0/ims_id'][0,0,].astype('int')
    ea = f['sim0/euler_angles'][:]


ic, ea, _ = fs.voronoi2image(size=[128,]*3, ngrain=4096)    
ims = run_mf(ic, ea, nsteps=1000, cut=0, cov=25, num_samples=64)
ims = run_mf(ic, ea, nsteps=1000, cut=25, cov=25, num_samples=64)

hps = ['./data/mf_sz(128x128x128)_ng(4096)_nsteps(1000)_cov(25)_numnei(64)_cut(0).h5',
       './data/mf_sz(128x128x128)_ng(4096)_nsteps(1000)_cov(25)_numnei(64)_cut(25).h5']
fs.compute_grain_stats(hps)
# fs.make_videos(hps) 
fs.make_time_plots(hps)


fp = './data/mf_sz(128x128x128)_ng(8192)_nsteps(1000)_cov(3)_numnei(64)_cut(0).h5'
fs.make_time_plots(fp)



ic, ea, _ = fs.voronoi2image(size=[1024,]*2, ngrain=4096)  

ims = run_mf(ic, ea, nsteps=10, cut=0, cov=25, num_samples=64)


plt.imshow(ic)

hps = ['./data/mf_sz(1024x1024)_ng(4096)_nsteps(1000)_cov(25)_numnei(64)_cut(0).h5']
fs.compute_grain_stats(hps)
fs.make_videos(hps) 
fs.make_time_plots(hps)

#12
#37
#26
#65




fs.create_3D_paraview_vtr(ic)























# Create a set of ratios (assuming three grains)
d = 3
m = 5000
rr = torch.rand(d, m)
rr = rr/rr.sum(0)

# Set the pair factors
num_pairs = np.arange(1,d).sum()
xt = torch.ones(num_pairs)
# xt = torch.rand(num_pairs)/2+0.5
xt = torch.rand(num_pairs)

#Find all the argmin outputs and plot them by r values 
l = []
for ii in tqdm(range(rr.shape[1])):
    r = rr[:,ii]
    A, _, _ = find_pf_matrix(r)
    b = torch.matmul(A, xt)
    bb = ((b-torch.min(b))==0).float()
    l.append(bb)
    
p = torch.stack(l).T
plot_ratios(rr, p, i=0, j=2)



angles, midpoints, ij, is_valid = estimate_angles(rr, p)
inters = find_intersect_lines(midpoints)
x = find_pair_factor_ratios(inters)
print(x*xt[0])
print(xt)
print(xt-x*xt[0])