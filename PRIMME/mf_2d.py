#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 10:30:51 2022

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


def normal_mode_filter(im, miso_matrix, cut=0, cov_mat=torch.Tensor([[25,0],[0,25]]), num_samples=64, bcs=['p','p'], memory_limit=1e9):
    
    #Create sampler
    cov_mat = cov_mat.to(im.device)
    mean_arr = torch.zeros(2).to(im.device)
    mvn = torch.distributions.MultivariateNormal(mean_arr, cov_mat)
    
    #Calculate the index coords
    arr0 = torch.arange(im.shape[0]).to(im.device)
    arr1 = torch.arange(im.shape[1]).to(im.device)
    coords = torch.cartesian_prod(arr0, arr1).float().transpose(0,1).reshape(1, 2, -1)
    
    num_batches = 2*num_samples*2*num_samples*im.numel()*64/8/memory_limit
    batch_size = int(im.numel()/num_batches)
    # batch_size = int(im.numel()/10)
    l = []
    coords_split = torch.split(coords, batch_size, dim=2)
    for c in coords_split: 
        
        #Sample neighborhoods
        samples = mvn.sample((num_samples, c.shape[2])).int().transpose(1,2) #sample separately for each site
        samples = torch.cat([samples, samples*-1], dim=0).to(im.device) #mirror the samples to keep a zero mean
        c = samples + c
    
        #Set bounds for the indices (wrap or clamp - add a reflect)
        if bcs[1]=='p': c[:,0,:] = c[:,0,:]%im.shape[0] #periodic or wrap
        else: c[:,0,:] = torch.clamp(c[:,0,:], min=0, max=im.shape[0]-1)
        if bcs[0]=='p': c[:,1,:] = c[:,1,:]%im.shape[1] #periodic or wrap
        else: c[:,1,:] = torch.clamp(c[:,1,:], min=0, max=im.shape[1]-1)
    
        #Flatten indices
        index = (c[:,1,:]+im.shape[1]*c[:,0,:]).long()
            
        #Gather the coord values and take the mode for each pixel (replace this with the matrix approach)   
        im_expand = im.reshape(-1,1).expand(-1, batch_size)
        im_sampled = torch.gather(im_expand, dim=0, index=index)
        # im_part = sample_counts(im_sampled, miso_matrix, cut)
        im_part = find_mode(im_sampled, miso_matrix, cut)
        l.append(im_part)
        
    im_next = torch.hstack(l).reshape(im.shape)
    
    return im_next


def run_mf(ic, ea, nsteps, cut, cov_mat=torch.Tensor([[25,0],[0,25]]), num_samples=64, miso_array=None, if_plot=False, bcs=['p','p'], memory_limit=1e9, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    #run mode filter simulation
    
    # Setup
    im = torch.Tensor(ic).float().to(device)
    size = ic.shape
    ngrain = len(torch.unique(im))
    tmp = np.array([8,16,32], dtype='uint64')
    dtype = 'uint' + str(tmp[np.sum(ngrain>2**tmp)])
    if np.all(miso_array==None): miso_array = torch.Tensor(fs.find_misorientation(ea, mem_max=1) )
    miso_matrix = fs.miso_conversion(miso_array[None,]).to(device)[0]
    fp_save = './data/mf_sz(%dx%d)_ng(%d)_nsteps(%d)_cov(%d-%d-%d)_numnei(%d)_cut(%d).h5'%(size[0],size[1],ngrain,nsteps,cov_mat[0,0],cov_mat[1,1],cov_mat[0,1], num_samples, cut)
    
    # Run simulation
    log = [im.clone()]
    for i in tqdm(range(nsteps), 'Running MF simulation:'): 
        im = normal_mode_filter(im, miso_matrix, cut, cov_mat, num_samples, bcs, memory_limit=memory_limit)
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

    return ims_id


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


def find_sample_coords(im, cov_mat=torch.Tensor([[25,0],[0,25]]), num_samples=64, bcs=['p','p']):
    
    #Create sampler
    cov_mat = cov_mat.to(im.device)
    mean_arr = torch.zeros(2).to(im.device)
    mvn = torch.distributions.MultivariateNormal(mean_arr, cov_mat)
    
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












#run a mode filter with certain f values



ic, ea, _ = fs.voronoi2image(size=[64, 64], ngrain=6) 
ims_id = run_mf(ic, ea, nsteps=100, cut=0, cov_mat=torch.Tensor([[3,0],[0,3]]), num_samples=64, memory_limit=1e10)

im = torch.from_numpy(ims_id[0,0])
im2 = torch.from_numpy(ims_id[1,0])

coords, samples, index = find_sample_coords(im, cov_mat=torch.Tensor([[3,0],[0,3]]), num_samples=64)
 
im_expand = im.reshape(-1,1).expand(-1, im.numel())
im_sampled = torch.gather(im_expand, dim=0, index=index)
outcomes = im2.reshape(-1)






#set up for estimation

log_ids = torch.Tensor(0,3)
log_r = torch.Tensor(0,3)
log_o = torch.Tensor(0,3)
log_i = torch.Tensor(0)
log_j = torch.Tensor(0)
for i in range(im_sampled.shape[1]):
    
    ids, c = im_sampled[:,i].unique(return_counts=True)
    
    (ids==outcomes[i]).int()
    
    
    if len(ids)==3:
        log_r = torch.cat([log_r, (c/c.sum())[None,]])
        log_o = torch.cat([log_o, (ids==outcomes[i]).int()[None,]])
        matches = (log_ids==ids).all(1)
        if matches.any():
            j = matches.int().argmax()
        else:
            log_ids = torch.cat([log_ids, ids[None,:]])
            j = log_ids.shape[0]-1
        log_i = torch.cat([log_i, torch.Tensor([i])])
        log_j = torch.cat([log_j, torch.Tensor([j])])

c = []
rs = []
ps = []
for j in range(len(log_ids)):
    
    tmp = torch.Tensor([[1,0,0],[0,1,0],[0,0,1]])
    
    rr = log_r[log_j==j].T
    rr = torch.cat([tmp,rr],dim=1)
    
    p = log_o[log_j==j].T
    p = torch.cat([tmp,p],dim=1)
    
    rs.append(rr)
    ps.append(p)
    c.append(p.shape[1])
    


i=0
plot_ratios(rs[i], ps[i])
i+=1







#estimate ratios
#I need to stack all of them together and see what I can get
#I need to deal with noise, which I didn't expect

i=0
ids = log_ids[i]

ij = torch.combinations(ids, r=2)

rr = rs[i]
p = ps[i]

angles, midpoints, ij, is_valid = estimate_angles(rr, p)
inters = find_intersect_lines(midpoints)
x = find_pair_factor_ratios(inters)
print(x)
print(xt)
print(xt-x*xt[0])































#project onto a single plane



rr2 = rr-torch.Tensor([0.5,0,0.5])[:,None]



# rr2 = torch.Tensor([0,1,0])[:,None]-torch.Tensor([0.5,0,0.5])[:,None]
rr2 = rr2*np.sqrt(2)

cs = np.cos(45/180*np.pi)
sn = np.sin(45/180*np.pi)
rmy = torch.Tensor([[cs,0,sn],[0,1,0],[-sn,0,cs]])

rr2 = torch.matmul(rmy,rr2)

a = -90+np.arcsin(2/np.sqrt(6))/np.pi*180
cs = np.cos(a/180*np.pi)
sn = np.sin(a/180*np.pi)
rmz = torch.Tensor([[cs,-sn,0],[sn,cs,0],[0,0,1]])

rr2 = torch.matmul(rmz,rr2)


plt.scatter(rr2[2], rr2[1])



a = 45/180*np.pi
u = torch.ones(3)/np.sqrt(3)


#convert in affine space
#find angle, then point on angle
#convert point back to normal space










i = p[:,0]==1
j = p[:,1]==1
k = p[:,2]==1
i0, j0, k0 = rr2[:,i]
i1, j1, k1 = rr2[:,j]
i2, j2, k2 = rr2[:,k]

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(i0, j0, k0, depthshade=0, cmap='Blues')
ax.scatter3D(i1, j1, k1, depthshade=0, cmap='Oranges')
ax.scatter3D(i2, j2, k2, depthshade=0, cmap='Greens')
ax.set_xlabel('r0')
ax.set_ylabel('r1')
ax.set_zlabel('r2')

plt.scatter(rr2[0,:], rr2[2,:])






angles = estimate_angles(rr, p.T)

























#Estimate the center point
ii = torch.argmin(i0)
p1 = rr[:,i][:,ii]

ii = torch.argmin(j1)
p1 += rr[:,j][:,ii]

ii = torch.argmin(k2)
p1 += rr[:,k][:,ii]

p1 = p1/3



# plt.scatter(i0,j0)
plt.scatter(i1,j1)
plt.scatter(i2,j2)
plt.xlim([0,1])
plt.ylim([0,1])

(p1[1]-1/2)/p1[0]
(xt[0]-xt[1]-xt[2])/(2*xt[2])

s = (xt[0]-xt[1]-xt[2])/(2*xt[2])
(j1 < 0.5 + s*i1).sum()
(j2 > 0.5 + s*i2).sum()


plt.scatter(j0,i0)
# plt.scatter(j1,i1)
plt.scatter(j2,i2)
plt.xlim([0,1])
plt.ylim([0,1])

(p1[0]-1/2)/p1[1]
(xt[0]-xt[2]-xt[1])/(2*xt[1])

s = (xt[0]-xt[2]-xt[1])/(2*xt[1])
(i0 < 0.5 + s*j0).sum()
(i2 > 0.5 + s*j2).sum()


plt.scatter(k0,j0)
plt.scatter(k1,j1)
# plt.scatter(k2,j2)
plt.xlim([0,1])
plt.ylim([0,1])

(p1[1]-1/2)/p1[2]
(xt[2]-xt[1]-xt[0])/(2*xt[0])

s = (xt[2]-xt[1]-xt[0])/(2*xt[0])
(j0 > 0.5 + s*k0).sum()
(j1 < 0.5 + s*k1).sum()











x_train = torch.stack([torch.cat([i1,i2]), torch.cat([j1,j2])])
y_train = torch.cat([torch.zeros(len(i1)), torch.ones(len(i2))])
(xt[0]-xt[1]-xt[2])/(2*xt[2])

x_train = torch.stack([torch.cat([j0,j2]), torch.cat([i0,i2])])
y_train = torch.cat([torch.zeros(len(j0)), torch.ones(len(j2))])
(xt[0]-xt[2]-xt[1])/(2*xt[1])

x_train = torch.stack([torch.cat([k0,k1]), torch.cat([j0,j1])])
y_train = torch.cat([torch.zeros(len(k0)), torch.ones(len(k1))])
(xt[2]-xt[1]-xt[0])/(2*xt[0])


from sklearn import svm
clf = svm.SVC(kernel='linear')
clf.fit(x_train.T, y_train)

w = clf.coef_[0]
it = clf.intercept_

a = -w[0]/w[1]
b = -it[0]/w[1]

print(a)
print(b)


#it's alright, though not as good as the argmins
#Would like to set y-intercept











ii = torch.argmax(i1)
p1 = rr[:,j][:,ii]
ax.scatter3D(p1[0], p1[1], p1[2], c='k', s=100, zorder=100)

ii = torch.argmin(j1)
p1 = rr[:,j][:,ii]
ax.scatter3D(p1[0], p1[1], p1[2], c='k', s=100, zorder=100)

ii = torch.argmax(k1)
p1 = rr[:,j][:,ii]
ax.scatter3D(p1[0], p1[1], p1[2], c='k', s=100, zorder=100)














ii = torch.argmin(i0)
i0[ii]
rr[:,i][:,ii]
p1 = rr[:,i][:,ii]
print(p1)

ii = torch.argmin(j1)
j1[ii]
rr[:,j][:,ii]
p1 = rr[:,j][:,ii]
print(p1)

ii = torch.argmin(k2)
k2[ii]
rr[:,k][:,ii]
p1 = rr[:,k][:,ii]
print(p1)








print((xt[2]-xt[1])/xt[0])
print((p1[1]-p1[0])/p1[2])

print((xt[0]-xt[2])/xt[1])
print((p1[0]-p1[2])/p1[1])

print((xt[0]-xt[1])/xt[2])
print((p1[1]-p1[2])/p1[0])









#Line from x=0
p0 = torch.Tensor([0, 0.5, 0.5])
pd = p1-p0

#z=0
p0[0]-p0[2]*pd[0]/pd[2] #x
p0[1]-p0[2]*pd[1]/pd[2] #y

#y=0
p0[0]-p0[1]*pd[0]/pd[1] #x
p0[2]-p0[1]*pd[2]/pd[1] #z

#x=0
p0[1]-p0[0]*pd[1]/pd[0] #y
p0[2]-p0[0]*pd[2]/pd[0] #z


#Line from y=0
p0 = torch.Tensor([0.5, 0, 0.5])
pd = p1-p0

#z=0
p0[0]-p0[2]*pd[0]/pd[2] #x
p0[1]-p0[2]*pd[1]/pd[2] #y

#y=0
p0[0]-p0[1]*pd[0]/pd[1] #x
p0[2]-p0[1]*pd[2]/pd[1] #z

#x=0
p0[1]-p0[0]*pd[1]/pd[0] #y
p0[2]-p0[0]*pd[2]/pd[0] #z



#Line from z=0
p0 = torch.Tensor([0.5, 0.5, 0])
pd = p1-p0

#z=0
p0[0]-p0[2]*pd[0]/pd[2] #x
p0[1]-p0[2]*pd[1]/pd[2] #y

#y=0
p0[0]-p0[1]*pd[0]/pd[1] #x
p0[2]-p0[1]*pd[2]/pd[1] #z

#x=0
p0[1]-p0[0]*pd[1]/pd[0] #y
p0[2]-p0[0]*pd[2]/pd[0] #z















#line that goes through 
p0 = torch.Tensor([0, 1, 0])
p1 = torch.Tensor([1, 0, 0])

#Where do they intercept?





# Find the ID to flip to for each set of ratios
xt = torch.Tensor([.1, .2, 1])
xt0 = 2*torch.Tensor([xt[0]/(xt[0]+xt[2]), xt[1]/(xt[1]+xt[0]), xt[2]/(xt[2]+xt[1])]) #another way to normalize

l = []
for ii in range(rr.shape[1]):
    r = rr[:,ii]
    
    #Energy method of choosing the ID (argmin)
    A, _, _ = find_pf_matrix(r)
    e = torch.matmul(A, xt)
    
    #Probability method for choosing (argmin for probability of NOT choosing that ID, argmax for probability of choosing each ID)
    A0 = torch.Tensor([r[2],r[0],r[1]])*2
    A = torch.Tensor([[r[1], 0, -r[2]],[0, -r[0], r[2]],[-r[1], r[0], 0]])
    p0 = A0 + torch.matmul(A,xt0) #probabilities of those IDs NOT occuring, (should add up to 2, none should be larger than 1, but they are sometimes with this method)
    B = torch.Tensor([[0,1,1],[1,0,1],[1,1,0]])
    p = torch.matmul(torch.linalg.inv(B),p0) #probabilities of those IDs occuring (should add up to 1, should be non-negative, but they are sometimes with this method)
    
    #Make sure each method gives the same response
    a0 = torch.argmin(e)
    a1 = torch.argmin(p0)
    a2 = torch.argmax(p)
    
    if a0!=a1: 
        print('a0!=a1')
        print(e)
        print(p0)
        
    if a1!=a2: 
        print('a1!=a2')
        print(p0)
        print(p)
    







# Find the ID to flip to for each set of ratios (for plane fitting)

xt = torch.Tensor([0.5, 0.1, 1]) #pair factors

bt = []
bp = []
bm = []

for ii in range(rr.shape[1]):
    r = rr[:,ii]
    
    #Sum of energy (example = [0.1, 0.5, 1.1])
    A, _, pf_loc = find_pf_matrix(r)
    bt_tmp = torch.matmul(A,xt) #sum of energies
    bt.append(bt_tmp)
    
    #Return random sampling from one minus the energy (example = [1,0,0], stochastic)
    aa = 1-bt_tmp
    a = aa.cumsum(0)
    tmp = np.random.rand()*a[-1]
    jj = (a>tmp).nonzero()[0]
    bp_tmp = torch.zeros(len(a))
    bp_tmp[jj] = 1
    bp.append(bp_tmp)
    
    #Return argmin (example = [1, 0, 0], determinisitic)
    tmp = torch.min(bt_tmp)
    jj = (bt_tmp==tmp).nonzero()
    kk = np.random.randint(len(jj))
    jj = jj[kk]
    bm_tmp = torch.zeros(len(bt_tmp))
    bm_tmp[jj] = 1
    bm.append(bm_tmp)
    
bt = torch.stack(bt)
bp = torch.stack(bp)
bm = torch.stack(bm)



# Fit a plane to each grain probability

i = np.random.randint(0, rr.shape[1], 3)
A = rr.T[i,]
B = bt[i,0]
x = torch.linalg.lstsq(A, B).solution.numpy()
print(x.round(4))


i = np.random.randint(0, rr.shape[1], 6000)
A = rr.T[i,]
B = bp[i,0]
x = torch.linalg.lstsq(A, B).solution.numpy()
print(x.round(4))


i = np.random.randint(0, rr.shape[1], 6000)
A = rr.T[i,]
B = bm[i,2]
x = torch.linalg.lstsq(A, B).solution.numpy()
print(x.round(4))


for h in range(3):

    from scipy.optimize import nnls
    i = np.random.randint(0, rr.shape[1], 6000)
    A = rr.T[i,]
    B = bm[i,h]
    
    x = nnls(A.numpy(), B.numpy())[0]
    print(x.round(4))


#Non-negative LMS that regularizes to an answer that adds up to 1
from scipy.optimize import nnls
l = B.shape[0]
AA = torch.cat([A, l*torch.ones(3)[None,]], 0)
BB = torch.cat([B, l*torch.ones(1)], 0)

x = nnls(AA.numpy(), BB.numpy())[0]
print(x.round(4))







### Select code from untitled 1, might not all work






#Run mode fitler and compute grain stats for a large microstructure
ic, ea, _ = fs.voronoi2image(size=[1024, 1024], ngrain=4096) 

ims_id = run_mf(ic, ea, nsteps=2000, cut=25, cov_mat=torch.Tensor([[10,0],[0,10]]), num_samples=64, memory_limit=1e10)

hps = ['./data/mf_sz(1024x1024)_ng(4096)_nsteps(2000)_cov(10-10-0)_numnei(64)_cut(25).h5']
gps = ['sim0']
fs.compute_grain_stats(hps, gps)


#Calculate stats and plots for small microstructure (have to create seperately with about code)
hps = ['./data/mf_sz(1024x1024)_ng(4096)_nsteps(1000)_cov(25-25-0)_numnei(64)_cut(0).h5',
       './data/mf_sz(1024x1024)_ng(4096)_nsteps(1000)_cov(25-25-0)_numnei(64)_cut(25).h5',
       './data/mf_sz(1024x1024)_ng(4096)_nsteps(2000)_cov(10-10-0)_numnei(64)_cut(0).h5',
       './data/mf_sz(1024x1024)_ng(4096)_nsteps(2000)_cov(10-10-0)_numnei(64)_cut(25).h5']
gps = ['sim0', 'sim0', 'sim0', 'sim0']
fs.compute_grain_stats(hps, gps)
fs.make_time_plots(hps, gps, scale_ngrains_ratio=0.20)






# Estimation grain pair energy factors (should be equal to misorientations in this case)

#Create a small isotropic simulation with MF to work with

ic, ea, _ = fs.voronoi2image(size=[64, 64], ngrain=64) 
ims_id = run_mf(ic, ea, nsteps=100, cut=25, cov_mat=torch.Tensor([[25,0],[0,25]]), num_samples=64, memory_limit=1e10)

fp = './data/mf_sz(64x64)_ng(64)_nsteps(100)_cov(25-25-0)_numnei(64)_cut(0).h5'
with h5py.File(fp) as f:
    ims_id = f['sim0/ims_id'][:]
    miso_matrix = f['sim0/miso_matrix'][:] #cut off angle was zero, so this isn't ground truth (should actually get all ones)


im = torch.from_numpy(ims_id[0,0])
im2 = torch.from_numpy(ims_id[1,0])

coords, samples, index = find_sample_coords(im, cov_mat=torch.Tensor([[25,0],[0,25]]), num_samples=64)
 
im_expand = im.reshape(-1,1).expand(-1, im.numel())
im_sampled = torch.gather(im_expand, dim=0, index=index)



pf = torch.zeros(miso_matrix.shape) #hold pair factors
c = torch.zeros(miso_matrix.shape) #count how many times a pair factor is seen

#find the actual factors I am solving for
# if cut==0: cut = 1e-10
# cut_rad = cut/180*np.pi
# miso_mat_norm = miso_matrix/cut_rad
# miso_mat_norm[miso_mat_norm>1] = 1
#since cut=0, I should actually find all ones

for ii in range(im_sampled.shape[1]):
    s = im_sampled[:,ii] #samples
    n = im2.reshape(-1)[ii] #ID switched to
    u = torch.unique(s) #IDs present
    y = (u==n)
    r = (s[:,None]==u[None,]).sum(0)/len(s)
    A = torch.Tensor([[r[1],r[2],0],[r[0],0,r[2]],[0,r[0],r[1]]])
    x = torch.linalg.inv(A)[:,y==True][:,0]
    
    tmp = torch.arange(len(r))
    i, j = torch.meshgrid(tmp, tmp)
    kkk = torch.stack([i.flatten(), j.flatten()])
    kk = torch.unique(torch.sort(kkk, 0)[0], dim=1)
    k = kk[:,kk[0]!=kk[1]]
    
    for i in range(k.shape[1]):  
        j0, j1 = k[:,i]
        k0, k1 = u[j0].long(), u[j1].long()
        pf[k0, k1] += x[i]
        pf[k1, k0] += x[i]
        c[k0, k1] += 1
        c[k1, k0] += 1









#3D plots of pair factors where colors are energy

xt = torch.Tensor([0.5, 0.1, 1]) #pair factors

log0 = []
for ii in tqdm(range(rr.shape[1])): 
    r = rr[:,ii]
    A, _,_ = find_pf_matrix(r)
    
     
    tmp2 = torch.matmul(A,xt)
    tmp3 = torch.min(tmp2)
    b0 = (tmp2<=tmp3).float()
    
    log0.append(tmp2)
    


p = torch.stack(log0)
p0 = p[:,0]
p1 = p[:,1]
p2 = p[:,2]

bd = np.linspace(0,1,11)
bb = bd[:-1]+0.5/(len(bd)-1)
a = np.histogramdd(rr.T, bins=[bd,bd,bd])[0]
a0 = np.histogramdd(rr.T, bins=[bd,bd,bd], weights=p0)[0]
a1 = np.histogramdd(rr.T, bins=[bd,bd,bd], weights=p1)[0]
a2 = np.histogramdd(rr.T, bins=[bd,bd,bd], weights=p2)[0]

a0[a!=0] = a0[a!=0]/a[a!=0]
a1[a!=0] = a1[a!=0]/a[a!=0]
a2[a!=0] = a2[a!=0]/a[a!=0]



j,i = np.where(a0[0,:,:]>0)
plt.plot(bb[j], a0[i,j,0], '.')
AA = torch.from_numpy(np.stack([np.ones(len(j)),bb[j]]).T).float()
BB = torch.from_numpy(a0[i,j,0]).float()
print(torch.linalg.lstsq(AA, BB).solution[1])

plt.plot(bb[j], a0[i,0,j])
AA = torch.from_numpy(np.stack([np.ones(len(j)),bb[j]]).T).float()
BB = torch.from_numpy(a0[i,0,j]).float()
print(torch.linalg.lstsq(AA, BB).solution[1])


j,i = np.where(a1[:,0,:]>0)
plt.plot(bb[j], a1[j,i,0], '.')
AA = torch.from_numpy(np.stack([np.ones(len(j)),bb[j]]).T).float()
BB = torch.from_numpy(a1[j,i,0]).float()
print(torch.linalg.lstsq(AA, BB).solution[1])

plt.plot(bb[j], a1[0,i,j])
AA = torch.from_numpy(np.stack([np.ones(len(j)),bb[j]]).T).float()
BB = torch.from_numpy(a1[0,i,j]).float()
print(torch.linalg.lstsq(AA, BB).solution[1])


j,i = np.where(a2[:,:,0]>0)
plt.plot(bb[j], a2[j,0,i], '.')
AA = torch.from_numpy(np.stack([np.ones(len(j)),bb[j]]).T).float()
BB = torch.from_numpy(a2[j,0,i]).float()
print(torch.linalg.lstsq(AA, BB).solution[1])

plt.plot(bb[j], a2[0,j,i])
AA = torch.from_numpy(np.stack([np.ones(len(j)),bb[j]]).T).float()
BB = torch.from_numpy(a2[0,j,i]).float()
print(torch.linalg.lstsq(AA, BB).solution[1])




i,j,k = np.where(a0>0)
c0 = a0[i,j,k]
i0,j0,k0 = bb[i], bb[j], bb[k]
i,j,k = np.where(a1>0)
c1 = a1[i,j,k]
i1,j1,k1 = bb[i], bb[j], bb[k]
i,j,k = np.where(a2>0)
c2 = a2[i,j,k]
i2,j2,k2 = bb[i], bb[j], bb[k]

fig = plt.figure()

ax = plt.axes(projection='3d')
# ax.view_init(0, 90)


# f0=0.2;f1=0.4;f2=1

# tmp = np.linspace(0,1,2)
# X,Y,Z = np.meshgrid(tmp,tmp,tmp)

# AA = torch.from_numpy(np.stack([np.ones(len(i0)),i0,j0,k0]).T).float()
# BB = torch.from_numpy(c0).float()
# x = torch.linalg.lstsq(AA, BB).solution.numpy()
# r = np.array([[1,0,0],[0,1,0],[0,0,1]])
# print(x[0]+x[1]*r[0]+x[2]*r[1]+x[3]*r[2])

# AA = torch.from_numpy(np.stack([np.ones(len(i1)),i1,j1,k1]).T).float()
# BB = torch.from_numpy(c1).float()
# x = torch.linalg.lstsq(AA, BB).solution.numpy()
# r = np.array([[1,0,0],[0,1,0],[0,0,1]])
# print(x[0]+x[1]*r[0]+x[2]*r[1]+x[3]*r[2])

# AA = torch.from_numpy(np.stack([np.ones(len(i2)),i2,j2,k2]).T).float()
# BB = torch.from_numpy(c2).float()
# x = torch.linalg.lstsq(AA, BB).solution.numpy()
# r = np.array([[1,0,0],[0,1,0],[0,0,1]])
# print(x[0]+x[1]*r[0]+x[2]*r[1]+x[3]*r[2])



# ax.scatter3D(i0,j0,k0)
# ax.scatter3D(i1,j1,k1)
# ax.scatter3D(i2,j2,k2)
ax.scatter3D(i0,j0,k0,c=1-c0, depthshade=0, cmap='Blues')
# ax.scatter3D(i1,j1,k1,c=1-c1, depthshade=0, cmap='Oranges')
# ax.scatter3D(i2,j2,k2,c=1-c2, depthshade=0, cmap='Greens')
# ax.scatter3D(i0,j0,1-c0)
# ax.scatter3D(i1,j1,1-c1)
# ax.scatter3D(i2,j2,1-c2)
ax.set_xlim([1,0])
ax.set_ylim([0,1])
ax.set_zlim([1,0])

ax.set_xlabel('r0')
ax.set_ylabel('r1')
ax.set_zlabel('r2')

#find those lines, which gives you the inequalities, which gives you the pair_factors






#find the locations where blue and yellow overlap


b0 = (a0!=0)*(a0!=1)*(a1!=0)*(a1!=1)
b1 = (a0!=0)*(a0!=1)*(a2!=0)*(a2!=1)
b2 = (a1!=0)*(a1!=1)*(a2!=0)*(a2!=1)

i,j,k = np.where(b0)
i0,j0,k0 = bb[i], bb[j], bb[k]
i,j,k = np.where(b1)
i1,j1,k1 = bb[i], bb[j], bb[k]
i,j,k = np.where(b2)
i2,j2,k2 = bb[i], bb[j], bb[k]

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(i0,j0,k0)
ax.scatter3D(i1,j1,k1)
ax.scatter3D(i2,j2,k2)
ax.set_xlim([1,0])
ax.set_ylim([0,1])
ax.set_zlim([1,0])

ax.set_xlabel('r0')
ax.set_ylabel('r1')
ax.set_zlabel('r2')



#fit a line to those coordinates



AA = torch.from_numpy(np.stack([np.ones(len(i0)),j0,k0]).T).float()
BB = torch.from_numpy(i0).float()
print(torch.linalg.lstsq(AA, BB).solution)

AA = torch.from_numpy(np.stack([np.ones(len(i1)),j1,k1]).T).float()
BB = torch.from_numpy(i1).float()
print(torch.linalg.lstsq(AA, BB).solution)

AA = torch.from_numpy(np.stack([np.ones(len(i2)),j2,k2]).T).float()
BB = torch.from_numpy(i2).float()
print(torch.linalg.lstsq(AA, BB).solution)






