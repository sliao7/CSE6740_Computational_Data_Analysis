#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 16:56:35 2020

@author: PFC

This script is for OMS-6740-Fall, HW1: Kmeans algorithm
Note the term Kmeans specifically refer to the one with L2 distance
If other distance is used, then the algorithm would be refer to 'generalized Kmeans', which
will be more complicated because of the optimization problem.

"""

import numpy as np
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt
import time
import imageio

raw_img = imageio.imread('beach.bmp')

def Kmeans(raw, k, ct_init):
    
    ct_old = ct_init
    
    dist_choice = 'euclidean'  # L2 norm
    
    n1,n2,n3 = np.shape(raw)
    raw = np.reshape(raw,(-1, 3))
    
    data = raw
    dim = np.shape(data)[1]
    
        
    ct_new = np.full((k,dim), np.nan)
    
    # Looping parameter
    maxIter = 200
    nIter = 1
    cost_old = 1e10
    cost_list = []
    
    while (nIter <=maxIter):
    
        # find the distances of all data points to each centroid
        dist_mtx = cdist(data, ct_old, dist_choice)
        
        current_cost = 0
        
        # find the cluster assignment for each data point
        cl = np.argmin(dist_mtx, axis=1)
        
        # update the centroid for each group
        for jj in range(k):
                        
            # find the index of data points in cluster ii
            idx_j = np.where(cl==jj)
            x_j = data[idx_j]
            
            ct_new[jj] = np.mean(x_j, axis=0)
            
            if ~np.isfinite(sum(ct_new[jj])):
                ct_new[jj] = np.full(np.shape(ct_new[jj]), fill_value = np.inf)
                        
            current_cost = current_cost + np.sum(x_j.dot(ct_new[jj]))
        
        # save the cost of current iteration to for record
        cost_list.append(current_cost)
        
        # check converge
        if current_cost == cost_old:
            break
        
        # update the variable for next iteration
        cost_old = current_cost
        ct_old = ct_new
        nIter = nIter+1
    
    # assign the new pixel value with new centroid
    dist_all = cdist(raw, ct_new, dist_choice)
    cl_all = np.argmin(dist_all, axis=1)
    
    # prepare to output the result
    img = np.full(np.shape(raw), fill_value = np.nan)
    for ii in np.unique(cl_all):
        img[np.where(cl_all == ii)] = ct_new[ii]/255
    
    img_out = np.reshape(img,(n1,n2,n3))
    
    # check empty cluster:
    n_empty = sum(1 - np.isfinite( np.sum(ct_new, axis=1) ))
    
    return img_out, n_empty



k_mesh = [2,4,8, 16, 32]
run_time = []
n_empty_all = []

fig, ax = plt.subplots(3,2)
ax[0,0].imshow(raw_img)
ax[0,0].set_title('original', fontsize = 8)
ax[0,0].get_xaxis().set_visible(False)
ax[0,0].get_yaxis().set_visible(False)



'''
 set random seed, different will lead different initialization, thus different final result
 
'''
rseed = 6

for ii in range(5):
    startt = time.time()
    
    # initialization would affect the result
    np.random.seed(rseed)
    
    # set the initialization within a certain range to reduce the chance of bad initialization
    ct_init = np.random.random((k_mesh[ii],3))*100+100
    
    img, n_empty = Kmeans(raw_img, k_mesh[ii], ct_init)
    endt = time.time()
    
    ax[int((ii+1)/2), np.remainder(ii+1,2)].imshow(img)
    ax[int((ii+1)/2), np.remainder(ii+1,2)].set_title('k='+str(k_mesh[ii]), fontsize = 8)
    ax[int((ii+1)/2), np.remainder(ii+1,2)].get_xaxis().set_visible(False)
    ax[int((ii+1)/2), np.remainder(ii+1,2)].get_yaxis().set_visible(False)
    
    run_time.append(endt - startt)
    n_empty_all.append(n_empty)
    
fig.tight_layout(pad=1.0)
fig.suptitle('Kmeans results')
fig.subplots_adjust(top=0.85)

plt.savefig('Kmeans_result.pdf', dpi = 300)

print('Kmeans result, current random seed: '+str(rseed))
print('the running time for each k')
for kk in range(5):
    print('k = '+str(k_mesh[kk])+':   '+'%.2f'%run_time[kk]+
          'sec.    # of empty cluser: '+ str(n_empty_all[kk]))


