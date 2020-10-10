#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 16:56:35 2020

@author: PFC

This script is for OMS-6740-Fall, HW1: Kmedoid algorithm
To reduce the computational complexity, a subset of data point were sample for training

"""

import numpy as np
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt
import time
import imageio
import random

raw_img = imageio.imread('beach.bmp')

def Kmedoid(raw, k, dist_choice, nsample):
    
    n1,n2,n3 = np.shape(raw)
    raw = np.reshape(raw,(-1, 3))
    
    # sample a subset of data point from the whole data point
    # here we select 6000 out of all pixels
    selected = random.sample(range(len(raw)),nsample)
    data = raw[selected]
    dim = np.shape(data)[1]
    
    # startt = time.time()
        
    ct_new = np.full((k,dim), np.nan)
    
    # initialing the centroids
    ct_old = data[random.sample(range(len(data)), k)]
    
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
            
            # calculate the pair-wise distance in cluster_j
            dist_j = cdist(x_j, x_j, dist_choice)
            
            # find the data point in cluster_j that minimized the in-cluster cost. 
            # It will be the new centroid
            idx_sub = np.argmin(np.sum(dist_j, axis=1), axis=0)
            ct_new[jj] = x_j[idx_sub]
            
            current_cost = current_cost + np.sum(dist_j[idx_sub])
        
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
    
    img = np.full(np.shape(raw), fill_value = np.nan)
    for ii in np.unique(cl_all):
        img[np.where(cl_all == ii)] = ct_new[ii]/255
    
    img_out = np.reshape(img,(n1,n2,n3))
    return img_out


# choose the distance metric: 
# dist_choice = 'euclidean'  # L2 norm
dist_choice = 'cityblock'  # L1 norm

# k_mesh = np.array([2,4,8, 8, 32])
k_mesh = [2,4,8, 16, 32]
run_time = []

'''
choose the sub-sampling level, 
    the larger nsample, the program run slower
    the smaller nsample, clusters may have zeros member
'''
nsample = 10000

fig, ax = plt.subplots(3,2)
ax[0,0].imshow(raw_img)
ax[0,0].set_title('original', fontsize = 8)
ax[0,0].get_xaxis().set_visible(False)
ax[0,0].get_yaxis().set_visible(False)
for ii in range(5):
    startt = time.time()
    img = Kmedoid(raw_img, k_mesh[ii], dist_choice, nsample)
    endt = time.time()
    
    ax[int((ii+1)/2), np.remainder(ii+1,2)].imshow(img)
    ax[int((ii+1)/2), np.remainder(ii+1,2)].set_title('k='+str(k_mesh[ii]), fontsize = 8)
    ax[int((ii+1)/2), np.remainder(ii+1,2)].get_xaxis().set_visible(False)
    ax[int((ii+1)/2), np.remainder(ii+1,2)].get_yaxis().set_visible(False)
    
    run_time.append(endt - startt)
fig.tight_layout(pad=1.0)
fig.suptitle('Kmedoids results ('+ dist_choice+')')
fig.subplots_adjust(top=0.85)
plt.savefig('kmedoid_result.pdf', dpi = 300)

print('Kmedoid with distance '+dist_choice)
print('sub-sample size: '+str(nsample))
print('the running time for each k')
for kk in range(5):
    print('k = '+str(k_mesh[kk])+':   '+'%.2f'%run_time[kk]+'sec')


