
import numpy as np
import numpy.matlib
import pandas as pd
from scipy.stats import multivariate_normal as mvn

import matplotlib.pyplot as plt
from sklearn import preprocessing
# import random


data = pd.read_csv('wine.csv',header=None).to_numpy()
y = data[:,0]
data = data[:,1:]

ndata = preprocessing.scale(data)
m, n = ndata.shape
C = np.matmul(ndata.T, ndata)/m

# pca the data
d = 2  # reduced dimension
V,_,_ = np.linalg.svd(C)
V = V[:, :d]

# project the data to the top 2 principal directions
pdata = np.dot(ndata,V)
plt.scatter(pdata[np.where(y == 1),0],pdata[np.where(y == 1),1])
plt.scatter(pdata[np.where(y == 2),0],pdata[np.where(y == 2),1])
plt.scatter(pdata[np.where(y == 3),0],pdata[np.where(y == 3),1])
#plt.show()

# EM-GMM for wine data
# number of mixtures
K = 3

# random seed
seed = 4

# initialize prior
# np.random.seed(seed)
pi = np.random.random(K)
pi = pi/np.sum(pi)

# initial mean and covariance
# np.random.seed(seed)
mu = np.random.randn(K,2)
mu_old = mu.copy()

sigma = []
for ii in range(K):
    # to ensure the covariance psd
    # np.random.seed(seed)
    dummy = np.random.randn(2, 2)
    sigma.append(dummy@dummy.T)
    
# initialize the posterior
tau = np.full((m, K), fill_value=0.)

# # parameter for countour plot
# xrange = np.arange(-5, -5, 0.1)
# yrange = np.arange(-5, -5, 0.1)

# ####
maxIter= 100
tol = 1e-3

plt.ion()
    
for ii in range(100):

    # E-step    
    for kk in range(K):
        tau[:, kk] = pi[kk] * mvn.pdf(pdata, mu[kk], sigma[kk])
    # normalize tau
    sum_tau = np.sum(tau, axis=1)
    sum_tau.shape = (m,1)    
    tau = np.divide(tau, np.tile(sum_tau, (1, K)))
    
    
    # M-step
    for kk in range(K):
        # update prior
        pi[kk] = np.sum(tau[:, kk])/m
        
        # update component mean
        mu[kk] = pdata.T @ tau[:,kk] / np.sum(tau[:,kk], axis = 0)
        
        # update cov matrix
        dummy = pdata - np.tile(mu[kk], (m,1)) # X-mu
        sigma[kk] = dummy.T @ np.diag(tau[:,kk]) @ dummy / np.sum(tau[:,kk], axis = 0)
        
    print('-----iteration---',ii)  
    tau_color = np.hstack((tau,np.ones((m,1))))  
    plt.scatter(pdata[:,0], pdata[:,1], c= tau_color)
    plt.axis('scaled')
    plt.draw()
    plt.pause(0.1)
    if np.linalg.norm(mu-mu_old) < tol:
        print('training coverged')
        break
    mu_old = mu.copy()
    if ii==99:
        print('max iteration reached')
        break



print(tau)





