import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import pairwise_distances
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import scipy.io
import pandas as pd
from sklearn import preprocessing
from scipy.stats import multivariate_normal as mvn
import scipy.sparse.linalg as ll

# load data
data = scipy.io.loadmat('data/data.mat')['data']
label = scipy.io.loadmat('data/label.mat')['trueLabel']

data = np.array(data) # each column is a data point 
label = np.array(label)

# visualize two of the data points
# plt.imshow(x[:,0].reshape((28,28)), cmap = 'gray') #2
# plt.imshow(x[-1].reshape((28,28)).T, cmap = 'gray') #6
# plt.show()
mu = np.mean(data, axis = 1, keepdims = True)
ndata = data - mu # shape = (784,1990)
# ndata = preprocessing.scale(data)
n, m = ndata.shape
C = np.matmul(ndata, ndata.T)/m

# pca the data
d = 5  # reduced dimension
G, V = ll.eigs(C, k = d) # return the first d eigenvalues and eigenvectors
# V,_,_ = np.linalg.svd(C)
G = G.real; G = G.reshape((d,1))# eigenvalues
V = V.real # eigenvectors (784, 5)

# project the data to the top 2 principal directions
# pdata = np.dot(ndata,V)
pdata = np.sqrt(G)* (V.T @ ndata)


# visualze the two sets of data in a plane
# plt.scatter(pdata[0, np.where(label == 2)],pdata[1,np.where(label == 2)])
# plt.scatter(pdata[0, np.where(label == 6)],pdata[1,np.where(label == 6)])
# plt.show()

# EM-GMM for wine data
# number of mixtures
K = 2

# # random seed
seed = 5

# # initialize prior
np.random.seed(seed)
pi = np.random.random(K)
pi = pi/np.sum(pi)

# # initial mean and covariance
np.random.seed(seed)
mu = np.random.randn(d,K) # generate a random matrix of size (d,K) from normal distribution
mu_old = mu.copy()

# np.random.seed(seed)
sigma = []
for ii in range(K):
    # to ensure the covariance psd
    seed = 1 if ii == 0 else 4
    np.random.seed(seed)
    dummy = np.random.randn(d, d)
    sigma.append(dummy@dummy.T + np.eye(d))
    
# # initialize the posterior
tau = np.full((K,m), fill_value=0.)

# # # parameter for countour plot
# # xrange = np.arange(-5, -5, 0.1)
# # yrange = np.arange(-5, -5, 0.1)

# # ####
maxIter= 10
tol = 1e-1

# # plt.ion() #uncomment to see the process of convergence
log_likelihood = []
    
for ii in range(maxIter):

#     # E-step    
    for kk in range(K):
        sigma_det = np.linalg.det(sigma[kk])
        # print(sigma_det)

        tau[kk] = pi[kk] * np.exp(-1/2 * np.diag((pdata - mu[:,kk][:,None]).T @ np.linalg.inv(sigma[kk]) @ (pdata - mu[:,kk][:,None])))
        print(tau[kk])
        # tau[kk] = pi[kk] * mvn.pdf(pdata, mu[kk], sigma[kk])
#     # normalize tau
    sum_tau = np.sum(tau, axis=0)
    sum_tau.shape = (1,m)    
    tau = tau / sum_tau
    
    log_likelihood.append(np.sum(np.log(sum_tau)))

#     # M-step
    for kk in range(K):
        # update prior
        pi[kk] = np.sum(tau[kk])/m
        
        # update component mean
        mu[:,kk] = (pdata @ tau[kk][None,:].T).reshape(d) / np.sum(tau[kk])
        
        # update cov matrix
        dummy = pdata - mu[:,kk][:,None] # X-mu
        
        sigma[kk] = dummy @ np.diag(tau[kk]) @ dummy.T / np.sum(tau[kk])
    


    # print('-----iteration---',ii)   
#     # tau_color = np.hstack((tau,np.ones((m,1))))
#     # plt.scatter(pdata[:,0], pdata[:,1], c= tau_color)
#     # plt.axis('scaled')
#     # plt.draw()
#     # plt.pause(0.1)
    if np.linalg.norm(mu-mu_old) < tol:
        print('training coverged')
        break
    mu_old = mu.copy()
    if ii==29:
        print('max iteration reached')
        break
# print(log_likelihood)
# plt.figure(figsize = (12,8))
# plt.plot(log_likelihood,'*')
# plt.title('log_likelihood VS iterations')
# plt.ylabel('log_likelihood')
# plt.xlabel('iterations')
# plt.show()



