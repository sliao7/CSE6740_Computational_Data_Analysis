import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import scipy.io
import pandas as pd
from sklearn import preprocessing
from scipy.stats import multivariate_normal as mvn
import scipy.sparse.linalg as ll
from kmeans import kmeans

# load data
data = scipy.io.loadmat('../data/data.mat')['data']
label = scipy.io.loadmat('../data/label.mat')['trueLabel']

data = np.array(data).T # each row is a data point 
label = np.array(label)

# visualize two of the data points
# plt.imshow(x[:,0].reshape((28,28)), cmap = 'gray') #2
# plt.imshow(x[-1].reshape((28,28)).T, cmap = 'gray') #6
# plt.show()
mu_original = np.mean(data, axis = 0, keepdims = True)
ndata = data - mu_original # shape = (784,1990)
# ndata = preprocessing.scale(data)
m, n = ndata.shape
C = np.matmul(ndata.T, ndata)/m

# pca the data
d = 5  # reduced dimension
V,Gamma,_ = np.linalg.svd(C)
V = V[:, :d]
Gamma = np.diag(Gamma[:d])

# project the data to the top 2 principal directions
# y = label
pdata = np.dot(ndata,V)
# plt.scatter(0, pdata[np.where(y == 2)],pdata[1,np.where(y == 2)])
# plt.scatter(0, pdata[np.where(y == 6)],pdata[1,np.where(y == 6)])
#plt.show()

# EM-GMM for wine data
# number of mixtures
K = 2

# random seed
seed = 5

# initialize prior
np.random.seed(seed)
pi = np.random.random(K)
pi = pi/np.sum(pi)

# initial mean and covariance
# np.random.seed(seed)
mu = np.random.randn(K,d)
mu_old = mu.copy()

sigma = []
for ii in range(K):
    # to ensure the covariance psd
    seed = 1 if ii == 0 else 4
    np.random.seed(seed)
    dummy = np.random.randn(d, d)
    sigma.append(dummy@dummy.T + np.eye(d))
    
# initialize the posterior
tau = np.full((m, K), fill_value=0.)

# # parameter for countour plot
# xrange = np.arange(-5, -5, 0.1)
# yrange = np.arange(-5, -5, 0.1)

# ####
maxIter= 100
tol = 1e-3

# plt.ion()
log_likelihood = []    
for ii in range(100):

    # E-step    
    for kk in range(K):
        sigma_det = np.linalg.det(sigma[kk])
        # print(sigma_det)
        tau[:, kk] = pi[kk] * mvn.pdf(pdata, mu[kk], sigma[kk])
        # print(tau[:,kk])
    # normalize tau
    sum_tau = np.sum(tau, axis=1)
    sum_tau.shape = (m,1)    
    tau = np.divide(tau, np.tile(sum_tau, (1, K)))
    
    log_likelihood.append(np.sum(np.log(sum_tau)))
    # M-step
    for kk in range(K):
        # update prior
        pi[kk] = np.sum(tau[:, kk])/m
        
        # update component mean
        mu[kk] = pdata.T @ tau[:,kk] / np.sum(tau[:,kk], axis = 0)
        
        # update cov matrix
        dummy = pdata - np.tile(mu[kk], (m,1)) # X-mu
        sigma[kk] = dummy.T @ np.diag(tau[:,kk]) @ dummy / np.sum(tau[:,kk], axis = 0)
        
    # print('-----iteration---',ii)  
    # tau_color = np.hstack((tau,np.ones((m,1))))  
    # plt.scatter(pdata[:,0], pdata[:,1], c= tau_color)
    # plt.axis('scaled')
    # plt.draw()
    # plt.pause(0.1)
    if np.linalg.norm(mu-mu_old) < tol:
        print('training coverged')
        break
    mu_old = mu.copy()
    if ii==99:
        print('max iteration reached')
        break
# plt.ioff()
# plt.close()


# plt.figure(figsize = (12,8))
# plt.plot(log_likelihood,'-*')
# plt.title('log_likelihood VS iterations')
# plt.ylabel('log_likelihood')
# plt.xlabel('iterations')
# plt.savefig('../latex/log_likelihood')
# plt.show()


# first_mean = (V @ mu[0] + mu_original).reshape((28,28)).T
# plt.imshow(first_mean, cmap = 'gray')
# plt.title('Mean of the first component with weight = ' + str(pi[0]))
# plt.savefig('../latex/mean1')
# plt.show()

# second_mean = (V @ mu[1] + mu_original).reshape((28,28)).T
# plt.imshow(second_mean, cmap = 'gray')
# plt.title('Mean of the second component with weight = ' + str(pi[1]))
# plt.savefig('../latex/mean2')
# plt.show()

# first_cov = V @ np.sqrt(Gamma) @ sigma[0] @ np.sqrt(Gamma) @ V.T 
# plt.imshow(first_cov, cmap = 'gray')
# plt.title('Covariance matrix of the first component')
# plt.savefig('../latex/cov1')
# plt.show()

# second_cov = V @ np.sqrt(Gamma) @ sigma[1] @ np.sqrt(Gamma) @ V.T 
# plt.imshow(second_cov, cmap = 'gray')
# plt.title('Covariance matrix of the second component')
# plt.savefig('../latex/cov2')
# plt.show()

# print(tau[:,0])
# label_em = np.argmax(tau,axis = 1)
# label_em[label_em == 1] = 2
# label_em[label_em == 0] = 6
# print('Mis-classification rate for digit' + str('2') + 'is')

# label_kmeans, _ = kmeans(pdata,2)
# label_kmeans[label_kmeans == 0] = 2
# label_kmeans[label_kmeans == 1] = 6
# print(label[0].shape, label_kmeans.shape,label_em.shape)
# print(confusion_matrix(label[0], label_kmeans))
# print(confusion_matrix(label[0], label_em))

# label2 = np.where(label == 2, label, 0)
# label6 = np.where(label == 6, label, 0)
# print(np.sum(label2 == label_em)) # 
# print(np.sum(label == 2))


