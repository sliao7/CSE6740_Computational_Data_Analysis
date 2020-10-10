import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import pairwise_distances
from Matrix_D import Matrix_D
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import scipy.io
import pandas as pd
from food_PCA import myPCA


# load data
x = scipy.io.loadmat('data/isomap.mat')['images'].T

# Step 1: build a weighted graph A using nearest neigbors
# compute pairwise distances between pictures
A = pairwise_distances(x,x,metric = 'l1')
m = A.shape[0] # number of nodes
n_neighbors = 101
for i in range(m):
    # find the threshold epsilon for each node so that it has at least 100 neighbors
    threshold = np.partition(A[i],n_neighbors)[n_neighbors-1]    
    for j in range(m):
        if A[i,j] > threshold:
            A[i,j] = 1e6 # set distance to be large if two nodes are unconnected
            
A = (A + A.T)/2 # make the similarity matrix symmetric

# Step 2: Compute pairwise shortest distance matrix
D = Matrix_D(A)  # time consuming step

# visualize the similarity matrix
plt.imshow(A)
plt.show()

# Step 3: Use a centering matrix H to get C
H = np.eye(m) - np.ones((m,m))/m
C = np.matmul(H,D*D)
C = np.matmul(C,H)
C = -C/2
C = (C + C.T)/2

# eigendecomposition on C
lambdas, w = np.linalg.eig(C)
k = 2
lambdas = np.sqrt(lambdas[:k])
w = w[:,0:k]

# obtain the projected 2d data set after the isomap
z = np.matmul(w,np.diag(lambdas))

## save the data to local
# pd.DataFrame(z).to_csv("twoPrincipalComponents_l1.csv")

## read saved local data
# z = pd.read_csv('twoPrincipalComponents_l1.csv')
# z = np.array(z)[:,1:]


# selected images to show in the 2D scatter plot
selected_image_idx = [493, 298, 351, 604, 201,228, 101, 697, 357, 114, 265, 532, 343, 520, 506, 471, 517,
                      43, 293, 40, 323, 363, 25, 652, 558, 422, 525, 259, 542, 194, 633, 644]

# function to visualize the projected dataset

def plot_faces(z):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(-z[:, 0], -z[:, 1], '.k')
    for i in selected_image_idx:
        single_image = x[i].reshape(64, 64).T

        imagebox = OffsetImage(single_image, zoom=0.6, cmap = 'gray')
        ab = AnnotationBbox(imagebox, -z[i], pad=0.1)
        ax.add_artist(ab)
    plt.show()

# run the function to visualize the isomap projected data set
plot_faces(-z)


## Apply PCA on the image data directly 
z = np.zeros((m,2))
_, z[:,0], z[:,1] = myPCA(x)
# z = np.column_stack((dim1,dim2))
plot_faces(z)


