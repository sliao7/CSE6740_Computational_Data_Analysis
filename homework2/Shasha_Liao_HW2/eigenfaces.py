import numpy as np
import glob
from matplotlib import pyplot as plt
from scipy.linalg import svd

def quarter_res_avg(im):
    # to downsample an image

    original_width = im.shape[1]
    original_height = im.shape[0]

    width = original_width // 4
    height = original_height // 4

    resized_image = np.zeros(shape=(height, width), dtype=np.uint8)
    scale = 4

    for i in range(0, original_height-scale, scale):
        for j in range(0, original_width-scale, scale):
           resized_image[i//scale, j//scale] = np.mean(im[i:i + scale, j:j+scale], axis=(0, 1))

    resized_image = resized_image.astype(np.uint8)

    return resized_image


# to store data from two sujects
subject01 = [] 
subject02 = []

# read data
files = [file for file in glob.glob("data/yalefaces/*")]
for file_name in files:       
    image = plt.imread(file_name)
    # downsampling each image
    resized_image = quarter_res_avg(image).flatten() 
    if '1' in file_name:
        subject01.append(resized_image)
    else:
        subject02.append(resized_image)

test1 = subject01[-1]
test2 = subject02[-1]
subject01 = np.array(subject01)[:-1] # remove the test image
subject02 = np.array(subject02)[:-1] # remove the test image

# Apply PCA to obtain first six eigenfaces
def pca(X):
    m, n = X.shape
    X = X.T
    # PCA
    mu = np.mean(X, axis = 1, keepdims = True)
    X = X - mu
    W, _, _, = svd(X)   
    K = 6    
    W = W[:,0:K] # eigenvectors/eigenfaces

    return W


# plot eigenfaces  
def plot_eigenfaces(W):
    k = W.shape[1]
    fig, axs = plt.subplots(1,k, figsize=(14,2), facecolor='w', edgecolor='k')
    axs = axs.ravel()
    for i in range(k):
        image = W[:,i].reshape((60,80))
        axs[i].imshow(image,cmap='gray')
    plt.show()


# compute and visualize the first six eigenfaces
W1 = pca(subject01)
plot_eigenfaces(W1)

W2 = pca(subject02)
plot_eigenfaces(W2)

# perform face recognition
def similarity(x,y):
    return np.sum(x*y)/(np.linalg.norm(x) * np.linalg.norm(y))

e1 = W1[:,0]
e2 = W2[:,0]

def score(E,y):
    ans = 0
    m = E.shape[1]
    for i in range(m):
        ans += abs(similarity(E[:,i], y))
    return ans/m


# print the similarity scores
print("Scores using one eigenface are:")
print(similarity(e1,test1))
print(similarity(e2,test1))
print(similarity(e1,test2))
print(similarity(e2,test2))

print("Scores using six eigenfaces are:")
print(score(W1, test1))
print(score(W2, test1))
print(score(W1, test2))
print(score(W2, test2))



