import numpy as np 
import pandas as pd 
import math
import scipy.sparse.linalg as ll
import matplotlib.pyplot as plt

data = pd.read_csv('data/food-consumption.csv')
X = np.array(data[data.columns[1:]])
countries = data['Country']
foods = data.columns[1:]

## uncomment this part to see the result from sklearn PCA
#from sklearn.decomposition import PCA
# # apply PCA using sklearn
# pca = PCA(n_components = 2)
# pca.fit(X)
# x_new = pca.transform(X)

# fig, ax = plt.subplots()
# ax.scatter(x_new[:,0], x_new[:,1])
# for i in range(m):
#     ax.annotate(countries[i], (x_new[i,0], x_new[i,1]))
# plt.title("Scatter plot of the data after PCA from sklearn")
# plt.show()

def myPCA(X):
    # Each row in X is a data point
    m, n = X.shape
    X = X.T
    # PCA
    mu = np.mean(X, axis = 1, keepdims = True)
    X = X - mu
    C = np.dot(X, X.T)/m
    K = 2
    S,W = ll.eigs(C, k = K)
    S = S.real
    W = W.real
    dim1 = np.dot(W[:,0].T,X)/math.sqrt(S[0]) # extract the 1st principal component
    dim2 = np.dot(W[:,1].T,X)/math.sqrt(S[1]) # extract the 2nd principal component
    return W.real, dim1, dim2

# plot weights vectors w1 and w2
def plot_weights(W):

    plt.figure(figsize = (10,5))
    plt.subplot(121)
    plt.ylim(-.6,0.6)
    plt.stem(W[:,0], use_line_collection = True)
    plt.title("Weight vector w1")
    plt.subplot(122)
    plt.ylim(-.6,0.6)
    plt.stem(W[:,1], use_line_collection = True)
    plt.title("Wight vector w2")
    plt.show()


def scatter2D(dim1, dim2, labels):
    # scatter plot of the first two principal components of all the data points
    fig, ax = plt.subplots(figsize = (10,10))
    ax.scatter(dim1, dim2)
    # mark each data with it's country name
    for i in range(len(dim1)):
        ax.annotate(labels[i], (dim1[i], dim2[i]))
    plt.title("Scatter plot of the data after PCA")
    plt.show()


if __name__ == '__main__':
    # each data corresponds to a country
    W, dim1, dim2 = myPCA(X)
    plot_weights(W) # for part 3
    scatter2D(dim1, dim2, countries) # for part 4

    # change the perspective to explore the data
    # Now each data corresponds to a food item
    _, dim1, dim2 = myPCA(X.T)
    scatter2D(dim1, dim2, foods)