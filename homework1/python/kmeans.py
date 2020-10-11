from matplotlib import pyplot as plt
import numpy as np
import time
from scipy.sparse import csc_matrix, find


# pixels = pixels[np.random.randint(pixels.shape[0],size = (1,20))[0]]

#number of data points to work with
# n = pixels.shape[0]

def kmeans(x,k):
	# Randomly initialize centroids with data points  
	n = x.shape[0]
	c = np.random.uniform(0,1,size=(x.shape[1],k))
	# c = x[np.random.randint(0,x.shape[0], k)].T

	# initial data assignment
	c2 = np.sum(np.power(c,2),axis=0,keepdims=True)# norm squred of the centroids
	tempdiff = (2*np.dot(x,c)-c2)
	labels = np.argmax(tempdiff,axis = 1)

	iter = 0
	while True:

		# update date assignment matrix
		# the assignment matrix is a sparse matrix,
		# with size n x k
		P = csc_matrix((np.ones(n), (np.arange(0, n, 1), labels)), shape=(n, k))
		count = P.sum(axis = 0)
		

		nonzero_index = [i for i in range(k) if count[0,i] > 0]
		count = count[0,nonzero_index]


		# Recompute centroids
		# pixels.T * P implements summation of data points assigned to a given cluster
		c = np.array(P.T.dot(x)).T
		c = np.array(c[:,nonzero_index]/count)
		
		# update cluster number
		k = len(nonzero_index)

		# cluster assignment
		c2 = np.sum(np.power(c,2),axis=0,keepdims=True)# norm squred of the centroids
		tempdiff = (2*np.dot(x,c)-c2)
		new_labels = np.argmax(tempdiff,axis = 1)

		iter += 1

		if all(new_labels == labels):
			break
		else:
			labels = new_labels

	# print('kmeans cluster with k = ' +  str(k) + ' after ' + str(iter) + ' iterations.')


	# clustered = np.array([c.T[i] for i in labels])
	# clustered = clustered.reshape(shapes)
	# plt.imshow(clustered)
	# plt.title('kmeans cluster with k = ' +  str(k) + ' after ' + str(iter) + ' iterations.')
	# plt.show()


	return labels, c.T

# kmeans(pixels,15)
# kmeans(pixels,10)

# times = []
# for k in [3,5,10,16,32]:
#     print('k = ', k)
#     start = time.time()
#     kmeans(pixels,k)
#     end = time.time()
#     times.append(end - start)
# print('elapsed times = ', times)

if __name__ == '__main__':
	path = 'data/beach.bmp'
	pixels = plt.imread(path)/255
	# plt.imshow(pixels)
	# plt.title('Original picture')
	# plt.show()
	shapes = pixels.shape
	pixels = np.array(pixels.reshape((shapes[0]*shapes[1],3)),dtype=np.float)
	kmeans(pixels,15)

