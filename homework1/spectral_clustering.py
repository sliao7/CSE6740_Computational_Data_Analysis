from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import time
from scipy.sparse import csc_matrix, find
from os.path import abspath, exists
from scipy import sparse
from sklearn.cluster import KMeans
from kmeans import kmeans


def import_edges():
	# read the edges from 'edges.txt'
	f_path = abspath('data/edges.txt')
	if exists(f_path):
		with open(f_path) as graph_file:
			lines = [line.split() for line in graph_file]
	return np.array(lines).astype(int)

def import_labels():
	# read the labels from 'nodes.txt'
	f_path = abspath('data/nodes.txt')
	labels = []
	if exists(f_path):
		with open(f_path) as fid:
			for line in fid.readlines():
				label = line.split('\t')[2]
				labels.append(label)
	return np.array(labels).astype(int)

def main():
	# spectral clustering
	n = 1490
	

	# load the labels
	labels = import_labels()

	# load the graph
	a = import_edges()

	i = a[:,0] - 1
	j = a[:,1] - 1
	v = np.ones((a.shape[0],1)).flatten()

	A = sparse.coo_matrix((v,(i,j)),shape = (n,n))
	A = (A + A.T)/2

	degrees = np.sum(A, axis = 1).A1
	zero_nodes = [index + 1 for index in range(n) if degrees[index] == 0]
	nonzero_indices = [index for index in range(n) if degrees[index] != 0]

	# remove nodes with zero degree
	A = A[nonzero_indices,:]
	A = A[:,nonzero_indices]
	n = len(nonzero_indices)
	labels = labels[nonzero_indices]

	D = np.diag(1/np.sqrt(np.sum(A, axis = 1)).A1)
	L = D @ A @ D




	v,eigv = np.linalg.eig(L)

	overall_mismatch_rates = []

	for k in range(2,201):
		x = eigv[:,0:k].real 
		# print(x)
		x = x/np.repeat(np.sqrt(np.sum(x*x,axis=1).reshape(-1,1)),k,axis=1)
		# print(x)

		# scatter
		# plt.scatter(x[:,0],x[:,1])
		# plt.show()

		#kmeans
		c_idx, _ = kmeans(x,k)
		new_k = max(c_idx) + 1
	


		major_labels = []
		mismatch_rates = []
		cluster_weights = []
		overall_mismatch_rate = 0
		for i in range(new_k):
			cluster_size = len(c_idx[c_idx == i])
			cluster_ones = np.sum(labels[c_idx == i])
			ones_rate = cluster_ones/cluster_size
			if ones_rate > 0.5:			
				major_label = 1
				mismatch = 1 - ones_rate			
			else:			
				major_label = 0
				mismatch = ones_rate
				
			# major_labels.append(major_label)
			# mismatch_rates.append(mismatch)
			# cluster_weights.append(cluster_size/n)
			overall_mismatch_rate += mismatch * cluster_size/n
			print(new_k)

		overall_mismatch_rates.append(overall_mismatch_rate)
		# print('k = ', k)
		# print('major_labels:\n', major_labels)
		# print('mismatch_rates:\n', mismatch_rates)
		# print('cluster_weights:\n', cluster_weights)
		# print('overall_mismatch_rate:\n', overall_mismatch_rate)

	ks = [i for i in range(2,201)]
	plt.plot(ks, overall_mismatch_rates)
	plt.xlabel('k')
	plt.ylabel('mismatch_rates')
	plt.show()

	
	
if __name__ == '__main__':
	main()



