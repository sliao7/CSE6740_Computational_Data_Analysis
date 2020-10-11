from matplotlib import pyplot as plt
import numpy as np
import time
from scipy.sparse import csc_matrix, find
path = 'data/beach.bmp'
pixels = plt.imread(path)/255
shapes = pixels.shape
pixels = np.array(pixels.reshape((shapes[0]*shapes[1],3)),dtype=np.float)

# pixels = pixels[np.random.randint(pixels.shape[0],size = (1,20))[0]]

#number of data points to work with
n = pixels.shape[0]



def objective_value(cluster,pixel, norm = 'l2'):
    '''
    Goal: To calculate the objective function value inside the cluster using pixel as the medoid
    
    Inputs:
    cluster: an *x3 array representing all pixels in a cluster
    pixel: a 1x3 array representing the medoid for the cluster

    Output:
    The objective function value inside the cluster using pixel as the medoid
    '''

    diff = cluster - pixel
    if norm == 'l2':
        return np.linalg.norm(diff)/len(diff)
    elif norm == 'l1':
        return np.sum(abs(diff))/len(diff)
    elif norm == 'loo':
        return np.max(abs(diff))
    else:
        print('Please use an acceptable similarity function.')


def label(x,medoids,norm = 'l2'):
    '''
    Inputs:
    x: pixels matrix with size n x 3
    medoids: centroids matrix with size k x 3
    norm: str indicating the similarity function, may be 'l1','l2','loo'

    Output:
    labels: an n x 1 matrix indicating the cluter each pixel belongs to
    '''

    diff = np.array([x - medoids[i] for i in range(len(medoids))])
    if norm == 'l2':
        diff = np.apply_along_axis(lambda a: np.sum(a*a), -1, diff)
    elif norm == 'l1':
        diff = np.apply_along_axis(lambda a: np.sum(np.abs(a)), -1, diff)
    elif norm == 'loo':
        diff = np.apply_along_axis(lambda a: np.max(np.abs(a)), -1, diff)
    else:
        print('Please use an acceptable similarity function.')
        return 

    labels = np.argmin(diff,axis = 0)
    return labels

# x = pixels[[i for i in range(1,1000,10)]]

def update_medoids(x,labels,medoids,norm):

    '''
    Find the medoids minimizing the objective function

    Inputs:
    x: pixels array with size n x 3
    labels: an array with size n x 1, labels[i] = j indicating x[i] belongs to the jth cluster
    medoids: an array with size k x 3 presenting the medoids for each cluster

    Output:
    med: an array with shape k x 3. Each row represent a medoid point
    '''

    k = len(medoids)
    new_meds = []
    total_obj = 0

    for i in range(k):
        cluster = x[labels == i]
        if len(cluster) == 0:
            continue
        min_obj = objective_value(cluster,medoids[i],norm)
        new_med = medoids[i]

        num = len(cluster)//20 + 1 # take approximately 10 percent of the pixels in each cluster
        random_indices = np.random.choice(len(cluster),num,replace=False)

        cluster_sample = cluster[random_indices]
        for pixel in cluster_sample:
            obj = objective_value(cluster,pixel,norm)
            if obj < min_obj:
                min_obj = obj
                new_med = pixel
        new_meds.append(new_med)
        total_obj += min_obj
    return np.array(new_meds), total_obj


def kmedoids(pixels,k,norm = 'l2'):
    # Randomly initialize medoids with data points  
    # Need to make sure that the centers in c are distinct 
    # meds = np.random.uniform(0,1,size=(k,3))
    meds = pixels[np.random.randint(0,pixels.shape[0], k)]

    # initial data assignment
    
    labels = label(pixels,meds,norm)

    # inital objective function value
    obj = float('inf')

    objs = []

    i = 0
    while i < 20:

        # update medoids
        meds, new_obj = update_medoids(pixels,labels,meds,norm)

        objs.append(new_obj)

        # update labels
        new_labels = label(pixels,meds,norm)

        i += 1

        if len(objs) > 2 and min(abs(objs[-3]-objs[-2]), abs(objs[-2] - objs[-1])) < 1e-8:
            break
        else:
            labels = new_labels
            obj = new_obj

    k = len(meds)
    print('kmedoids cluster with k = ' +  str(k) + ' after ' + str(i) + ' iterations.')
    # clustered = np.array([meds[i] for i in labels])
    # clustered = clustered.reshape(shapes)
    # plt.imshow(clustered)
    # plt.title('kmedoids cluster with k = ' +  str(k) + ' after ' + str(i) + ' iterations.')
    # plt.show()

    # plt.plot(objs)
    # plt.show()

    return labels, meds

# kmedoids(pixels,8,'loo')
# kmedoids(pixels,8,'loo')

times = []
for k in [3,5,10,16,32]:
    print('k = ', k)
    start = time.time()
    kmedoids(pixels,k,'loo')
    end = time.time()
    times.append(end - start)
    print('time = ', end - start)
print('elapsed times = ', times)



