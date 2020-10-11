import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import svd
from scipy import stats
from sklearn.neighbors import KernelDensity

df = pd.read_csv("../data/n90pol.csv")
# data = np.array(df)
amygdala = np.array(df.amygdala)
acc = np.array(df.acc)
orientation = np.array(df.orientation)

##### part(a)

def plot_hist(pdata, data_name):
# histogram for first dimension of pdata
# find the range of the data
    m = len(pdata)
    min_data = min(pdata)
    max_data = max(pdata)
    print(min_data, max_data)
    nbin = 10     # you can change the number of bins in each dimension
    sbin = (max_data - min_data) / nbin
    #create the bins
    boundary = np.arange(min_data-0.001, max_data,sbin)
    # just loop over the data points, and count how many of data points are in each bin
    myhist = np.zeros(nbin+1)
    for i in range (m):
        whichbin = np.max(np.where(pdata[i] > boundary))
        myhist[whichbin] = myhist[whichbin] + 1

    myhist = np.divide(np.dot(myhist, nbin), m)
    
    # bar chart
    plt.figure()
    plt.bar(boundary+0.5 * sbin, myhist,width=sbin*0.8, align='center', alpha=0.5)
    plt.title("histogram of " + data_name)
    plt.show()

# plot_hist(acc, "acc")
# plot_hist(amygdala, "amygdala")


def plot_kde():
    kde = df[["amygdala", "acc"]].plot.kde()
    plt.title("KDE")
    plt.show()


#### part(b)


def plot_2dHist(x,y):
    """
    ==============================
    Create 3D histogram of 2D data
    ==============================

    plot a histogram for 2 dimensional data as a bar graph in 3D.
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # x_min, x_max = min(x), max(x)
    # y_min, y_max = min(y), max(y)
    hist, xedges, yedges = np.histogram2d(x, y, bins=14)#, range=[[x_min, x_max], [y_min, y_max]])

    # Construct arrays for the anchor positions of the 16 bars.
    # Note: np.meshgrid gives arrays in (ny, nx) so we use 'F' to flatten xpos,
    # ypos in column-major order. 
    xpos, ypos = np.meshgrid(xedges[:-1] + 0.005, yedges[:-1] + 0.005)
    xpos = xpos.flatten('F')
    ypos = ypos.flatten('F')
    zpos = np.zeros_like(xpos)


    # Construct arrays with the dimensions for the 16 bars.
    dx = xedges [1] - xedges [0]
    dy = yedges [1] - yedges [0]
    dz = hist.flatten()

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', zsort='average')   

    plt.show()



# plot_hist()
# plot_kde()
# plot_2dHist(amygdala,acc)

# plot 2d kde
def plot_kde2d(plots = 'surface'):
    # This function uses scipy package to choose the bandwidth automatically

    m1,m2 = np.array(df.amygdala), np.array(df.acc)
    xmin, xmax = np.min(m1), np.max(m1)
    ymin, ymax = np.min(m2), np.max(m2)
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([m1, m2])
    kernel = stats.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)

    if plots == 'contour':
        fig, ax = plt.subplots()
        ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,
                  extent=[xmin, xmax, ymin, ymax])
        ax.plot(m1, m2, 'k.', markersize=2)
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])

    if plots == 'surface':
        fig = plt.figure()
        ax=fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z)
        plt.show()

# plot_kde2d()  


def plot_kde2D(bandwidth, i, plots = 'surface'):

    """
    bandwidth:large bandwith leads to underfitting and small bandwith leads to overfitting
    i: for file name 
    plots: set plots = 'surface' to plot surface in 3d and plots to 'contour' in 2d

    """

    pdata = np.array(df[['amygdala','acc']])
    min_data = pdata.min(0)
    max_data = pdata.max(0)
    m = len(pdata)

#kernel density estimator
# create an evaluation grid
    gridno = 40
    inc1 = (max_data[0]-min_data[0])/gridno
    inc2 = (max_data[1]-min_data[1])/gridno
    gridx, gridy = np.meshgrid(np.arange(min_data[0], max_data[0]+inc1,inc1), np.arange(min_data[1], max_data[1]+inc2,inc2) )
    gridxno = gridx.shape[0]
    gridyno = gridx.shape[1]

    gridall = [gridx.flatten(order = 'F'), gridy.flatten(order = 'F')]
    gridall = (np.asarray(gridall)).T
    gridallno, nn= gridall.shape
    norm_pdata = (np.power(pdata, 2)).sum(axis=1)
    norm_gridall = (np.power(gridall, 2)).sum(axis=1)
    cross = np.dot(pdata,gridall.T)
    # compute squared distance between each data point and the grid point;
    # dist2 = np.matlib.repmat(norm_pdata, 1, gridallno)
    dist2 = norm_pdata.reshape((m,1)) + norm_gridall.reshape((1,gridallno)) - 2 * cross
    #choose kernel bandwidth 1; please also experiment with other bandwidth;
    
        
    #evaluate the kernel function value for each training data point and grid
    dist2 = dist2 / (bandwidth**2)
    kernelvalue = np.exp(-dist2/2)/(2*np.pi)
    kernelvalue = kernelvalue/(bandwidth**2)

    #sum over the training data point to the density value on the grid points;
    # here I dropped the normalization factor in front of the kernel function,
    # and you can add it back. It is just a constant scaling;
    mkde = sum(kernelvalue).reshape(gridallno,1) / m 
    #reshape back to grid;
    mkde = ((mkde.T).reshape((gridyno, gridxno))).T

    if plots == 'surface':
        fig = plt.figure()
        ax=fig.add_subplot(111, projection='3d')   
        ax.plot_surface(gridx, gridy, mkde)
        plt.title("Bandwith = " + str(bandwidth))
        plt.show()
    
    if plots == 'contour':
        fig, ax = plt.subplots()
        # CS = ax.contour(X, Y, Z)
        # fig = plt.figure()
        CS = ax.contour(gridx, gridy, mkde, 20)
        # plt.title("Bandwith = " + str(bandwidth))
        # plt.clabel(inline = True)
        # ax.clabel(CS, inline=True, fontsize=10, inline_spacing = 3)
        cbar = fig.colorbar(CS)
        ax.set_title("Bandwith = " + str(bandwidth))
        plt.savefig('../latex/contour' + str(i))
        plt.show()


# for i, bandwidth in enumerate([0.05, 0.025, 0.015, 0.01, 0.0075, 0.005]):
#     plot_kde2D(bandwidth, i, 'contour')
# plot_kde2D(0.015, 'contour')


#### part(c)


data = np.vstack([amygdala, acc])

xmin, xmax = np.min(amygdala), np.max(amygdala)
ymin, ymax = np.min(acc), np.max(acc)

X, Y = np.mgrid[2*xmin:2*xmax:200j, 2*ymin:2*ymax:200j]
positions = np.vstack([X.ravel(), Y.ravel()])

### using scipy, where the bandwidths were chosen automatically
def scipy_kde():
    kernel_joint = stats.gaussian_kde(data)
    kernel_amy = stats.gaussian_kde(amygdala)
    kernel_acc = stats.gaussian_kde(acc)
    joint = np.reshape(kernel_joint(positions).T, X.shape)
    amygdala_kde = kernel_amy(X[:,0])
    acc_kde = kernel_acc(Y[0])

    product = amygdala_kde[:,None] * acc_kde[None,:]

    plt.imshow(joint, cmap='viridis')
    plt.colorbar()
    plt.title("Heatmap of the joint kernel distribution")
    plt.savefig('latex/scipy_joint')
    plt.close()
    # plt.show()
    

    plt.imshow(product, cmap='viridis')
    plt.colorbar()
    plt.title("Heatmap of the product")
    plt.savefig('latex/scipy_product')
    plt.close()
    # plt.show()
    

    plt.imshow(np.abs(joint - product), cmap='viridis')
    plt.colorbar()
    plt.title("Heatmap of the error")
    plt.savefig('latex/scipy_difference')
    plt.close()
    # plt.show()
    

# scipy_kde()


def sklearn_kde():
    ### find kde of amygdala
    x_d = X[:,1]
    x = amygdala
    kde = KernelDensity(bandwidth=0.01, kernel='gaussian')
    kde.fit(x[:, None])
    logprob = kde.score_samples(x_d[:, None])
    amygdala_kde = np.exp(logprob)

    ### find kde of acc
    y_d = Y[0]
    y = acc
    kde = KernelDensity(bandwidth=0.01, kernel='gaussian')
    kde.fit(y[:, None])
    logprob = kde.score_samples(y_d[:, None])
    acc_kde = np.exp(logprob)

    ### find the products of the marginal distributions
    product = amygdala_kde[:,None] * acc_kde[None,:]


    ### find the joint kde
    data = np.array(df[['amygdala','acc']])
    kde = KernelDensity(bandwidth=0.01, kernel='gaussian')
    kde.fit(data)
    logprob = kde.score_samples(positions.T)
    joint = np.reshape(np.exp(logprob), X.shape)


    ### plot the heap maps
    plt.imshow(joint, cmap='viridis')
    plt.colorbar()
    plt.title("Heatmap of the joint kernel distribution")
    plt.savefig('latex/joint')
    plt.close()
    # plt.show()
    

    plt.imshow(product, cmap='viridis')
    plt.colorbar()
    plt.title("Heatmap of the product")
    plt.savefig('latex/product')
    plt.close()
    # plt.show()
    

    plt.imshow(np.abs(joint - product), cmap='viridis')
    plt.colorbar()
    plt.title("Heatmap of the error")
    plt.savefig('latex/difference')
    plt.close()
    # plt.show()
    

# part(d) 
def kde1d(data,grids):
    # generate the values of 1d kde on grides from data
    kde = KernelDensity(bandwidth=0.01, kernel='gaussian')
    kde.fit(data[:, None])
    logprob = kde.score_samples(grids[:, None])   
    return np.exp(logprob)

def plot_conditional_kde(data,grids,data_name):
    # plot the conditional kdes
    for i in range(2,6):
        kde = kde1d(data[orientation == i],grids)
        plt.plot(grids, kde)
        plt.title('P(' + data_name + ' | ' + 'orientation = ' + str(i) + ')')
        plt.xlabel(data_name)
        plt.ylabel('density')
        # plt.show()
        plt.savefig(../latex/data_name + str(i))
        plt.close()

# plot_conditional_kde(amygdala, X[:,0], 'amygdala')
# plot_conditional_kde(acc, Y[0], 'acc')
 
### part(e)

def kde2d(data, grids):
    # generate the values of 2d kde on 2d grides from 2d data
    kde = KernelDensity(bandwidth=0.01, kernel='gaussian')
    kde.fit(data)
    logprob = kde.score_samples(grids.T)
    joint = np.reshape(np.exp(logprob), X.shape)
    return joint

def plot_conditonal_joint_kde(data, grids):
    # for part (d) to study conditional joint distribution of the amygdala and acc
    for i in range(2,6):
        joint_kde = kde2d(data[orientation == i],grids)
        fig = plt.figure()
        ax=fig.add_subplot(111, projection='3d')   
        ax.plot_surface(X, Y, joint_kde)       
        plt.title('P(' + 'amygdala, acc' + ' | ' + 'orientation = ' + str(i) + ')')
        plt.savefig('../latex/joint_surface' + str(i))
        plt.close()    

data = np.array(df[['amygdala','acc']])
grids = positions
# plot_conditonal_joint_kde(data,grids)



### part(f)
def plot_heatmaps(joint_data, amygdala, acc):
    # for part(f) to study condional independence between amygdala and acc
    for i in range(2,6):
        joint_kde = kde2d(data[orientation == i],positions)
        amygdala_kde = kde1d(amygdala[orientation == i],X[:,0])
        acc_kde = kde1d(acc[orientation == i],Y[0])
        product = amygdala_kde[:,None] * acc_kde[None, :]

        plt.imshow(joint_kde, cmap='viridis')
        plt.colorbar()
        plt.title('P(' + 'amygdala, acc' + ' | ' + 'orientation = ' + str(i) + ')')
        plt.savefig('../latex/joint' + str(i))
        plt.close()

        plt.imshow(product, cmap='viridis')
        plt.colorbar()
        plt.title('P(' + 'amygdala' + ' | ' + 'orientation = ' + str(i) + ')' + 'P(' + 'acc' + ' | ' + 'orientation = ' + str(i) + ')')
        plt.savefig('../latex/product' + str(i))
        plt.close()

        plt.imshow(np.abs(joint_kde - product), cmap='viridis')
        plt.colorbar()
        plt.title("error for orientation = " + str(i))
        plt.savefig('../latex/error' + str(i))
        plt.close()


# plot_heatmaps(data, amygdala, acc)



