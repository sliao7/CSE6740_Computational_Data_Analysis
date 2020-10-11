import warnings
warnings.filterwarnings("ignore")
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import scipy.io
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
from itertools import product
# from scipy.stats import multivariate_normal as mvn
# import scipy.sparse.linalg as ll


# divorce dataset
# file = '../data/marriage.csv'

# columns = ['x_' + str(i) for i in range(1,55)] + ['label']
# X = pd.read_csv(file, header = None, index_col = False)
# X.columns = columns

# X, y = X[columns[:-1]], X['label']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle = False)


### MNIST data set
# load data
data = scipy.io.loadmat('../data/data.mat')['data']
label = scipy.io.loadmat('../data/label.mat')['trueLabel']

X = np.array(data).T # each row is a data point 
y = np.array(label).reshape((X.shape[0],))
y = (y == 6).astype(int)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle = False)


models = {'Naive Bayes': GaussianNB(var_smoothing = 1e-3), 
        'Logistic Regressor': LogisticRegression(),
        'KNN classifier': KNeighborsClassifier()
         }

def classifer(model_name):
    model = models[model_name]
    model.fit(X_train, y_train)
    print('***************************')
    print( model_name + ' training accuracy: ', model.score(X_train, y_train))
    print( model_name + ' testing accuracy: ', model.score(X_test, y_test))



for model_name in models:
    classifer(model_name)


pca = PCA(n_components = 2)
pca.fit(X_train)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
print("Results after PCA:")
for model_name in models:
    classifer(model_name)


# Plotting decision regions

def plot_boundary(file_name, X_train, y_train):
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
    f, axarr = plt.subplots(1, 3, sharex='col', sharey='row', figsize=(15, 5))

    for idx, clf, tt in zip([0, 1, 2], list(models.values()), list(models.keys())):
        print(idx)
        clf.fit(X_train,y_train)
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        axarr[idx].contourf(xx, yy, Z, alpha=0.4)
        for i in range(2):
            axarr[idx].scatter(X_train[y_train == i, 0], X_train[y_train == i, 1], label = str(i))
                                      
        axarr[idx].legend()
        axarr[idx].set_title(tt)
    
    plt.show()
    f.savefig('../latex/'+ file_name, bbox_inches='tight')

plot_boundary('MNIST', X_train, y_train)



