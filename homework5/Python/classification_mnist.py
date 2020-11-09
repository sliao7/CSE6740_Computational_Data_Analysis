import warnings
warnings.filterwarnings("ignore")
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.neural_network import MLPClassifier
import random
import scipy.io

### MNIST data set
# load data
data = scipy.io.loadmat('../data/mnist_10digits.mat')
X_train, y_train, X_test, y_test = data['xtrain'], data['ytrain'].T, data['xtest'], data['ytest'].T
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

y_train = y_train[:,0]
y_test = y_test[:,0]

# downsampling the training data
random.seed(1)
indices = random.sample(range(60000), 5000)
indices_median_trick = random.sample(range(60000), 1000)

# sample for median trick
X_sample = X_train[indices]/255

# downsampling the training data
X_train = X_train[indices]
y_train = y_train[indices]

# standardize the features
X_train = X_train/255
X_test = X_test/255

# KNN classifier
# tune k and p in KNN

#List Hyperparameters that we want to tune.
n_neighbors = list(range(1,10))
p=[1,2]
#Convert to dictionary
hyperparameters = dict(n_neighbors=n_neighbors, p=p)
#Create new KNN object
knn = KNeighborsClassifier()
#Use GridSearch
clf = GridSearchCV(knn, hyperparameters, cv=10)
#Fit the model
best_model = clf.fit(X_train,y_train)
#Print The value of best Hyperparameters
print('Best p:', best_model.best_estimator_.get_params()['p'])
print('Best n_neighbors:', best_model.best_estimator_.get_params()['n_neighbors'])

knn = best_model.best_estimator_
print('Finished tunning hyperparameters for KNN classifer')

## For the following 3 models, we fine tuned the hyperparameters in the jupyter notebook file

# Logistic Regression
lr = LogisticRegression(C = 0.2, penalty = 'l2')
lr.fit(X_train,y_train)
print('Finished fitting Logistic Regressor')

# Linear Support vector classifier
linearSVC = LinearSVC(C = 0.1, penalty = 'l2')
linearSVC.fit(X_train,y_train)
print('Finished fitting Linear Support Vector Classifier')

# Kernel Support Vector Classifier
# find M
P = X_sample @ X_sample.T
pairwise_distances = []
for i in range(1000):
    for j in range(i+1,1000):
        pairwise_distances.append(P[i,i] - 2 * P[i,j] + P[j,j])
M = np.median(np.array(pairwise_distances))

kernelSVC = SVC(kernel = 'rbf', gamma = 1/M, C = 12)
kernelSVC.fit(X_train, y_train)
print('Finished fitting Kernel Support Vector Classifier')

# Neural Network
NN = MLPClassifier(hidden_layer_sizes = (20,10))  
NN.fit(X_train,y_train)
print('Finished fitting Neural Network Classifier')

# Predict on test data
classifiers = { 'KNN classifier': knn,
        'Logistic Regressor': lr,
        'SVM': linearSVC,
        'Kernel SVM': kernelSVC,
        'Neural Networks': NN       
         }
for model_name in classifiers:
    model = classifiers[model_name]
    y_pred = model.predict(X_test)
    print(model_name + ':')
    print(classification_report(y_test, y_pred))
    print('confusion matrix: ')
    print(confusion_matrix(y_test,y_pred))
    print('*****************************************************')
