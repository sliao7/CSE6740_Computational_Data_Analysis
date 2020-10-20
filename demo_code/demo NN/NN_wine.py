#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 23:01:05 2020

@author: yxie77
"""

import pandas as pd

wine = pd.read_csv('wine.csv', names = ["Cultivator", "Alchol", "Malic_Acid", "Ash", "Alcalinity_of_Ash", "Magnesium", "Total_phenols", "Falvanoids", "Nonflavanoid_phenols", "Proanthocyanins", "Color_intensity", "Hue", "OD280", "Proline"])

"""
Check data
"""
wine.head()
wine.describe().transpose()

X = wine.drop('Cultivator',axis=1)
y = wine['Cultivator']
""" 
Split data for training and test
"""
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.33, random_state=42)

"""
The neural network in Python may have difficulty converging before 
the maximum number of iterations allowed if the data is not normalized. 
Multi-layer Perceptron is sensitive to feature scaling, 
so it is highly recommended to scale your data. 
"""

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


"""
First fit to training data
"""
scaler.fit(X_train) 
StandardScaler(copy=True, with_mean=True, with_std=True)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


"""
Training model
"""
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(10,10),max_iter=500)

mlp.fit(X_train,y_train)
MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(10, 10), learning_rate='constant',
       learning_rate_init=0.001, max_iter=500, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=None,
       shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
       verbose=False, warm_start=False)

"""
Performance evaluation
"""

predictions = mlp.predict(X_test)

"""
Use default metrics
"""

from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(y_test,predictions))

print(classification_report(y_test,predictions))







