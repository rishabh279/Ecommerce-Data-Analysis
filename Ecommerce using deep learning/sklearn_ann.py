# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 02:18:21 2018

@author: rishabh
"""

from process import getData
from sklearn.utils import shuffle
from sklearn.neural_network import MLPClassifier

Xtrain,Ytrain,Xtest,Ytest=getData()

#create the neural network
model=MLPClassifier(hidden_layer_sizes=(20,20),max_iter=2000)

#train the model
model.fit(Xtrain,Ytrain)

train_accuracy=model.score(Xtrain,Ytrain)
test_accuracy=model.score(Xtest,Ytest)
print("train accuracy",train_accuracy,'test accuracy',test_accuracy)