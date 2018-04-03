# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 23:02:58 2018

@author: rishabh
"""

import numpy as np
import matplotlib.pyplot as plt

from process import get_binary_data

Xtrain,Ytrain,Xtest,Ytest=get_binary_data()

D=Xtrain.shape[1]
W=np.random.randn(D)
b=0

def sigmoid(z):
  return 1/(1+np.exp(-z))

#hypothesis function  
def forward(X,W,B):
  return sigmoid(X.dot(W)+B) 

#cost function  
def costFunction(Yhat,Y):
  return -np.mean(Y*np.log(Yhat)+((1-Y)*np.log(1-Yhat)))

#total correct/total number
def classification_rate(Y,P):
  return np.mean(Y==P)
  
#train loop
train_cost=[]
#test_cost=[]
learning_rate=0.0001
for i in range(10000):
  pYtrain=forward(Xtrain,W,b)
  
  trainCost=costFunction(pYtrain,Ytrain)
  #testCost=costFunction(pYtest,Ytest)
  
  train_cost.append(trainCost)
  #test_cost.append(testCost)
  
  #gradient Descent
  W=W-learning_rate*Xtrain.T.dot(pYtrain-Ytrain)
  b=b-learning_rate*(pYtrain-Ytrain).sum()  
  
print('Classification Rate for train data',classification_rate(Ytrain,np.round(pYtrain)))

  
legend1 =plt.plot(train_cost,label='train_cost')
#legend2 =plt.plot(test_cost,label='test_cost')
#plt.legend([legend1,legend2])
plt.legend(legend1)
'''
#####################----------------PREDICTION----------------------##############################
'''
pYtest=forward(Xtest,W,b)
print('Classification Rate for test data',classification_rate(Ytest,np.round(pYtest)))


