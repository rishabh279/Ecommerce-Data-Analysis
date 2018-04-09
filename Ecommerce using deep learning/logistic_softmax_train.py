# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 15:08:05 2018

@author: rishabh
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from process import getData

Xtrain,Ytrain,Xtest,Ytest=getData()
K=len(set(Ytrain)|set(Ytest))
D=len(Xtrain[0,:])#Xtrain.shape[1]

W=np.random.randn(D,K)
b=np.random.randn(K)
def y2indicator(Y,K):
  N=len(Y)
  ind=np.zeros((N,K))
  for i in range(N):
    ind[i,Y[i]]=1
  return ind
  
def forward(X,W,b):
  return softmax(X.dot(W)+b)
  
def softmax(A):
  expA=np.exp(A)
  return (expA/expA.sum(axis=1,keepdims=True))

def crossEntropy(T,Yhat):
  return -np.mean(T*np.log(Yhat))
  
def classification(T,Yhat):
  return np.mean(T==Yhat)
  
  
Ytrain_ind = y2indicator(Ytrain,K)
Ytest_ind=y2indicator(Ytest,K)
  
#train loop
train_costs=[]
test_costs=[]
learning_rate=0.001
for i in range(10000):
  pYtrain=forward(Xtrain,W,b)
  pYtest=forward(Xtest,W,b)
  ctrain=crossEntropy(Ytrain_ind,pYtrain)
  ctest=crossEntropy(Ytest_ind,pYtest)
  train_costs.append(ctrain)
  test_costs.append(ctest)
  
  W-=learning_rate*Xtrain.T.dot(pYtrain-Ytrain_ind)
  b-=learning_rate*(pYtrain-Ytrain_ind).sum(axis=0)
  #if i%1000==0:
  #  print('Classification_Rate',classification)
  
print('Training Classification_rate',classification(Ytrain,np.argmax(pYtrain,axis=1)))
print('Test Classification_rate',classification(Ytest,np.argmax(pYtest,axis=1)))
'''
legend1, = plt.plot(train_costs, label='train cost')
legend2, = plt.plot(test_costs, label='test cost')
plt.legend([legend1, legend2])
'''
legend1,=plt.plot(train_costs,label='train_cost')
legend2,=plt.plot(test_costs,label='test_cost') 
plt.legend() 
