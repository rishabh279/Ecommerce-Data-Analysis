# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 03:19:02 2018
Prediction ecommerce data analysis
@author: rishabh
"""

import numpy as np
from process import getData

X,Y,_,_=getData()

M=5
D=X.shape[1]
K=len(set(Y))
W1=np.random.randn(D,M)
b1=np.zeros(M)
W2=np.random.randn(M,K)
b2=np.zeros(K)

def softmax(z):
  expA=np.exp(z)
  return expA/expA.sum(axis=1,keepdims=True)
  
def forward(X,W1,b1,W2,b2):
  Z=np.tanh(X.dot(W1)+b1)
  return softmax(Z.dot(W2)+b2)
  
P_predict=forward(X,W1,b1,W2,b2)
predictions=np.argmax(P_predict,axis=1)

print('Classification:',np.mean(predictions==Y))


  