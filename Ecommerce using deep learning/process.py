# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 01:53:15 2018
Illustrate ecommerece data preprocessing
@author: rishabh
"""

import numpy as np
import pandas as pd
  
def getData():
  df=pd.read_csv('E:/RS/ML/Machine learning tuts/Target/Projects/Ecommerce Data Analysis/Ecommerce using deep learning/ecommerce_data.csv')
  
  data=df.as_matrix()
  
  np.random.shuffle(data)
  
  X=data[:,:-1]
  Y=data[:,-1].astype(np.int32)
  
  N,D=X.shape
  X2=np.zeros((N,D+3))
  
  X2[:,0:(D-1)]=X[:,0:(D-1)]
  
  #One-HotEncoding
  for i in range(N):
    t=int(X[i,D-1])
    X2[i,D-1+t]=1
  
  X=X2
  Xtrain=X[:-100]
  Ytrain=Y[:-100]
  Xtest=X[-100:]
  Ytest=Y[-100:]

  #normalize data
  for i in (1,2):
    m=Xtrain[:,i].mean()
    s=Xtrain[:,i].std()
    Xtrain[:,i]=(Xtrain[:,i]-m)/s
    Xtest[:,i]=(Xtest[:,i]-m)/s

  return Xtrain,Ytrain,Xtest,Ytest

  
def classification(Y,P):
  return (Y==P).mean()

  
def get_binaryData():
  Xtrain,Ytrain,Xtest,Ytest=getData()
  X2train=Xtrain[Ytrain<=1]
  Y2train=Ytrain[Ytrain<=1]
  X2test=Xtest[Ytest<=1]
  Y2test=Ytest[Ytest<=1]
  return X2train,Y2train,X2test,Y2test


  

  