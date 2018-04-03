# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 23:04:25 2018

@author: rishabh
data pre-processing of ecommerce data
"""

import pandas as pd
import numpy as np

def getData():
  df=pd.read_csv('E:\RS\ML\Machine learning tuts\Target\Part1(Regression)\Logistic Regression from Scratch\Ecommerce\ecommerce_data.csv')
  
  data=df.as_matrix()
  
  np.random.shuffle(data)
  
  X=data[:,:-1]
  Y=data[:,-1].astype(np.int32)
  
  #duplicate of X to perform onehotencoding 
  N,D = X.shape
  X2=np.zeros((N,D+3))
  X2[:,0:(D-1)]=X[:,0:(D-1)]
     
  #performing onehotencoding 
  for i in range(N):
    t=X[i,D-1]
    X2[i,t+D-1]=1
  
  X=X2
    
  Xtrain=X[:-100]
  Ytrain=Y[:-100]
  Xtest=X[-100:]
  Ytest=Y[-100:]
  
  #normalize column 1 and 2
  for i in (1,2):
    m=X[:,1].mean()
    s=X[:,2].std()
    Xtrain[:,i]=(Xtrain[:,i]-m)/s
    Xtest[:,i]=(Xtest[:,i]-m)/s
    
  return Xtrain,Ytrain,Xtest,Ytest

def get_binary_data():
  Xtrain,Ytrain,Xtest,Ytest=getData()
  X2train=Xtrain[Ytrain<=1]
  Y2train=Ytrain[Ytrain<=1]
  X2test=Xtest[Ytest<=1]
  Y2test=Ytest[Ytest<=1]
  return X2train,Y2train,X2test,Y2test