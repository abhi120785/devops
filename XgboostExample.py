# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 16:00:29 2019

@author: pnl06c6y
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')

X = dataset.iloc[:,3:13].values
y = dataset.iloc[:,13].values

#Encode categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X_1 = LabelEncoder()
X[:,1]=labelencoder_X_1.fit_transform(X[:,1])
labelencoder_X_2 = LabelEncoder()
X[:,2]=labelencoder_X_2.fit_transform(X[:,2])
onehotencoder=OneHotEncoder(categorical_features=[1]) 
X=onehotencoder.fit_transform(X).toarray()
X=X[:,1:]

#Splitting the dataset into Training and Test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=0)

#Fitting xgboost to the training set

from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train,y_train)
