# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 18:14:41 2018

@author: George

Start with a rudimental clean and fit to test performance
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

dataFolder = 'C:\\Users\\George\\OneDrive - University Of Cambridge\\Others\\Machine learning\\Kaggle\\House prices\\Data\\'

train = pd.read_csv(dataFolder+'train.csv')
test = pd.read_csv(dataFolder+'test.csv')
trainY = train[['SalePrice']]
trainX = train.drop('SalePrice', axis=1)
test = test.set_index('Id')
trainX = trainX.set_index('Id')

# For now lets just drop all columns with NaN values and later I will identify
# how to deal with each column
trainSize = trainX.shape[0]
X = trainX.append(test)
X = X.dropna(axis=1, how='any')
trainX = X.loc[0:trainSize]
test = X.loc[trainSize:]


def encodeLabels(X):
    """Encode non-numeric data types as numeric"""
    Xobj = X.select_dtypes(include=['object']).copy()
    Xval = X.select_dtypes(exclude=['object']).copy()
    le = LabelEncoder()
    for column in Xobj:
        le.fit(Xobj[column])
        Xobj[column] = le.transform(Xobj[column])
    X = pd.concat([Xval, Xobj], axis=1)
    return X


test = encodeLabels(test)
trainX = encodeLabels(trainX)
