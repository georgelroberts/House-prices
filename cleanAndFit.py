# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 18:14:41 2018

@author: George

Start with a rudimental clean and fit to test performance
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.model_selection import train_test_split

dataFolder = 'C:\\Users\\George\\OneDrive - University Of Cambridge\\Others\\Machine learning\\Kaggle\\House prices\\Data\\'

train = pd.read_csv(dataFolder+'train.csv')
test = pd.read_csv(dataFolder+'test.csv')

# %% Seprate the data into X and Y
trainY = train[['SalePrice']]
trainX = train.drop('SalePrice', axis=1)
test = test.set_index('Id')
trainX = trainX.set_index('Id')

# %% For now lets just drop all columns with NaN values and later I will
# identify how to deal with each column
trainSize = trainX.shape[0]
X = trainX.append(test)
X = X.dropna(axis=1, how='any')
trainX = X.loc[0:trainSize]
test = X.loc[trainSize:]

# %% Label encoding for non-numeric data


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

# %% Basic fit on data - Kaggle uses root mean squared log error, so this is
# implemented and an xgboost regressor is used

trainX2, CVX, trainY2, CVY = train_test_split(
        trainX, trainY, test_size=0.33, random_state=42)

model = xgb.XGBRegressor()
model.fit(trainX2, trainY2)
CVYfit = model.predict(CVX)


def rmsle(y, y0):
    """ Thanks to https://www.kaggle.com/jpopham91/rmlse-vectorized """
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(y0), 2)))


CVError = rmsle(CVY, CVYfit.reshape(-1, 1)).values[0]
