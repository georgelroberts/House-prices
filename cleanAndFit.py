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
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

dataFolder = 'Data\\'

train = pd.read_csv(dataFolder+'train.csv')
test = pd.read_csv(dataFolder+'test.csv')

# %% Seprate the data into X and Y
trainY = train[['SalePrice']]
trainX = train.drop('SalePrice', axis=1)
test = test.set_index('Id')
trainX = trainX.set_index('Id')

# %% Label encoding for non-numeric data. Fit the train and test together so
# LabelEncoder doesn't throw an error


def encodeLabels(XTrain, XTest):
    """Encode non-numeric data types as numeric"""
    trainSize = XTrain.shape[0]
    X = XTrain.append(XTest)

    Xobj = X.select_dtypes(include=['object']).astype(str).copy()
    Xval = X.select_dtypes(exclude=['object']).copy()

    le = LabelEncoder()
    for column in Xobj:
        le.fit(Xobj[column])
        Xobj[column] = le.transform(Xobj[column])
    X = pd.concat([Xval, Xobj], axis=1)
    XTrain = X.loc[0:trainSize]
    XTest = X.loc[trainSize+1:]

    return XTrain, XTest


trainX, test = encodeLabels(trainX, test)

# %% Basic fit on data - Kaggle uses root mean squared log error, so this is
# implemented and an xgboost regressor is used.
# TODO: Use k-fold splitting to find the best hyperparameters

max_depth = np.linspace(1, 10, 10)
depthVsError = []
for depth in max_depth:

    n_splits = 3
    CVError = []

    skf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_index, test_index in skf.split(trainX, trainY):
        trainX2, CVX = trainX.iloc[train_index], trainX.iloc[test_index]
        trainY2, CVY = trainY.iloc[train_index], trainY.iloc[test_index]

        model = xgb.XGBRegressor(max_depth=int(depth))
        model.fit(trainX2, trainY2)
        CVYfit = model.predict(CVX)

        def rmsle(y, y0):
            """ Thanks to https://www.kaggle.com/jpopham91/rmlse-vectorized """
            assert len(y) == len(y0)
            return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(y0), 2)))

        CVError.append(rmsle(CVY, CVYfit.reshape(-1, 1)).values[0])
    CVError = np.mean(CVError)
    depthVsError.append(CVError)

fig, ax = plt.subplots()
plt.plot(max_depth, depthVsError)
# %% Fit test data for submission

submission = pd.DataFrame(test.index)
submission['SalePrice'] = model.predict(test)
subFolder = 'C:\\Users\\George\\OneDrive - University Of Cambridge\\Others\\Machine learning\\Kaggle\\House prices\\Output\\'
submission.to_csv(subFolder+'XGBoost2.csv', index=False)
