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

dataFolder = 'Data\\'

train = pd.read_csv(dataFolder+'train.csv')
test = pd.read_csv(dataFolder+'test.csv')

# %% Seprate the data into X and Y
trainY = train[['SalePrice']]
trainX = train.drop('SalePrice', axis=1)
test = test.set_index('Id')
trainX = trainX.set_index('Id')

# %% XGBoost dealds well with NAs, but labelencoder doesn't, so lets deal with
# the object values (see EDA analysis).

#nanCols = pd.isnull(train).sum()
#nanCols = nanCols[nanCols > 0]
#nanObj = train[nanCols.index].select_dtypes(include=['object']).copy()
#modal = ['MasVnrType', 'Electrical']
#for col in modal:
#    train[col].fillna(train[col].mode().values[0], inplace=True)
#    test[col].fillna(train[col].mode().values[0], inplace=True)
#
#nanObj.drop(modal, axis=1, inplace=True)
#
#for col in nanObj.columns:
#    train[col].fillna(0, inplace=True)
#    test[col].fillna(0, inplace=True)


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

# %% Fit test data for submission

submission = pd.DataFrame(test.index)
submission['SalePrice'] = model.predict(test)
subFolder = 'C:\\Users\\George\\OneDrive - University Of Cambridge\\Others\\Machine learning\\Kaggle\\House prices\\Output\\'
submission.to_csv(subFolder+'XGBoost2.csv', index=False)
