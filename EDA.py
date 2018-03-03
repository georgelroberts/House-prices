# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 17:24:32 2018

@author: George

EDA
TODO: Overview of sale prices
TODO: Sale prices over time
TODO: Any significant differences between test and train?

Data cleaning

Fitting
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import HuberRegressor

dataFolder = 'C:\\Users\\George\\OneDrive - University Of Cambridge\\Others\\Machine learning\\Kaggle\\House prices\\Data\\'

train = pd.read_csv(dataFolder+'train.csv')
test = pd.read_csv(dataFolder+'test.csv')

print(train.info())

# Lets first look at the distribution of sale prices. They appear left skewed.
# This will be easier to see in a box plot.

fig, ax = plt.subplots()
ax.hist(train.SalePrice, bins=20)

# There are a surprising number of outliers at the expensive range, and there
# doesn't appear to be any trend with the year.

train[['SalePrice', 'YrSold']].boxplot(by='YrSold')

# From a human perspective, the lot area looks like the strongest indicator of
# price, lets see whether a basic fit can be made. Need to use HuberRegressor
# because it is more robust to outliers

fig, ax = plt.subplots()
ax.scatter(train.LotArea, train.SalePrice)
clf = HuberRegressor()
clf.fit(train.LotArea.reshape(-1, 1), train.SalePrice.reshape(-1, 1))
salePredictLotArea = clf.predict(train.LotArea.reshape(-1, 1))
ax.plot(train.LotArea.reshape(-1, 1), clf.predict(train.LotArea.reshape(-1, 1)))
plt.ylim([0, 800000])
