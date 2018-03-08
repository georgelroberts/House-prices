# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 17:24:32 2018

@author: George

EDA
TODO: Any correlations between features? (multicollinearity) - maybe use ridge
         regression
TODO: Remove missing data and create labels
TODO: Any significant differences between test and train?
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import HuberRegressor
import seaborn as sns

dataFolder = 'C:\\Users\\George\\OneDrive - University Of Cambridge\\Others\\Machine learning\\Kaggle\\House prices\\Data\\'

train = pd.read_csv(dataFolder+'train.csv')
test = pd.read_csv(dataFolder+'test.csv')

print(train.info())

#%% Lets first look at the distribution of sale prices. They appear right
# skewed. This will be easier to see in a box plot.

fig, ax = plt.subplots()
ax.hist(train.SalePrice, bins=20)

#%% There are a surprising number of outliers at the expensive range, and there
# doesn't appear to be any trend with the year.

train[['SalePrice', 'YrSold']].boxplot(by='YrSold')

#%% From a human perspective, the living space looks like the strongest
# indicator of price, lets see whether a basic fit can be made. Need to use
# HuberRegressor because it is more robust to outliers.

fig, ax = plt.subplots()
ax.scatter(train.GrLivArea, train.SalePrice)
clf = HuberRegressor()
clf.fit(train.GrLivArea.values.reshape(-1, 1),
        train.SalePrice.values.reshape(-1, 1))
salePredictGrLivArea = clf.predict(train.GrLivArea.values.reshape(-1, 1))
ax.plot(train.GrLivArea.values.reshape(-1, 1),
        clf.predict(train.GrLivArea.values.reshape(-1, 1)))
plt.ylim([0, 800000])
plt.show()

#%% Lets look at correlations in the dataset (at least between numeric values)
# We only care about correlations with sale price, so lets visualise that.
# It turns out a number of variables have a large positive correlation
#  with salePrice, with overall quality ranking higher than living area.

corr = train.corr()
corr = corr['SalePrice']
corr.drop('SalePrice', axis=0, inplace=True)
corr = corr.sort_values(ascending=False)
fig, ax = plt.subplots()
corrPlot = sns.barplot(x=corr.index, y=corr.values)
plt.xticks(rotation=45);



