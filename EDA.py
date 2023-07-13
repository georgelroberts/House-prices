# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 17:24:32 2018

@author: George

TODO: Remove missing data and create labels
TODO: Any significant differences between test and train?
"""


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import HuberRegressor
import seaborn as sns

dataFolder = 'Data\\'

train = pd.read_csv(f'{dataFolder}train.csv')
test = pd.read_csv(f'{dataFolder}test.csv')

print(train.info())

# %% Lets first look at the distribution of sale prices. They appear right
# skewed. This will be easier to see in a box plot.

fig, ax = plt.subplots()
ax.hist(train.SalePrice/1e3, bins=30, rwidth=0.9)
plt.xlabel('Sale price (1000$)')
plt.ylabel('Number of houses')
plt.show()

# %% There are a surprising number of outliers at the expensive range, and there
# doesn't appear to be any trend with the year.

fig, ax = plt.subplots()
train[['SalePrice', 'YrSold']].boxplot(by='YrSold', column='SalePrice', ax=ax)
plt.xlabel('Year sold')
plt.ylabel('Price ($)')
plt.suptitle("")
plt.show()

# %% From a human perspective, the living space looks like the strongest
# indicator of price, lets see whether a basic fit can be made. Need to use
# HuberRegressor because it is more robust to outliers.

fig, ax = plt.subplots()
ax.scatter(train.GrLivArea, train.SalePrice, alpha=0.2, label='Real data')
clf = HuberRegressor()
clf.fit(train.GrLivArea.values.reshape(-1, 1),
        train.SalePrice.values.reshape(-1, 1))
salePredictGrLivArea = clf.predict(train.GrLivArea.values.reshape(-1, 1))
ax.plot(train.GrLivArea.values.reshape(-1, 1),
        clf.predict(train.GrLivArea.values.reshape(-1, 1)),
        'black', label='Linear fit')
plt.xlabel('Living area')
plt.ylabel('Price ($)')
plt.legend()
plt.show()

# %% Lets look at correlations in the dataset (at least between numeric values)
# We only care about correlations with sale price, so lets visualise that.
# It turns out a number of variables have a large positive correlation
#  with salePrice, with overall quality ranking higher than living area.

corr = train.corr()
corr = corr['SalePrice']
corr.drop('SalePrice', axis=0, inplace=True)
corr = corr.sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(12,6))
corrPlot = sns.barplot(x=corr.index, y=corr.values)
plt.xticks(rotation=45);
plt.xlabel('House feature')
plt.ylabel('Correlation')
plt.show()

# %% We will now decide what to do with columns with null values in. This is
# done by analysing the data and also looking at the data descriptions. 
# Actually, XGBoost naturally deals with missing data, so this is unnecessary,
# at least for this fit. 

nanCols = pd.isnull(train).sum()
nanCols = nanCols[nanCols>0]
nanCols = nanCols.sort_values(ascending=False)

print(
    f'There are {train.shape[1] - 2} features in the training set, {nanCols.shape[0]} of which have some null values.'
)

fig, ax = plt.subplots()
corrPlot = sns.barplot(x=nanCols.index, y=nanCols.values)
plt.xticks(rotation=45);
plt.xlabel('House feature')
plt.ylabel('Number of Nans')
plt.show()

# First deal with the object-type NaNs.
nanObj = train[nanCols.index].select_dtypes(include=['object']).copy()
# Set modal value for: MasVnrType, Electrical. These have very few missing
# values
# Set to zero for: PoolQC, MiscFeature, Alley, Fence, FireplaceQu, GarageType,
#                  GarageFinish, GarageQual, GarageCond, BsmtFinType2,
#                  BsmtExposure, BsmtFinType1, BsmtCond, BsmtQual. These Nans
# are due to the coding used, so they do actually convey information.

modal = ['MasVnrType', 'Electrical']
for col in modal:
    train[col].fillna(train[col].mode().values[0], inplace=True)
    test[col].fillna(train[col].mode().values[0], inplace=True)

nanObj.drop(modal, axis=1, inplace=True)

for col in nanObj.columns:
    train[col].fillna(0, inplace=True)
    test[col].fillna(0, inplace=True)

# nanCols that are numeric. Are the numbers missing for a reason or can we just
# set the value equal to the average of the whole column? Check for a disparity
# with salePrice for each.

nanVal = train[nanCols.index].select_dtypes(exclude=['object']).copy()

for col in nanVal.columns:
    nanColComp = train[[col,'SalePrice']]
    nullAvg = nanColComp[pd.isnull(nanColComp[col])].SalePrice.mean()
    otherAvg = nanColComp[pd.notnull(nanColComp[col])].SalePrice.mean()
    print('For column {}, the average sale price for NaNs is {:.0f}, where for non-NaNs is {:.0f}'
          .format(col, nullAvg, otherAvg))

# Lets let xgboost deal with all rows where GarageYrBlt and MasVnrArea are
# NaNs, because it looks like setting it to an average value will skew the
# results. There aren't many values for these anyway. Lets average LotFrontage

train['LotFrontage'].fillna(train.LotFrontage.mean(), inplace=True)
test['LotFrontage'].fillna(train.LotFrontage.mean(), inplace=True)
