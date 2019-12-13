import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn
import statsmodels.api as sm

import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("poster")

# special matplotlib argument for improved plots
from matplotlib import rcParams

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score

boston = load_boston()

print(boston.keys())
print(boston.feature_names)

bos = pd.DataFrame(boston.data)
bos.columns = boston.feature_names
bos['PRICE'] = boston.target

print(bos.head())
print(bos.describe())

X = bos.drop('PRICE', axis = 1)
Y = bos['PRICE']

X_train, X_test, Y_train, Y_test = sklearn.cross_validation.train_test_split(X, Y, test_size = 0.33, random_state = 5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

lm = RandomForestRegressor(n_estimators=1000,
  criterion='mse',
  random_state=1,
  n_jobs=-1)
results = lm.fit(X_train, Y_train)

Y_train_pred = lm.predict(X_train)
Y_pred = lm.predict(X_test)

plt.scatter(Y_test, Y_pred)
plt.xlabel("Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")


mse1 = sklearn.metrics.mean_squared_error(Y_test, Y_pred)
print("MSE 1.0 SQUARE+++++")
print(np.sqrt(mse1))
print('Variance score: %.2f' % r2_score(Y_pred, Y_test))
#plt.show()

mse2 = sklearn.metrics.mean_squared_error(Y_train_pred, Y_train )
print("MSE 2.0 SQUARE+++++")
print(np.sqrt(mse2))
print('Variance score: %.2f' % r2_score(Y_train_pred, Y_train))