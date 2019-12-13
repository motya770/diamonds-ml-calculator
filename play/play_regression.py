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
from sklearn.cross_validation import train_test_split
from sklearn import svm

boston = load_boston()

print(boston.keys())
print(boston.feature_names)

bos = pd.DataFrame(boston.data)
bos.columns = boston.feature_names
bos['PRICE'] = boston.target


##print(bos.head())
print("++++++++++ DESCIPTION +++++++++++++")
print(bos.describe())

print("++++++++++ ATTRIBUTES CORR +++++++++++++")
print(bos.corr(method='pearson'))

pearson = bos.corr(method='pearson')
# assume target attr is the last, then remove corr with itself
corr_with_target = pearson.ix[-1][:-1]
# attributes sorted from the most predictive
predictivity = corr_with_target[np.abs(corr_with_target).argsort()[::-1]]

print("+++++++++++PREDICTIVITY+++++++++++")
print(predictivity)



X = bos.drop('PRICE', axis = 1)
Y = bos['PRICE']

X_train, X_test, Y_train, Y_test = sklearn.cross_validation.train_test_split(X, Y, test_size = 0.33, random_state = 5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

lm = LinearRegression()
results = lm.fit(X_train, Y_train)

Y_pred = lm.predict(X_test)

plt.scatter(Y_test, Y_pred)
plt.xlabel("Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")
#plt.show()

mse = sklearn.metrics.mean_squared_error(Y_test, Y_pred)
print(np.sqrt(mse))

print(results.score(X_train, Y_train))
#print(sklearn.metrics.accuracy_score(Y_test, Y_pred))

print("TEST END")


