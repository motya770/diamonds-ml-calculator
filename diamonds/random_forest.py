import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn
import statsmodels.api as sm
import sklearn.preprocessing
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
from sklearn.preprocessing import LabelEncoder

filename = '/Users/kudelin/Desktop/work/projects/dstat/diamonds4.csv'
#filename = '/Users/kudelin/Desktop/work/projects/dstat/diamonds7.csv'

data = pd.read_csv(filename)
#print(data.shape)

bos = pd.DataFrame(data);
bos.columns = ['PRICE', 'Date', 'PricePerCarat', 'Carat', 'ExternalId', 'Shape', 'Clarity',
               'Color', 'Culet', 'Cut', 'Depth', 'Fluorescence', 'LxwRatio', 'Polish', 'Symmetry']
bos = pd.DataFrame(data)

le = sklearn.preprocessing.LabelEncoder()

le.fit(bos["Shape"])
print(list(le.classes_))
shape = le.transform(bos["Shape"])

le.fit(bos["Fluorescence"])
print(list(le.classes_))
fluorescence = le.transform(bos["Fluorescence"])

le.fit(bos["Clarity"])
print(list(le.classes_))
clarity = le.transform(bos["Clarity"])

le.fit(bos["Color"])
print(list(le.classes_))
color = le.transform(bos["Color"])

le.fit(bos["Culet"])
print(list(le.classes_))
culet = le.transform(bos["Culet"])

le.fit(bos["Cut"])
print(list(le.classes_))
cut = le.transform(bos["Cut"])

le.fit(bos["Polish"])
print(list(le.classes_))
polish = le.transform(bos["Polish"])

le.fit(bos["Symmetry"])
print(list(le.classes_))
symmetry = le.transform(bos["Symmetry"])

bos["Shape"] = shape
bos["Clarity"] = clarity
bos["Fluorescence"] = fluorescence
bos["Clarity"] = clarity
bos["Color"] = color
bos["Culet"] = culet
bos["Cut"] = cut
bos["Polish"] = polish
bos["Symmetry"] = symmetry

X = bos.drop('PRICE', axis = 1).drop("ExternalId", axis = 1).drop("Date", axis=1).drop("PricePerCarat", axis=1)
Y = bos['PRICE']

"""
#SCALING IN CASE TO MAKE ALL PARAMS AROUND 0

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
X = pd.DataFrame(sc_x.fit_transform(X))
Y = pd.DataFrame(sc_y.fit_transform(Y.reshape(-1, 1)))
"""

#sns.set(style='whitegrid', context='notebook')
#sns.pairplot(bos[bos.columns], size=10);
#plt.show()

#329,Sep 22,1430,0.23,LD06945716,Round,SI1,H,None,Very Good,62.3,None,1.01,Good,Good

print(Y.head())
print(Y.describe())

print(X.head())
print(X.describe())

##print(bos.head())
print("++++++++++ DESCIPTION +++++++++++++")

print("++++++++++ ATTRIBUTES CORR +++++++++++++")
pearson = X.corr(method='pearson')
print(pearson)
# assume target attr is the last, then remove corr with itself
corr_with_target = pearson.ix[-1][:-1]
# attributes sorted from the most predictive
predictivity = corr_with_target[np.abs(corr_with_target).argsort()[::-1]]

print("+++++++++++PREDICTIVITY+++++++++++")
print(predictivity)

"""
sns.set(font_scale=1.5)
hm = sns.heatmap(pearson,
                 cbar=True,
                 annot=True,
                 square=True,
                 fmt='.2f',
                 annot_kws={'size': 15},
                 yticklabels=X.columns,
                 xticklabels=X.columns)
plt.show()
"""

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.33, random_state=5) #sklearn.cross_validation.train_test_split(X, Y, test_size = 0.33, random_state = 10)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

lm = RandomForestRegressor(n_estimators=1000,
  criterion='mse',
  random_state=1,
  n_jobs=-1)

results = lm.fit(X_train, Y_train)

#Y_train_pred = lm.predict(X_train)
Y_pred = lm.predict(X_test)

plt.scatter(Y_test, Y_pred)
plt.xlabel("Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")
plt.show()

mse1 = sklearn.metrics.mean_squared_error(Y_test, Y_pred)
print("RMSE 1.0")
print(np.sqrt(mse1))
print('Variance score: %.2f' % r2_score(Y_pred, Y_test))

#plt.show()

#mse2 = sklearn.metrics.mean_squared_error(Y_train_pred, Y_train)
#print("MSE 2.0 SQUARE+++++")
#print(np.sqrt(mse2))
#print('Variance score: %.2f' % r2_score(Y_train_pred, Y_train))