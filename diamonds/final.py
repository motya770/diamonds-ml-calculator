import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn
import statsmodels.api as sm
import sklearn.preprocessing
import seaborn as sns
import pandas.tseries

sns.set_style("whitegrid")
sns.set_context("poster")

# special matplotlib argument for improved plots
from matplotlib import rcParams

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder

filename = '/Users/kudelin/Desktop/work/projects/dstat/diamonds4.csv'
#filename = '/Users/kudelin/Desktop/work/projects/dstat/diamonds7.csv'

data = pd.read_csv(filename)

bos = pd.DataFrame(data);
bos.columns = ['PRICE', 'Date', 'PricePerCarat', 'Carat', 'ExternalId', 'Shape', 'Clarity',
               'Color', 'Culet', 'Cut', 'Depth', 'Fluorescence', 'LxwRatio', 'Polish', 'Symmetry']
bos = pd.DataFrame(data)
le = sklearn.preprocessing.LabelEncoder()

le.fit(bos["Shape"])
shape = le.transform(bos["Shape"])

le.fit(bos["Fluorescence"])
fluorescence = le.transform(bos["Fluorescence"])

le.fit(bos["Clarity"])
clarity = le.transform(bos["Clarity"])

le.fit(bos["Color"])
color = le.transform(bos["Color"])

le.fit(bos["Culet"])
culet = le.transform(bos["Culet"])

le.fit(bos["Cut"])
cut = le.transform(bos["Cut"])

le.fit(bos["Polish"])
polish = le.transform(bos["Polish"])


le.fit(bos["Symmetry"])
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
Y_pred = lm.predict(X_test)

mse1 = sklearn.metrics.mean_squared_error(Y_test, Y_pred)
print("RMSE 1.0")
print(np.sqrt(mse1))
print('Variance score: %.2f' % r2_score(Y_pred, Y_test))
