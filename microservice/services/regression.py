import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn
import statsmodels.api as sm
import sklearn.preprocessing
import seaborn as sns
import pandas.tseries

from flask_injector import inject
from pystat.microservice.dao.repository import DiamondRepository

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
from collections import defaultdict

sns.set_style("whitegrid")
sns.set_context("poster")

class RegressionProvider(object):

    def prepareData(self, data) -> pd.DataFrame:
        bos = pd.DataFrame(data)
        bos.columns = ['PRICE', 'Date', 'PricePerCarat', 'Carat', 'ExternalId', 'Shape', 'Clarity',
                       'Color', 'Culet', 'Cut', 'Depth', 'Fluorescence', 'LxwRatio', 'Polish', 'Symmetry']

        shapeEncoder = sklearn.preprocessing.LabelEncoder()
        shapeEncoder.fit(bos["Shape"])
        shape = shapeEncoder.transform(bos["Shape"])
        self._shapeEncoder = shapeEncoder

        fluorescenceEncoder = sklearn.preprocessing.LabelEncoder()
        fluorescenceEncoder.fit(bos["Fluorescence"])
        fluorescence = fluorescenceEncoder.transform(bos["Fluorescence"])
        self._fluorescenceEncoder = fluorescenceEncoder

        clarityEncoder = sklearn.preprocessing.LabelEncoder()
        clarityEncoder.fit(bos["Clarity"])
        clarity = clarityEncoder.transform(bos["Clarity"])
        self._clarityEncoder = clarityEncoder

        colorEncoder = sklearn.preprocessing.LabelEncoder()
        colorEncoder.fit(bos["Color"])
        color = colorEncoder.transform(bos["Color"])
        self._colorEncoder = colorEncoder

        culetEncoder = sklearn.preprocessing.LabelEncoder()
        culetEncoder.fit(bos["Culet"])
        culet = culetEncoder.transform(bos["Culet"])
        self._culetEncoder = culetEncoder

        cutEncoder = sklearn.preprocessing.LabelEncoder()
        cutEncoder.fit(bos["Cut"])
        cut = cutEncoder.transform(bos["Cut"])
        print(culetEncoder)
        self._cutEncoder = cutEncoder

        polishEncoder = sklearn.preprocessing.LabelEncoder()
        polishEncoder.fit(bos["Polish"])
        polish = polishEncoder.transform(bos["Polish"])
        self._polishEncoder = polishEncoder

        symmetryEncoder = sklearn.preprocessing.LabelEncoder()
        symmetryEncoder.fit(bos["Symmetry"])
        symmetry = symmetryEncoder.transform(bos["Symmetry"])
        self._symmetryEncoder = symmetryEncoder

        bos["Shape"] = shape
        bos["Clarity"] = clarity
        bos["Fluorescence"] = fluorescence
        bos["Clarity"] = clarity
        bos["Color"] = color
        bos["Culet"] = culet
        bos["Cut"] = cut
        bos["Polish"] = polish
        bos["Symmetry"] = symmetry
        return bos

    def __init__(self):

        #filename = '/Users/kudelin/Desktop/work/projects/dstat/diamonds4.csv'
        # filename = '/Users/kudelin/Desktop/work/projects/dstat/diamonds7.csv'

        #data = pd.read_csv(filename)

        diamond_repository = DiamondRepository();
        diamonds = diamond_repository.findAll()
        data = []

        ['PRICE', 'Date', 'PricePerCarat', 'Carat', 'ExternalId', 'Shape', 'Clarity',
         'Color', 'Culet', 'Cut', 'Depth', 'Fluorescence', 'LxwRatio', 'Polish', 'Symmetry']
        for d in diamonds:
            item = [d['price'], d['date'], d['pricePerCarat'], d['carat'], d['_id'], d['shape'],
                    d['clarity'], d['color'], d['culet'], d['cut'], d['depth'], d['fluorescence'],
                    d['lxwRatio'], d['polish'], d['symmetry']]
            data.append(item)
            ##print(d)

        bos = self.prepareData(data)

        X = bos.drop('PRICE', axis=1).drop("ExternalId", axis=1).drop("Date", axis=1).drop("PricePerCarat", axis=1)
        print(X.head())

        Y = bos['PRICE']

        X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.33,
                                                                                    random_state=5)  # sklearn.cross_validation.train_test_split(X, Y, test_size = 0.33, random_state = 10)
        print(X_train.shape)
        print(X_test.shape)
        print(Y_train.shape)
        print(Y_test.shape)

        model = RandomForestRegressor(n_estimators=1000,
                                   criterion='mse',
                                   random_state=1,
                                   n_jobs=-1)
        results = model.fit(X_train, Y_train)

        Y_pred = model.predict(X_test)

        mse1 = sklearn.metrics.mean_squared_error(Y_test, Y_pred)
        print("RMSE 1.0")
        print(np.sqrt(mse1))
        print('Variance score: %.2f' % r2_score(Y_pred, Y_test))

        self._model = model
    def calculate(self, args) -> str:
        print("calculate")

        result = self._model.predict([args])

        print("prediction result" + str(result))
        return "{'result':" + str(result) + "}"

    def buildArgs(self, carat, shape, clarity, color, culet, cut, depth, fluorescence, lxwRatio, polish, symmetry)->[]:

        shape = self._shapeEncoder.transform([shape])
        clarity = self._clarityEncoder.transform([clarity])
        color = self._colorEncoder.transform([color])
        culet = self._culetEncoder.transform([culet])
        cut = self._cutEncoder.transform([cut])
        fluorescence = self._fluorescenceEncoder.transform([fluorescence])
        polish = self._polishEncoder.transform([polish])
        symmetry = self._symmetryEncoder.transform([symmetry])

        args = [carat, shape, clarity, color, culet, cut, depth, fluorescence, lxwRatio, polish, symmetry]
        #print("shape: " + args.shape())
        return args