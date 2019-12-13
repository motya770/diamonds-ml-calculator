#! /usr/bin/env python
from flask import Flask

from flask_injector import inject
from injector import Binder
from flask_injector import FlaskInjector
from pystat.microservice.services.provider import ItemsProvider
from pystat.microservice.services.regression import RegressionProvider
from pystat.microservice.dao.repository import DiamondRepository
from flask import request

def configure(binder: Binder) -> Binder:
    binder.bind(
        ItemsProvider,
        ItemsProvider())
    binder.bind(
        RegressionProvider,
        RegressionProvider())

app = Flask(__name__)

@app.route("/search")
@inject
def search(data_provider: ItemsProvider) -> list:
    return data_provider.get()

@app.route("/calculate")
@inject
def calculate(regression_provider: RegressionProvider) -> str:

    carat = request.args.get('carat', '')
    shape = request.args.get('shape', '').replace("%20", " ")
    clarity = request.args.get('clarity', '').replace("%20", " ")
    color = request.args.get('color', '').replace("%20", " ")
    culet = request.args.get('culet', '').replace("%20", " ")
    cut = request.args.get('cut', '').replace("%20", " ")
    depth = request.args.get('depth', '')
    fluorescence = request.args.get('fluorescence', '').replace("%20", " ")
    lxwRatio = request.args.get('lxwRatio', '')
    polish = request.args.get('polish', '').replace("%20", " ")
    symmetry = request.args.get('symmetry', '').replace("%20", " ")

    args = regression_provider.buildArgs(carat, shape, clarity, color, culet, cut, depth, fluorescence, lxwRatio, polish, symmetry);

    return regression_provider.calculate(args)

if __name__ == '__main__':
    FlaskInjector(app, modules=[configure])
    app.run(debug=True, use_reloader=False)