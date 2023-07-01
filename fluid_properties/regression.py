from multiprocessing import Pipe
import os
import pandas as pd
from sklearn import model_selection

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import plotly.express as px

from fluid_properties.generate_data import generate_data

class DataLoader:
    def __init__(self, config):
        self.config = config

    def load_data(self):
        if self.config["path"] is not None:
            path = self.config["path"]
            if os.path.isfile(path):
                print(f"Existing data found: {os.getcwd()}/{path}")
                df = pd.read_csv(path)
            else:
                raise FileNotFoundError(f"Existing data not found: '{os.getcwd()}/{path}'."
                                        "Pass in parameters to generate data or a specific path to the data.")
        else:
            df = generate_data(self.config)
            df.to_csv(f'data/{self.config["FLUID"]}_nist_data.csv', index=False)
        return df


class FluidDensityPredictor:
    def __init__(self, config=None, model=None):
        self.config = config
        if model is None:
            POLY_DEGREE = self.config["poly_degree"] if self.config["poly_degree"] is not None else 11
            model = Pipeline([
                ('scaler', StandardScaler()), 
                ('poly', PolynomialFeatures(degree=POLY_DEGREE)),
                ('linear', LinearRegression(fit_intercept=False))])
        self.model = model

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


def calculate_error(y_true, y_pred):
    return (y_true - y_pred) ** 2
