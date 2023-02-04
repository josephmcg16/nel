import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from scipy.stats import norm

import matplotlib.pyplot as plt

from utils import scrape_nist_data


FLUID = "CO2"
TEMPERATURE_RANGE = np.arange(5, 30 + 5, 1)  # degC
PRESSURE_RANGE = np.arange(3, 46 + 1, 1)  # bar

df = scrape_nist_data(FLUID, TEMPERATURE_RANGE, PRESSURE_RANGE)
print(f"Number of Samples: {df.shape[0]}")

pi = df['pi'].to_numpy()
tau = df['tau'].to_numpy()

X = np.asarray([
    pi * tau ** 0.75,
    pi * tau ** 1.25,
    pi * tau ** 3,
    pi ** 2 * tau ** 2,
    pi ** 3 * tau ** 1.5,
    pi ** 3 * tau ** 1.75,
]).T

Y = df['Compressibility Factor'].to_numpy()
