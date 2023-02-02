import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from scipy.stats import norm

import matplotlib.pyplot as plt

from utils import scrape_nist_data


FLUID = "CO2"
TEMPERATURE_RANGE = np.arange(5, 30 + 5, 1)  # degC
PRESSURE_RANGE = np.arange(3, 46 + 1, 1)  # bar
N_TRAIN_SPLITS = 200

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
])

Y = df['Compressibility Factor'].to_numpy() - 1

rel_err = []
for i in range(N_TRAIN_SPLITS):
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.3, train_size=0.7)
    a_lsq = np.linalg.pinv(X_train.T.dot(X_train)).dot(
        X_train.T).dot(Y_train)  # least squares solution
    Z_pred = 1 + a_lsq.dot(X_test.T)
    Z_test = Y_test + 1
    rel_err.append(np.mean(100 * np.abs(Z_pred - Z_test) / (Z_test)))
rel_err = np.asarray(rel_err)

# console output
print(f"Relative Error (%) = {rel_err.mean():.3e} ± {rel_err.std() * 1.96:.3e} (k=1.96)")

plt.hist(rel_err, bins=30)
plt.title(
    f"MAPE Distribution (%): {rel_err.mean():.2e} ± {rel_err.std() * 1.96:.2e} (k=1.96)")
plt.savefig(f"figures/error_distribution_{FLUID}.png")
plt.show()
