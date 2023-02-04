import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
import matplotlib.pyplot as plt

from utils import scrape_nist_data

# Define fluid and temperature and pressure ranges to scrape data for
FLUID = "CO2"
TEMPERATURE_RANGE = np.arange(5, 30 + 5, 1)  # degC
PRESSURE_RANGE = np.arange(3, 46 + 1, 1)  # bar

# Scrape data for defined fluid and temperature/pressure ranges
df = scrape_nist_data(FLUID, TEMPERATURE_RANGE,
                      PRESSURE_RANGE, sat_curve_filter=0, callback=True)
print(f"Number of Samples: {df.shape[0]}")

# Define number of splits for KFold
N_SPLITS = df.shape[0]  # leave-one-out split

# Extract normalized pressure and temperature arrays from dataframe
pi = df['pi'].to_numpy()
tau = df['tau'].to_numpy()

# Create feature matrix from normalized pressure and temperature
X = np.asarray([
    pi * tau ** 0.75,
    pi * tau ** 1.25,
    pi * tau ** 3,
    pi ** 2 * tau ** 2,
    pi ** 3 * tau ** 1.5,
    pi ** 3 * tau ** 1.75,
]).T

# Extract response variable from dataframe
Y = df['Compressibility Factor'].to_numpy() - 1

# Estimate error using KFold cross validation
kf = KFold(n_splits=N_SPLITS)
MSE = []
for train_index, test_index in kf.split(X):
    # Split data into training and testing sets
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    # Calculate least squares solution for training data
    a_lsq = np.linalg.pinv(X_train.T.dot(X_train)).dot(X_train.T).dot(Y_train)

    # Predict response variable for testing data
    Z_pred = 1 + a_lsq.dot(X_test.T)
    Z_test = Y_test + 1

    # Calculate mean squared error for prediction
    MSE.append(mean_squared_error(Z_test, Z_pred))

# Convert list of MSE to numpy array
MSE = np.asarray(MSE)

# Calculate least squares solution for all data
a_lsq = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)

# Add predicted Z and relative errors to the dataframe
df['Predicted Compressbility Factor'] = 1 + a_lsq.dot(X.T)
df['Relative Error (%)'] = 100 * (df['Predicted Compressbility Factor'] -
                                  df['Compressibility Factor']) / df['Compressibility Factor']


# Output optimal coeffecients and error metrics to the console
for i, a in enumerate(a_lsq):
    print(f"a{i+1} = {a:.3e}")
print(f"\nMeanRelErr = {df['Relative Error (%)'].abs().mean():.3e} %")
print(f"MaxRelErr = {df['Relative Error (%)'].abs().max():.3e} %")
print(f"MinRelErr = {df['Relative Error (%)'].abs().min():.3e} %\n")

# Calculate the points to be plotted in the PDF
pdf_points = np.linspace(MSE.mean() - 2 * MSE.std(),
                         MSE.mean() + 2 * MSE.std(), 200)

# Calculate the PDF based on the points and mean and standard deviation of MSE
pdf = norm.pdf(pdf_points, MSE.mean(), MSE.std())

# Plot the histogram of MSE
plt.hist(MSE, bins=100, density=True)

# Plot the PDF
# plt.plot(pdf_points, pdf)

plt.title(
    f"MSE = {MSE.mean():.2e} ± {MSE.std() * 1.96:.2e} (k=1.96)\n{N_SPLITS} k-folds")
plt.xlabel("MSE")
plt.ylabel("Relative Frequency")
plt.savefig(f"figures/mse_distribution_{FLUID}.png")
plt.show()