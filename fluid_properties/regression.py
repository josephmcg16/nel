import os
import pandas as pd
import numpy as np

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

import plotly.express as px
import plotly.graph_objects as go

from fluid_properties.generate_data import generate_data


class DataLoader:
    def __init__(self, config):
        self.config = config

        x_feature = config['x_feature']
        y_feature = config['y_feature']
        z_feature = config['z_feature']

        self.df = self.load_data()
        X = self.df[[x_feature, y_feature]].to_numpy()
        Y = self.df[z_feature].to_numpy()

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            X, Y, test_size=self.config["test_size"], random_state=self.config['random_state'], shuffle=True)

    def load_data(self):
        if "data_path" in self.config:
            if os.path.isfile(self.config['data_path']):
                print(
                    f"Existing data found: {os.getcwd()}/{self.config['data_path']}")
                return pd.read_csv(self.config['data_path'])
            else:
                raise FileNotFoundError(f"Existing data not found: '{os.getcwd()}/{self.config['data_path']}'."
                                        "Pass in parameters to generate data or a specific path to the data.")
        df = generate_data(self.config)
        df.to_csv(f'data/{self.config["FLUID"]}_nist_data.csv', index=False)
        return df


class FluidDensityPredictor:
    def __init__(self, poly_degree):
        self.poly_degree = poly_degree
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=self.poly_degree)),
            ('linear', LinearRegression(fit_intercept=False))])

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)
    

def train(loader, config):
    os.makedirs(config['output_dir']) if not os.path.exists(
        config['output_dir']) else None

    poly_range = np.arange(1, config['max_poly_degree'] + 1)
    mae_errors_sum_dict = {}
    df_errors_dict = {}
    df_coefficients_dict = {}

    for poly_degree in poly_range:
        predictor = FluidDensityPredictor(poly_degree)
        df_errors = pd.DataFrame(columns=[
            'mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error', 'max_error'])
        predictor.fit(loader.X_train, loader.Y_train)
        df_coefficients = pd.DataFrame(
            columns=predictor.model.named_steps['poly'].get_feature_names())
        for split in tqdm(range(config['num_splits']), desc=f"Training model with polynomial degree {poly_degree}", unit='split'):
            X_learn, X_val, Y_learn, Y_val = train_test_split(
                loader.X_train, loader.Y_train, shuffle=True, test_size=np.random.uniform(0.2, 0.5)
            )

            predictor.fit(X_learn, Y_learn)
            predictions = predictor.predict(X_val)

            mse_error = mean_squared_error(Y_val, predictions)
            mae_error = mean_absolute_error(Y_val, predictions)
            mape_error = mean_absolute_percentage_error(Y_val, predictions)
            max_error = max(predictions - Y_val)

            df_errors.loc[split] = [mse_error, mae_error, mape_error, max_error]
            df_coefficients.loc[split] = predictor.model.named_steps['linear'].coef_

        df_errors_dict[poly_degree] = df_errors
        df_coefficients_dict[poly_degree] = df_coefficients
        mae_errors_sum_dict[poly_degree] = df_errors['mean_squared_error'].sum()
    best_poly_degree = min(mae_errors_sum_dict, key=mae_errors_sum_dict.get)
    print(f"Best model with polynomial degree: {best_poly_degree}")
    return FluidDensityPredictor(best_poly_degree), df_errors_dict[best_poly_degree], df_coefficients_dict[best_poly_degree]


def plot_results(df_test, df_errors, config):
    x_feature = config['x_feature']
    y_feature = config['y_feature']
    z_feature = config['z_feature']
    z_feature_predicted = f'Predicted {z_feature}'
    error_feature = config['error_feature']

    show_plots = config['show_plots'] if 'show_plots' in config else True
    save_plots = config['save_plots'] if 'save_plots' in config else False
    output_dir = config['output_dir'] if 'output_dir' in config else 'figures'

    # Plot of Pressure vs Temperature colored by absolute error on the density
    fig_2d = px.scatter(df_test, x=x_feature,
                        y=y_feature, color=error_feature)
    fig_2d.update_traces(marker=dict(size=2))
    fig_2d.update_traces(marker=dict(size=4))

    # 3d scatter plot of actual vs predicted density
    fig_3d = go.Figure()
    fig_3d.add_trace(
        go.Scatter3d(
            x=df_test[x_feature],
            y=df_test[y_feature],
            z=df_test[z_feature],
            mode='markers',
            marker=dict(size=2),
            name=f'Actual {z_feature}'
        ))
    fig_3d.add_trace(
        go.Scatter3d(
            x=df_test[x_feature],
            y=df_test[y_feature],
            z=df_test[z_feature_predicted],
            mode='markers',
            marker=dict(size=2),
            name=z_feature_predicted
        ))
    fig_3d.update_layout(
        scene_xaxis_title=x_feature,
        scene_yaxis_title=y_feature,
        scene_zaxis_title=z_feature,
    )

    # Histograms of errors over the splits
    fig_mse_hist = px.histogram(df_errors, x='mean_squared_error')
    fig_mae_hist = px.histogram(df_errors, x='mean_absolute_error')
    fig_mape_hist = px.histogram(df_errors, x='mean_absolute_percentage_error')
    fig_max_hist = px.histogram(df_errors, x='max_error')

    # Save images of the plots to the output directory
    if save_plots:
        os.makedirs(output_dir) if not os.path.exists(output_dir) else None

        fig_2d.write_image(f"{output_dir}/{config['FLUID']}_density_error_plot.pdf")
        fig_mse_hist.write_image(
            f"{output_dir}/{config['FLUID']}_density_mse_histogram.pdf")
        fig_mae_hist.write_image(
            f"{output_dir}/{config['FLUID']}_density_mae_histogram.pdf")
        fig_mape_hist.write_image(
            f"{output_dir}/{config['FLUID']}_density_mape_histogram.pdf")
        fig_max_hist.write_image(
            f"{output_dir}/{config['FLUID']}_density_max_histogram.pdf")

    # Show the plots
    if show_plots:
        fig_2d.show()
        fig_3d.show()
        fig_mse_hist.show()
        fig_mae_hist.show()
        fig_mape_hist.show()
        fig_max_hist.show()
