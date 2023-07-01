import pandas as pd
import numpy as np
from tqdm import tqdm
import yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import plotly.express as px
import plotly.graph_objects as go
from fluid_properties.regression import DataLoader, FluidDensityPredictor


with open('config.yaml') as config_file:
    CONFIG = yaml.load(config_file, Loader=yaml.FullLoader)
loader = DataLoader(CONFIG)
df = loader.load_data()

X = df[['Temperature (C)', 'Pressure (bar)']].to_numpy()
Y = df['Density (kg/m3)'].to_numpy()

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=CONFIG["test_size"], random_state=CONFIG['random_state'], shuffle=True)

predictor = FluidDensityPredictor(CONFIG)
predictor.fit(X, Y)

df_errors = pd.DataFrame(columns=['mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error'])
df_coefficients = pd.DataFrame(columns=predictor.model.named_steps['poly'].get_feature_names())
for split in tqdm(range(CONFIG['num_splits'])):
    X_learn, X_val, Y_learn, Y_val = train_test_split(
        X_train, Y_train, shuffle=True, test_size=np.random.uniform(0.2, 0.5)
    )

    predictor.fit(X_learn, Y_learn)

    predictions = predictor.predict(X_val)

    mse_error = mean_squared_error(Y_val, predictions)
    mae_error = mean_absolute_error(Y_val, predictions)
    mape_error = mean_absolute_percentage_error(Y_val, predictions)

    df_errors.loc[split] = [mse_error, mae_error, mape_error]
    df_coefficients.loc[split] = predictor.model.named_steps['linear'].coef_

predictor.fit(X_train, Y_train)
df_test = pd.DataFrame()
df_test['Temperature (C)'], df_test['Pressure (bar)'] = X_test.T
df_test['Density (kg/m3)'] = Y_test
df_test['Predicted Density (kg/m3)'] = predictor.predict(X_test)
df_test['Absolute Error, Density'] = df_test['Predicted Density (kg/m3)'] - df_test['Density (kg/m3)']
df_test['Squared Error, Density'] = (df_test['Predicted Density (kg/m3)'] - df_test['Density (kg/m3)']) ** 2

df_coefficients.mean().to_csv('data/CO2_coefficients.csv', index_label='Feature Name', header=['Coeffecient'])
df_errors.to_csv('data/CO2_regression_errors.csv', index_label='Split Number')

# Plot of Pressure vs Temperature colored by absolute error on the density
fig_2d = px.scatter(df_test, x='Temperature (C)',
                 y='Pressure (bar)', color='Absolute Error, Density')
fig_2d.update_traces(marker=dict(size=2))
fig_2d.write_image('figures/CO2_desnity_error_plot.pdf')
fig_2d.update_traces(marker=dict(size=4))
fig_2d.show()

# 3d scatter plot of actual vs predicted density
fig_3d = go.Figure()
fig_3d.add_trace(
    go.Scatter3d(
        x=df_test['Temperature (C)'],
        y=df_test['Pressure (bar)'],
        z=df_test['Density (kg/m3)'],
        mode='markers',
        marker=dict(size=2),
        name='Actual Density'
    ))
fig_3d.add_trace(
    go.Scatter3d(
        x=df_test['Temperature (C)'],
        y=df_test['Pressure (bar)'],
        z=df_test['Predicted Density (kg/m3)'],
        mode='markers',
        marker=dict(size=2),
        name='Predicted Density'
    ))
fig_3d.update_layout(
    scene_xaxis_title="Pressure (bar)",
    scene_yaxis_title="Temperature (C)",
    scene_zaxis_title="Density (kg/m3)",
)
fig_3d.show()

# Histograms of errors over the splits
fig_mse_hist = px.histogram(df_errors, x='mean_squared_error')
fig_mse_hist.write_image('figures/CO2_density_mse_histogram.pdf')
fig_mse_hist.show()
fig_mae_hist = px.histogram(df_errors, x='mean_absolute_error')
fig_mae_hist.write_image('figures/CO2_density_mae_histogram.pdf')
fig_mae_hist.show()
fig_mape_hist = px.histogram(df_errors, x='mean_absolute_percentage_error')
fig_mape_hist.write_image('figures/CO2_density_mape_histogram.pdf')
fig_mape_hist.show()
