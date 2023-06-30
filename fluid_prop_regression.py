import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

import plotly.express as px
import plotly.graph_objects as go

from fluid_properties.regression import FluidDensityPredictor, DataLoader, calculate_error


config = {
    "path": "fluid_properties/data/CO2_nist_data.csv"
}

loader = DataLoader(config)
df = loader.load_data()

X = df[['Temperature (C)', 'Pressure (bar)']].to_numpy()
Y = df['Density (kg/m3)'].to_numpy()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = FluidDensityPredictor()
model.fit(X_train, Y_train)

predictions = model.predict(X_test)

df_test = pd.DataFrame()
df_test['Temperature (C)'], df_test['Pressure (bar)'] = X_test.T
df_test['Density (kg/m3)'] = Y_test
df_test['Predicted Density (kg/m3)'] = predictions
df_test['Absolute Error, Density'] = df_test['Predicted Density (kg/m3)'] - df_test['Density (kg/m3)']

fig = px.scatter(df_test, x='Temperature (C)', y='Pressure (bar)', color='Absolute Error, Density')
fig.show()

fig = go.Figure()
fig.add_trace(
    go.Scatter3d(
        x=df_test['Temperature (C)'],
        y=df_test['Pressure (bar)'],
        z=df_test['Density (kg/m3)'],
        mode='markers',
        marker=dict(size=2)
    ))
fig.add_trace(
    go.Scatter3d(
        x=df_test['Temperature (C)'],
        y=df_test['Pressure (bar)'],
        z=df_test['Predicted Density (kg/m3)'],
        mode='markers',
        marker=dict(size=2)
    ))
fig.show()
