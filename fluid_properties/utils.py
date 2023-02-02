import pandas as pd
import numpy as np
from scipy import optimize
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

GAS_CONSTANT_DICT = {
    "CO2": 188.9,
    "N2": 296.8,
    "CH4": 518.2
}  # J/kg.K

NIST_ID_DICT = {
    "CO2": "C124389",
    "N2": "C7727379",
    "CH4": "C74828"
}  # NIST webbook IDs


def scrape_nist_isotherm(fluid, isotherm, pressure_range):
    pressure_inc = (pressure_range[-1] - pressure_range[0]) / (len(pressure_range) - 1)
    path = f"https://webbook.nist.gov/cgi/fluid.cgi?Action=Load&ID={NIST_ID_DICT[fluid]}&Type=IsoTherm&Digits=5&PLow={pressure_range[0]}&PHigh={pressure_range[-1]}&PInc={pressure_inc}&T={isotherm}&RefState=DEF&TUnit=C&PUnit=bar&DUnit=kg%2Fm3&HUnit=kJ%2Fkg&WUnit=m%2Fs&VisUnit=cP&STUnit=N%2Fm"
    df = pd.read_html(path)[0]

    df['Compressibility Factor'] = 10 ** 5 * df['Pressure (bar)'] / (
        df['Density (kg/m3)'] * (df['Temperature (C)'] + 273.15) * GAS_CONSTANT_DICT[fluid])
    return df


def scrape_nist_data(fluid, temperature_range, pressure_range, vapor_phase=True):
    df = pd.DataFrame()
    for isotherm in temperature_range:
        df = df.append(scrape_nist_isotherm(fluid, isotherm, pressure_range))

    # normalized pressure and temperature
    df['pi'] = df['Pressure (bar)'] / 50
    df['tau'] = 300 / (df['Temperature (C)'] + 273.15)

    if vapor_phase:
        return df[df['Phase'] == "vapor"]
    else:
        return df
