import pandas as pd
import numpy as np
from scipy import optimize
from scipy.spatial import KDTree
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from tqdm import tqdm


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


def scrape_nist_satcurve(fluid, pressure_range):
    pressure_inc = (pressure_range[-1] - pressure_range[0]) / (len(pressure_range) - 1)
    df = pd.read_html(f"https://webbook.nist.gov/cgi/fluid.cgi?PLow={pressure_range[0]}&PHigh={pressure_range[-1]}&PInc={pressure_inc}&Digits=5&ID={NIST_ID_DICT[fluid]}&Action=Load&Type=SatT&TUnit=C&PUnit=bar&DUnit=kg%2Fm3&HUnit=kJ%2Fkg&WUnit=m%2Fs&VisUnit=cP&STUnit=N%2Fm&RefState=DEF#Vapor")[0]
    return df


def filter_near_sat_curve(df, df_sat, n_samples):
    # df indices now range from 0 to number of rows in df
    df = df.reset_index(drop=True)

    # create a KDTree from the saturation curve
    sat_tree = KDTree(df_sat[['Pressure (bar)', 'Temperature (C)']])

    # query the KDTree for the nearest neighbors of each sample in pT_data
    _, closest_indices = sat_tree.query(
        df[['Pressure (bar)', 'Temperature (C)']], k=1)

    # find the distances between the samples and their nearest neighbors in the curve
    distances = np.linalg.norm(df[['Pressure (bar)', 'Temperature (C)']].to_numpy(
    ) - df_sat.iloc[closest_indices][['Pressure (bar)', 'Temperature (C)']].to_numpy(), axis=1)

    # sort the distances in ascending order and select the n_samples closest samples based on their distances
    closest_samples = np.argsort(distances)[:n_samples]
    return df.drop(df.index[closest_samples], inplace=False)


def scrape_nist_data(fluid, temperature_range, pressure_range, sat_curve_filter=0, vapor_phase=True, callback=False):
    df = pd.DataFrame()

    if callback:
        for isotherm in tqdm(temperature_range):
            df = df.append(scrape_nist_isotherm(fluid, isotherm, pressure_range))

    else:
        for isotherm in temperature_range:
            df = df.append(scrape_nist_isotherm(fluid, isotherm, pressure_range))

    # normalized pressure and temperature
    df['pi'] = df['Pressure (bar)'] / 50
    df['tau'] = 300 / (df['Temperature (C)'] + 273.15)

    if sat_curve_filter > 0:
        df_sat = scrape_nist_satcurve(fluid, pressure_range)
        filter_near_sat_curve(df, df_sat, sat_curve_filter)

    if vapor_phase:
        df = df[df['Phase'] == "vapor"]


    # df indices now range from 0 to number of rows
    df = df.reset_index(drop=True)
    return df
