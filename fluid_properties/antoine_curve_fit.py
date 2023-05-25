import numpy as np
from scipy.optimize import curve_fit
import plotly.express as px
from utils import scrape_nist_satcurve


class antoineCurveFit:
    def __init__(self, fluid, press_min, press_max, press_increment, root_path='.'):
        self.fluid = fluid
        self.press_min = press_min
        self.press_max = press_max
        self.press_increment = press_increment
        self.pressure_range = np.arange(
            press_min, press_max + press_increment, press_increment)
        self.df_sat = scrape_nist_satcurve(self.fluid, self.pressure_range)
        self.temperature = self.df_sat['Temperature (C)'].to_numpy() + 273.15
        self.pressure = self.df_sat['Pressure (bar)'].to_numpy()
        self.root_path = root_path
        self.a_lsq = None

    def antoine(self, T, a1, a2, a3):
        return 10**(a1 - a2 / (T + a3))

    def fit_data(self, console_output=True):
        self.a_lsq, _ = curve_fit(self.antoine, self.temperature,
                             self.pressure, maxfev=100_000)
        pressure_calc = self.antoine(self.temperature, *self.a_lsq)

        self.df_sat['Pressure Calc (bar)'] = pressure_calc
        self.df_sat['Relative Error (%)'] = (self.df_sat['Pressure Calc (bar)'] -
                                             self.df_sat['Pressure (bar)']) / self.df_sat['Pressure (bar)'] * 100
        if console_output:
            self.console_output(self.a_lsq, self.df_sat)
        return self.a_lsq, self.df_sat

    def console_output(self, a_lsq, df):
        for i, a in enumerate(a_lsq):
            print(f"a{i+1} = {a:.3e}")
        print(f"\nMeanRelErr = {df['Relative Error (%)'].abs().mean():.3e} %")
        print(f"MaxRelErr = {df['Relative Error (%)'].abs().max():.3e} %")
        print(f"MinRelErr = {df['Relative Error (%)'].abs().min():.3e} %\n")
        return

    def plot_data(self):
        fig = px.scatter(self.df_sat, x='Temperature (C)', y='Pressure (bar)',
                         color='Relative Error (%)', title=f'{self.fluid} Saturation Curve')
        fig.update_layout(
            xaxis_title='Temperature (C)',
            yaxis_title='Pressure (bar)',
            legend_title='Relative Error (%)',
            font=dict(
                family="Courier New, monospace",
                size=18,
                color="#7f7f7f"
            )
        )
        return fig


if __name__ == "__main__":
    FLUID = "CO2"
    PRESS_MIN = 39.69  # bar
    PRESS_MAX = 46  # bar
    PRESS_INCREMENT = 0.01  # bar

    antoine_curve = antoineCurveFit(FLUID, PRESS_MIN, PRESS_MAX, PRESS_INCREMENT)
    coeffecients, df_sat = antoine_curve.fit_data()

    fig = antoine_curve.plot_data()
    fig.show()

    df_sat.to_csv(f'data/{FLUID}_sat_curve.csv', index=False)
    np.savetxt(f'data/{FLUID}_antoine_coeffecients.txt', coeffecients)
