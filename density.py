"""Playground codes for densitometer coeffecients optimisation."""
import pandas as pd
import numpy as np

from scipy.optimize import minimize, shgo
from sklearn.metrics import mean_squared_error


class Densitometer:
    """_summary_"""
    def __init__(
        self, data: pd.core.frame.DataFrame, 
        K0: float, K1: float, K2: float, K18: float, K19: float, 
        K20A: float, K20B: float, K21A: float, K21B :float
        ) -> 'Densitometer':
        """Instantiate class: `Densitometer`"""
        self.data = data
        self.K0 = K0
        self.K1 = K1
        self.K2 = K2
        self.K18 = K18
        self.K19 = K19
        self.K20A = K20A
        self.K20B = K20B
        self.K21A = K21A
        self.K21B = K21B
        self.coeffecients = np.array([
            K0, K1, K2, K18, K19, K20A, K20B, K21A, K21B
        ])
        self.rho_meter = self._calc_density(self.coeffecients)
        self.rho_nel = self.data['rnel'].to_numpy()

    def _calc_density(self, K_arr: np.array) -> np.ndarray:
        """Calculate densitometer density measurement from calibration coeffecients,
        temperature and pressure readings"""
        tau = self.data['tden']     # oscillation period [micro sec]
        t = self.data['tfluid']     # temperature [degC]
        p = self.data['pfluid']     # pressure [bara]

        T_REF = 20  # reference temperature [degC]
        P_REF = 1  # reference pressure [bara]

        # density at reference temperature [kg/ m3]
        rho_0 = K_arr[0] + K_arr[1] * tau + K_arr[2] * tau ** 2

        # density corrected for temperature [kg/ m3]
        rho_t = rho_0 * (1 + K_arr[3] * (t - T_REF)) + K_arr[4] * (t-T_REF)

        # density corrected for temperature and pressure [kg/ m3]
        rho_tp = rho_t * (1 + K_arr[5] * (p - P_REF) + K_arr[6] * (p - P_REF) ** 2) + \
            (K_arr[7] * (p - P_REF) + K_arr[8] * (p - P_REF) ** 2)

        return rho_tp.to_numpy()

    def optimize_error(self, method="shgo", **kwargs):
        def obj_func(K_arr):
            return mean_squared_error(
                self.rho_nel, self._calc_density(K_arr)
            )
        if method == "minimize":
            res = minimize(
                obj_func, self.coeffecients, **kwargs
            )
        elif method == "shgo":
            res = shgo(
                obj_func, [(0, 1)] * len(self.coeffecients), **kwargs
            )
        else:
            raise NotImplementedError
        self.coeffecients_opt = res.x
        self.rho_meter_opt = self._calc_density(res.x)
        return res
