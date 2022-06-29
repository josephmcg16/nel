"""Playground codes for densitometer coeffecients optimisation."""
import pandas as pd
import numpy as np

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