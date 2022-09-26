"""Module wrapping various helper functions."""
import numpy as np


# biogenic CO2
def carbon_fraction(concentrations, vol_tot):
    co2_flue = concetrations[0]
    co2_air = concetrations[1]

    o2_flue = concetrations[2]
    o2_air = concetrations[3]
    return (co2_flue - co2_air * (1 - o2_flue - co2_flue) / (1 - o2_air - co2_air)) / vol_tot


def oxygen_fraction(concentrations, vol_tot):
    return