"""Module based on BS ISO 18466:2016 Section 8 balance method and data reconciliation.
TODO: Inputs and outputs of each balance equation."""
import numpy as np
from utils import carbon_fraction, oxygen_fraction


# Section 8.2
def mass_balance_input(mass_fractions):
    """Mass balance formula input.

    Args:
        mass_fractions (np.ndarray): Array elements with mass fraction of each material group,
        [Biogenic, Fossil, Inert, Water].

    Returns:
        int: Total mass balance.
    """
    return np.sum(mass_fractions)


def mass_balance_output():
    """Mass balance formula output.

    Returns:
        int: Desired output of the mass balance (i.e., unity).
    """
    return 1


# Section 8.3
def ash_balance_input(mass_fractions):
    """Ash balance formula input.

    Args:
        mass_fractions (np.ndarray): Array elements with mass fraction of each material group,
        [Biogenic, Fossil, Inert, Water].

    Returns:
        float: Mass fraction of the inert (inorganic) material.
    """
    w_inert = mass_fractions[2]
    return w_inert


def ash_balance_output(m_reisdues, m_tot):
    """Ash balance formula output.

    Args:
        m_reisdues (_type_): _description_
        m_tot (_type_): _description_

    Returns:
        float: Quotient of the measured mass flow of solid residues ΣWs and the waste input mtot of
        the Waste for Energy (WfE) plant.
    """
    return np.sum(m_reisdues) / m_tot


def carbon_balance_input(
    mass_fractions, carbon_contents):
    """Carbon balance formula input.

    Args:
        mass_fractions (np.ndarray): Array elements with mass fraction of each material group,
        [Biogenic, Fossil, Inert, Water].
        carbon_contents (np.ndarray): Array elements with carbon contents of the organic mass 
        fractions,
        [Biogenic, Fossil, Gas, Oil]

    Returns:
        float: Product of organic mass fractions and their carbon contents.
    """
    w_biogenic = mass_fractions[0]
    w_fossil = mass_fractions[1]

    c_biogenic = carbon_contents[0]
    c_fossil = carbon_contents[1]

    return w_biogenic * c_biogenic + w_fossil * c_fossil


def carbon_balance_output(
    concentrations, carbon_contents, m_tot, m_carbon, m_gas, m_oil, vol_flue, vol_tot, vol_gas):
    """Carbon balance formula output.

    Args:
        concentrations (np.ndarray): Array elements with concentrations of gases in air and flue
        gas,
        [CO2 in flue, CO2 in air, O2 in flue, O2 in air]]

    Returns:
        float: The average content of organic carbon of the waste feed derived from the operating 
        data of the plant subtracting the contribution of auxiliary fuel.
    """
    return (vol_flue * carbon_fraction(concentrations, vol_tot) * m_carbon / m_tot)


def energy_balance_input() -> None:
    return 1


if __name__ == '__main__':
    W_INERT = 0.1
    W_BIOGENIC = 0.3
    W_INERT = 0.4
    W_H20 = 0.5
