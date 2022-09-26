"""Based on BS ISO 18466:2016 Section 8 balance method and data reconciliation.
TODO: Inputs and outputs of each balance equation."""
import numpy as np


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
        int: Desired output of the mass balance.
    """
    return 1


# Section 8.3
def ash_balance_input(mass_fractions):
    """Ash balance formula input.

    Args:
        mass_fractions (np.ndarray): Array elements with mass fraction of each material group,
        [Inert, Biogenic, Fossil and Water].

    Returns:
        float: Mass fraction of the inert (inorganic) material
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
        the Waste for Energy (WfE) plant
    """
    return np.sum(m_reisdues) / m_tot


# def carbon_balance(mass_fractions, carbon_contents, co2_concentrations, o2_concetrations)


if __name__ == '__main__':
    W_INERT = 0.1
    W_BIOGENIC = 0.3
    W_INERT = 0.4
    W_H20 = 0.5
