"""PVTt Calculation Module
This module is designed for performing calculations related to a primary gas flow measurement standard, 
used for calibrations in a PVTt (Pressure-Volume-Temperature-time) system.

Mass Flow Calculation:
    The mass flow rate is calculated using the following equation:
    \dot{m} = (m_T^f - m_T^i) - (m_I^f - m_I^i) / (t_f - t_i)
    Where:
        \dot{m}: Mass flow rate.
        m_T^f: Final mass of gas in the tank.
        m_T^i: Initial mass of gas in the tank.
        m_I^f: Final mass of gas in the inventory volume.
        m_I^i: Initial mass of gas in the inventory volume.
        t_f: Time at the end of the test.
        t_i: Time at the start of the test.
    The mass flow rate can also be calculated from volume and density values.

Tank Volume Determination:
    The internal tank volume is determined by the gas gravimetric method, used by NIST for their 677 L PVTt primary standard.
    V_grav = (m_C^i - m_C^f) / (ρ_T^f - ρ_T^i) - V_extra
    Where:
        V_grav: Internal volume of the tank determined via the gravimetric method.
        m_C^i: Initial mass of the high-pressure cylinder.
        m_C^f: Final mass of the high-pressure cylinder.
        ρ_T^f: Final tank density.
        ρ_T^i: Initial tank density.
        V_extra: Extra volume temporarily connected to the tank.

Cylinder Mass Determination:
    The mass of the high-pressure cylinder is determined using a substitution process with reference masses and a mass comparator.
    m_c = S_c / S_ref * m_ref * (1 - ρ_air / ρ_ref) + ρ_air * V_ext
    Where:
        m_c: Mass of the high-pressure cylinder.
        S_c: Scale reading of the high-pressure cylinder.
        S_ref: Scale reading for the reference mass.
        m_ref: Reference mass.
        ρ_ref: Density of the reference mass.
        ρ_air: Density of ambient air during the measurements.
        V_ext: External volume of the cylinder including its valves and fittings.

External Volume of Cylinder:
    The external volume of the cylinder is measured by Archimedes principle, using the change in apparent mass in two media.
    V_ext(T_ref) = (m_air^A - m_water^A) / (ρ_water[1 + 3α(T_water - T_ref)] - ρ_air[1 + 3α(T_air - T_ref)])
    Where:
        V_ext(T_ref): External volume of the cylinder at the reference temperature.
        m_air^A: Apparent mass of the cylinder in air.
        m_water^A: Apparent mass of the cylinder in water.
        ρ_water: Density of water.
        ρ_air: Density of air.
        T_water: Temperature of the water.
        T_air: Temperature of the air.
        T_ref: Reference temperature.
        α: Linear thermal expansion coefficient of the cylinder material.

Inventory Volume Determination:
    The inventory volume is determined by the volume expansion method, using the conservation of mass in a pressurized system.
    V_I = (ρ_T^f - ρ_T^i) V_T / (ρ_I^i - ρ_I^f) - V_extra
    Where:
        V_I: Inventory volume.
        V_T: Tank volume.
        ρ_I^i: Initial density in the inventory volume.
        ρ_I^f: Final density in the inventory volume.
        ρ_T^i: Initial density in the tank.
        ρ_T^f: Final density in the tank.
        V_extra: Extra volume attached to the tank.
"""
from dataclasses  import dataclass
import numpy as np


@dataclass
class PVTtCalculator:
    """
    A calculator for determining mass flow rate in a simplified PVTt system.

    This class assumes the volumes of the inventory and tank are constants and that the gas completely fills the tanks to a certain pressure.
    these volumes to a certain pressure. It calculates the mass flow rate based on changes in density and known volumes.

    Attributes:
        t0 (float): Start time of the test.
        tf (float): End time of the test.
        pressure_tank_initial (np.ndarray): Initial pressure of the gas in the tank (Pa).
        pressure_tank_final (np.ndarray): Final pressure of the gas in the tank (Pa).
        pressure_inventory_initial (np.ndarray): Initial pressure of the gas in the inventory volume (Pa).
        pressure_inventory_final (np.ndarray): Final pressure of the gas in the inventory volume (Pa).
        temperature_tank_initial (np.ndarray): Initial temperature of the gas in the tank (K).
        temperature_tank_final (np.ndarray): Final temperature of the gas in the tank (K).
        temperature_inventory_initial (np.ndarray): Initial temperature of the gas in the inventory volume (K).
        temperature_inventory_final (np.ndarray): Final temperature of the gas in the inventory volume (K).
        density_tank_initial (np.ndarray): Initial density of the gas in the tank (kg/m3).
        density_tank_final (np.ndarray): Final density of the gas in the tank (kg/m3).
        density_inventory_initial (np.ndarray): Initial density of the gas in the inventory volume (kg/m3).
        density_inventory_final (np.ndarray): Final density of the gas in the inventory volume (kg/m3).
        volume_tank (np.ndarray): Volume of the tank (m3).
        volume_extra (float): Extra volume temporarily connected to the tank (m3).
        volume_inventory (np.ndarray): Calculated volume of the inventory (m3).
        mass_flowrate (np.ndarray): Calculated mass flow rate (kg/s).
    """

    t0: float  # s
    tf: float  # s
    pressure_tank_initial: np.ndarray  # Pa
    pressure_tank_final: np.ndarray  # Pa
    pressure_inventory_initial: np.ndarray  # Pa
    pressure_inventory_final: np.ndarray  # Pa
    temperature_tank_initial: np.ndarray  # K
    temperature_tank_final: np.ndarray  # K
    temperature_inventory_initial: np.ndarray  # K
    temperature_inventory_final: np.ndarray  # K
    volume_tank: np.ndarray # m3
    volume_extra: float = 0  # m3
    gas_constant: float = 296.8  # J/kg/K, placeholder value for N2

    def __post_init__(self) -> None:
        """
        Additional initializations that depend on the fields defined in the dataclass.
        """
        self.density_tank_initial = self._calculate_density(
            self.pressure_tank_initial, self.temperature_tank_initial)  # kg/m3
        self.density_tank_final = self._calculate_density(
            self.pressure_tank_final, self.temperature_tank_final)  # kg/m3
        self.density_inventory_initial = self._calculate_density(
            self.pressure_inventory_initial, self.temperature_inventory_initial)
        self.density_inventory_final = self._calculate_density(
            self.pressure_inventory_final, self.temperature_inventory_final)

        self.mass_tank_initial = self.volume_tank * self.density_tank_initial  # kg
        self.mass_tank_final = self.volume_tank * self.density_tank_final

        self.volume_inventory = self._calculate_volume_inventory()  # m3
        self.mass_inventory_initial = self.volume_inventory * self.density_inventory_initial  # kg
        self.mass_inventory_final = self.volume_inventory * self.density_inventory_final  # kg

        self.mass_flowrate = self._calculate_mass_flowrate()  # kg/s

    def _calculate_mass_flowrate(self) -> np.ndarray:
        """
        Calculates the mass flow rate based on the changes in mass in the tank and inventory.

        Returns:
            np.ndarray: Mass flow rate (kg/s).
        """
        delta_mass_tank = self.mass_tank_final - self.mass_tank_initial
        delta_mass_inventory = self.mass_inventory_final - self.mass_inventory_initial
        
        return (delta_mass_tank - delta_mass_inventory) / (self.tf - self.t0)

    def _calculate_volume_inventory(self) -> np.ndarray:
        """
        Calculates the inventory volume using the volume expansion method, initially assumed for the uncertainty budget.
        This method involves pressurizing a known volume (the large primary standard tank), evacuating the unknown volume 
        (the inventory), and then opening a valve between the two volumes. The change in density within these volumes 
        is then used to calculate the unknown inventory volume. This approach is based on the conservation of mass principle 
        applied to the system.

        Returns:
            np.ndarray: Volume of the inventory (m3).
        """
        delta_density_tank = self.density_tank_final - self.density_tank_initial
        delta_density_inventory = self.density_inventory_initial - self.density_inventory_final
        return self.volume_tank * delta_density_tank / delta_density_inventory - self.volume_extra

    def _calculate_density(self, pressure, temperature):
        """
        Calculates density of the gas based on the known pressure and temperature.

        Parameters:
            pressure (float): Pressure of the gas.
            temperature (float): Temperature of the gas.

        Returns:
            float: Calculated density of the gas.
        """
        # Placeholder calculation using the ideal gas law
        return pressure / (self.gas_constant * temperature)


if __name__ == "__main__":
    import matplotlib.pyplot as plt


    # simulating sampled data over n scans
    num_of_scans = 600

    t0 = 0  # s
    tf = 3600  # s

    pressure_tank_initial = np.random.normal(5e3, 1e3, size=num_of_scans)  # Pa
    pressure_tank_final = np.random.normal(1.5e7, 1e3, size=num_of_scans)  # Pa
    pressure_inventory_initial = np.random.normal(1.5e7, 1e3, size=num_of_scans)  # Pa
    pressure_inventory_final = np.random.normal(1.2e5, 1e3, size=num_of_scans)  # Pa

    temperature_tank_initial = np.random.normal(296.8, 0.1, size=num_of_scans)  # K
    temperature_tank_final = np.random.normal(296.8, 0.1, size=num_of_scans)  # K
    temperature_inventory_initial = np.random.normal(296.8, 0.1, size=num_of_scans)  # K
    temperature_inventory_final = np.random.normal(296.8, 0.1, size=num_of_scans)  # K
    
    volume_tank = np.random.normal(0.23, 0.01, size=num_of_scans)  # m3


    # perform consultant calcs
    pg_calc = PVTtCalculator(
            t0,
            tf,
            pressure_tank_initial,
            pressure_tank_final,
            pressure_inventory_initial,
            pressure_inventory_final,
            temperature_tank_initial,
            temperature_tank_final,
            temperature_inventory_initial,
            temperature_inventory_final,
            volume_tank
            )
    
    # print results
    print(f"Mass Flowrate (kg/s) : {pg_calc.mass_flowrate.mean():.3e} +- {1.96 * pg_calc.mass_flowrate.std() / np.sqrt(num_of_scans):.3e} (95% CI))")
    # print("Inventory Volume (m3) : ", pg_calc.volume_inventory.mean())
    # print("Tank Volume (m3) : ", pg_calc.volume_tank.mean())
    # print(f"Initial Inventory Mass (kg) : {pg_calc.mass_inventory_initial.mean()}")
    # print(f"Final Inventory Mass (kg) : {pg_calc.mass_inventory_final.mean()}")
    # print(f"Initial Tank Mass (kg) : {pg_calc.mass_tank_initial.mean()}")
    # print(f"Final Tank Mass (kg) : {pg_calc.mass_tank_final.mean()}")

    # plot distribution of mass flowrates over the test point
    plt.hist(pg_calc.mass_flowrate, bins=int(num_of_scans/10))
    plt.title(f"Mass Flowrate Distribution ($n_{{scans}}={num_of_scans}$)")
    plt.xlabel("Mass Flowrate (kg/s)")
    plt.ylabel("Frequency")
    plt.show()
