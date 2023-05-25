import numpy as np
import plotly.express as px
from utils import scrape_nist_data


def generate_data(fluid, temp_min, temp_max, temp_increment, press_min, press_max, press_increment, vapor_phase=True, callback=True, root_path='.'):
    temperature_range = np.arange(
        temp_min, temp_max + temp_increment, temp_increment)  # degC
    pressure_range = np.arange(
        press_min, press_max + press_increment, press_increment)  # bar

    df = scrape_nist_data(
        fluid,
        temperature_range,
        pressure_range,
        vapor_phase=vapor_phase,
        callback=callback,
    )
    return df


if __name__ == "__main__":
    TEMP_MIN = 5
    TEMP_MAX = 30
    TEMP_INCREMENT = 0.5
    PRESS_MIN = 3
    PRESS_MAX = 46
    PRESS_INCREMENT = 0.1
    FLUID = "CO2"

    df = generate_data(FLUID, TEMP_MIN, TEMP_MAX, TEMP_INCREMENT,
                  PRESS_MIN, PRESS_MAX, PRESS_INCREMENT)
    df.to_csv(f'data/{FLUID}_nist_data.csv', index=False)
