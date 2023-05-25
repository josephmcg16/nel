import pandas as pd
import numpy as np


def df_flowrate_range(df, flowrate_min, flowrate_max):
    df_reduced = df[(df['E-FT008 Flowrate, l/s'] >= flowrate_min)
                    & (df['E-FT008 Flowrate, l/s'] <= flowrate_max)]
    return df_reduced


if __name__ == "__main__":
    df = pd.read_excel("data/T002502 FRIP23 2023-05-22 171025.xlsx")
    df["E-FT008 Flowrate ROC, l/s2"] = np.gradient(
        df["E-FT008 Flowrate, l/s"], df["E-FT008 Flowrate Timestamp, sec"])


    COOLING_OUTPUTS = [
        0, 0.4, 2, 4, 20, 40, 96, 100
    ]  # %

    MAXIMUM_DESIGN_FLOWRATE = 3.8  # l/s
    MINIMUM_DESIGN_FLOWRATE = 0.1  # l/s

    for i, cooling_output in enumerate(COOLING_OUTPUTS[:-1]):

        MIN_PERCENT = cooling_output
        MAX_PERCENT = COOLING_OUTPUTS[i + 1]

        print(f"MIN_PERCENT: {MIN_PERCENT} %")
        print(f"MAX_PERCENT: {MAX_PERCENT} %\n")

        MAXIMUM_FLOWRATE = MAX_PERCENT / 100 * \
            (MAXIMUM_DESIGN_FLOWRATE - MINIMUM_DESIGN_FLOWRATE) + MINIMUM_DESIGN_FLOWRATE
        MINIMUM_FLOWRATE = MIN_PERCENT / 100 * \
            (MAXIMUM_DESIGN_FLOWRATE - MINIMUM_DESIGN_FLOWRATE) + MINIMUM_DESIGN_FLOWRATE

        df_reduced = df_flowrate_range(df, MINIMUM_FLOWRATE, MAXIMUM_FLOWRATE)

        df_corr = df_reduced[
            [
                "E-FT008 Flowrate ROC, l/s2",
                "E-CV081 Travel, %",
                "E-CV082 Travel, %",
                "E-CV083 Travel, %",
                "E-VSD005 Speed, %"
            ]
        ].corr()
        df_out = df_corr.style.background_gradient(
            cmap='coolwarm')
        df_out.to_excel(f"data/corr_{MIN_PERCENT}%-{MAX_PERCENT}%.xlsx")
