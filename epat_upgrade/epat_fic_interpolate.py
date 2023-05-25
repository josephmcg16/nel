"""Script to replicate the FIC SCL split code in the EPAT Siemens PLC."""

import numpy as np
import pandas as pd
from dataclasses import dataclass

import plotly.express as px


@dataclass
class Control_Ranges_Lookup:
    Cooling_Output: np.ndarray  # FIC control output
    E_CV081: np.ndarray  # glycol recirculation 2" control valve
    E_CV082: np.ndarray  # glycol return control 1" valve
    E_CV083: np.ndarray  # glycol return control 2" valve
    E_VSD005: np.ndarray  # glycold return pump speed

def interpolate(
        Control_Output: float,
        Control_Ranges_Lookup: Control_Ranges_Lookup
        ) -> np.ndarray:
    
    N_OUTPUT: int = 4
    N_TABLE: int = 8

    Control_Split_Output = np.zeros(N_OUTPUT)

    X_Lookup = Control_Ranges_Lookup.Cooling_Output
    Y_Lookup = np.array([
        Control_Ranges_Lookup.E_CV081,
        Control_Ranges_Lookup.E_CV082,
        Control_Ranges_Lookup.E_CV083,
        Control_Ranges_Lookup.E_VSD005
    ])

    if Control_Output >= X_Lookup[N_TABLE - 1]:
        # Control_Output is greater than the last value in the table
        for i in range(N_OUTPUT):
            Control_Split_Output[i] = Y_Lookup[i, N_TABLE - 1]
    elif Control_Output <= X_Lookup[0]:
        # Control_Output is less than the first value in the table
        for i in range(N_OUTPUT):
            Control_Split_Output[i] = Y_Lookup[i, 0]
    else:
        # Control_Output is between the first and last values in the table
        for i in range(N_OUTPUT):
            for j in range(N_TABLE):
                if Control_Output >= X_Lookup[j] and Control_Output <= X_Lookup[j+1]:
                    # Control_Output is between the jth and j+1th values in the table
                    Control_Split_Output[i] = Y_Lookup[i, j] + (Y_Lookup[i, j+1] - Y_Lookup[i, j]) * (Control_Output - X_Lookup[j]) / (X_Lookup[j+1] - X_Lookup[j])
                    break
    return Control_Split_Output

def main():
    Control_Output_Range = np.linspace(0, 100, 10_001)

    Control_Ranges_Lookup = Control_Ranges_Lookup(
        Cooling_Output=np.array([0.1, 0.4, 2, 4, 20, 40, 96, 100]),  # %output
        E_CV081=np.array([100, 75, 72, 70, 69, 68, 10, 0]),  # %travel
        E_CV082=np.array([0, 26, 53, 70, 100, 100, 100, 100]),  # %travel
        E_CV083=np.array([0, 0, 0, 0, 34, 50, 81, 84]),  # %travel
        E_VSD005=np.array([0, 80, 80, 80, 80, 80, 80, 100])  # %speed
    )

    # interpolate over range of control outputs
    Control_Split_Output_Range = []
    for Control_Output in Control_Output_Range:
        Control_Split_Output_Range.append(
            interpolate(Control_Output, Control_Ranges_Lookup))
    Control_Split_Output_Range = np.array(Control_Split_Output_Range)

    # save outputs over the range to csv
    df = pd.DataFrame(
        np.vstack((Control_Output_Range, Control_Split_Output_Range.T)).T,
        columns=['Control_Output', 'E_CV081', 'E_CV082', 'E_CV083', 'E_VSD005'])
    df.to_csv('data/Control_Split_Output_Range.csv', index=False)

    # Plot
    fig = px.line(df, x='Control_Output', y=[
                  'E_CV081', 'E_CV082', 'E_CV083', 'E_VSD005'])
    fig.update_layout(
        title='FIC Control Split Output',
        xaxis_title='Control Output (%)',
        yaxis_title='Control Split Output (%)',
        legend_title='Control Split Output',
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="RebeccaPurple"
        )
    )
    fig.show()


if __name__ == '__main__':
    main()