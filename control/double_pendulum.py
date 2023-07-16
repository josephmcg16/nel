import numpy as np
from sdeint import itoint
import pandas as pd
import plotly.express as px
from utils import double_pendulum_animation

def ode(y, t, c, u):
    """Double pendulum ODE
    Args:
        y (n x 1 np.ndarray): [theta1, omega1, theta2, omega2]
        t (float): time
        c (float): damping coefficient
        u (float): control input
    Returns:
        n x 1 np.ndarray: [dtheta1/dt, domega1/dt, dtheta2/dt, domega2/dt]
    """
    theta1, omega1, theta2, omega2 = y

    # equations of motion
    omega1_dot = ((omega1**2*np.sin(theta2-theta1)*np.cos(theta2-theta1)
                   + np.sin(theta2)*np.cos(theta2-theta1)
                   + omega2**2*np.sin(theta2-theta1)
                   - np.sin(theta1))
                  / (2 - np.cos(theta2-theta1)**2) - c*omega1) + u[0]

    omega2_dot = ((-omega2**2*np.sin(theta2-theta1)*np.cos(theta2-theta1)
                   + (np.sin(theta1)*np.cos(theta2-theta1)
                   - omega1**2*np.sin(theta2-theta1)
                   - np.sin(theta2)))
                  / (2 - np.cos(theta2-theta1)**2) - c*omega2) + u[1]
    
    if np.abs(t - 5) < 10 * dt:
        omega1_dot += 500

    return np.array([omega1, omega1_dot, omega2, omega2_dot])


def diffusion(y, t, epsilon):
    """Pendulum diffusion matrix
    Args:
        y (n x 1 np.ndarray): [theta, omega]
        t (float): time
        epsilon (float, optional): maximum diffusion strength. Defaults to 0.1.
    Returns:
        n x n np.ndarray: diffusion matrix
    """
    n = y.shape[0]
    return np.random.uniform(-epsilon, epsilon, size=(n, n))


errors_dict = {0: np.zeros(2)}
control_signal = {0: 0}
def control_input(y, t, setpoint, kp, kd, ki):
    """Pendulum control law
    Args:
        y (n x 1 np.ndarray): [theta1, omega1, theta2, omega2]
        t (float): time
        setpoint (m x 1 np.ndarray): [setpoint_theta1, setpoint_theta2] setpoint for theta1 and theta2
        Kp (2 x 1 np.ndarray): proportional gain for omega_dot1 and omega_dot2
        Kd (2 x 1 np.ndarray): derivative gain for omega_dot1 and omega_dot2
        Ki (2 x 1 np.ndarray): integral gain for omega_dot1 and omega_dot2
    Returns:
        n x 1 np.ndarray: control input
    """
    theta1 = y[0]
    theta2 = y[2]

    e = np.array(setpoint) - np.array([theta1, theta2])
    errors_array = np.asarray(list(errors_dict.values()))
    e_dot = (e - errors_array[-1]) / dt
    e_sum = np.sum(errors_array, axis=0)

    errors_dict[t] = e
    control_signal[t] = kp * e + kd * e_dot + ki * e_sum * dt

    return control_signal[t]


if __name__ == '__main__':
    # pendulum initial conditions
    y0 = np.array([3 * np.pi/2, 0, np.pi/4, 0])

    # pendulum time span
    dt = 0.02
    tspan = np.arange(0, 100, dt).round(2)

    MAX_DIFFUSION = 0.1  # maximum diffusion strength
    C = 1e-2  # damping

    SETPOINT_THETA1 = np.pi
    SETPOINT_THETA2 = np.pi/4

    KP = 10
    KD = 10
    KI = 0.1

    ito = itoint(
        f=lambda y, t: ode(y, t, C, control_input(y, t, [SETPOINT_THETA1, SETPOINT_THETA2], KP, KD, KI)),
        G=lambda y, t: diffusion(y, t, MAX_DIFFUSION),
        y0=y0,
        tspan=tspan)

    df = pd.DataFrame(ito, columns=['theta1', 'omega1', 'theta2', 'omega2'])
    df['t'] = tspan
    df['x1'] = np.sin(df['theta1'])
    df['y1'] = -np.cos(df['theta1'])
    df['x2'] = df['x1'] + np.sin(df['theta2'])
    df['y2'] = df['y1'] - np.cos(df['theta2'])
    df['setpoint_theta1'] = SETPOINT_THETA1
    df['setpoint_theta2'] = SETPOINT_THETA2

    px.line(df, x='t', y=['theta1', 'theta2', 'setpoint_theta1', 'setpoint_theta2']).show()
    px.scatter(df, x='x2', y='y2', color='t', size_max=1, opacity=0.7).show()

    double_pendulum_animation(df, dt)
