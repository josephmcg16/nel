import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque


def double_pendulum_animation(df, dt):
    """Double pendulum animation using matplotlib FuncAnimation
    Args:
        df (pd.DataFrame): dataframe with columns ['x1', 'y1', 'x2', 'y2']
        dt (float): time step"""
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, autoscale_on=False,
                        xlim=(-2, 2), ylim=(-2, 2))
    ax.set_aspect('equal')
    ax.grid()

    line, = ax.plot([], [], 'o-', lw=2)
    time_template = 'time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    # Create deque objects with maxlen to hold the latest positions of pendulum
    history_length = 200  # Length of the path trace
    x2_history = deque(maxlen=history_length)
    y2_history = deque(maxlen=history_length)

    # Create trace_line for the path of pendulum
    trace_line, = ax.plot([], [], '-', color='blue', lw=1, alpha=0.5)

    def init():
        """initialize animation"""
        line.set_data([], [])
        trace_line.set_data([], [])
        time_text.set_text('')
        return line, trace_line, time_text

    def animate(i):
        """perform animation step"""
        thisx = [0, df['x1'][i], df['x2'][i]]
        thisy = [0, df['y1'][i], df['y2'][i]]

        x2_history.append(df['x2'][i])  # Store x2 position in history
        y2_history.append(df['y2'][i])  # Store y2 position in history

        # Draw trace_line using history
        trace_line.set_data(list(x2_history), list(y2_history))

        line.set_data(thisx, thisy)
        time_text.set_text(time_template % (i * dt))
        return line, trace_line, time_text

    ani = FuncAnimation(
        fig,
        animate,
        frames=range(1, len(df)),
        interval=dt * 1000,
        blit=True,
        init_func=init)
    plt.show()
    return ani, fig
