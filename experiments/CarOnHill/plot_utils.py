import numpy as np
import matplotlib.pyplot as plt


def plot_on_grid(values, n_states_x, n_states_v, zeros_to_nan=False, tick_size=2):

    plt.rc("font", size=10, family="serif", serif="Times New Roman")
    plt.rc("lines", linewidth=1)
    fig, ax = plt.subplots(figsize=(5.7, 5))

    max_pos = 1.0
    max_velocity = 3.0

    states_x = np.linspace(-max_pos, max_pos, n_states_x)
    states_v = np.linspace(-max_velocity, max_velocity, n_states_v)
    x, v = np.meshgrid(states_x, states_v)

    if zeros_to_nan:
        values = np.where(values == 0, np.nan, values)

    colors = ax.pcolormesh(x, v, values, shading="nearest")

    ax.set_xticks(states_x[::tick_size])
    ax.set_xticklabels(np.around(states_x[::tick_size], 1), rotation="vertical")
    ax.set_xlim(states_x[0], states_x[-1])
    ax.set_xlabel("$x$")

    ax.set_yticks(states_v[::tick_size])
    ax.set_yticklabels(np.around(states_v[::tick_size], 1))
    ax.set_ylim(states_v[0], states_v[-1])
    ax.set_ylabel("$v$")

    fig.colorbar(colors, ax=ax)
    fig.tight_layout()
    fig.canvas.draw()

    plt.show()
