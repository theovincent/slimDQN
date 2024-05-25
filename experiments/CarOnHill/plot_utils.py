import numpy as np
import matplotlib.pyplot as plt
from slimRL.environments.car_on_hill import CarOnHill
from experiments.base.utils import confidence_interval


def plot_on_grid(values, n_states_x, n_states_v, zeros_to_nan=False, tick_size=2):

    plt.rc("font", size=10, family="serif", serif="Times New Roman")
    plt.rc("lines", linewidth=1)
    fig, ax = plt.subplots(figsize=(5.7, 5))

    env = CarOnHill()

    states_x = np.linspace(-env.max_pos, env.max_pos, n_states_x)
    states_v = np.linspace(-env.max_velocity, env.max_velocity, n_states_v)
    x, v = np.meshgrid(states_x, states_v, indexing="ij")

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


def plot_value(xlabel, ylabel, x_val, y_val, **kwargs):
    plt.rc("font", size=15, family="serif", serif="Times New Roman")
    plt.rc("lines", linewidth=1)
    fig = plt.figure(kwargs.get("title", ""))
    ax = fig.add_subplot(111)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    ax.set_xticks(x_val[:: kwargs.get("ticksize", 2)])

    for i, exp in enumerate(y_val):
        y_mean = y_val[exp].mean(axis=0)
        y_std = y_val[exp].std(axis=0)
        y_cnf = confidence_interval(y_mean, y_std, y_val[exp].shape[0])
        ax.plot(
            range(1, y_val[exp].shape[1] + 1, 1),
            y_mean,
            label=exp,
        )
        ax.fill_between(
            range(1, y_val[exp].shape[1] + 1, 1),
            y_cnf[0],
            y_cnf[1],
            alpha=0.3,
        )

    plt.legend()
    plt.tight_layout()
    plt.show()
