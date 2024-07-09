import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from slimRL.environments.car_on_hill import CarOnHill
from experiments.base.iqm import get_iqm_and_conf_parallel
from experiments.car_on_hill.optimal import NX, NV


def plot_on_grid(values, shared_cmap, zeros_to_nan=False, **kwargs):

    plt.rc(
        "font", size=kwargs.get("fontsize", 15), family="serif", serif="Times New Roman"
    )
    plt.rc("lines", linewidth=kwargs.get("linewidth", 5))
    nrows = int(np.ceil(len(values) / 3.0))
    ncols = min(len(values), 3)
    fig, ax = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        sharex=True,
        sharey=True,
        figsize=(7 * ncols, 5.6 * nrows),
    )

    if nrows == 1:
        ax = [ax]
    if ncols == 1:
        ax = [ax]

    env = CarOnHill()
    states_x = np.linspace(-env.max_pos, env.max_pos, NX)
    states_v = np.linspace(-env.max_velocity, env.max_velocity, NV)
    x, v = np.meshgrid(states_x, states_v, indexing="ij")

    if zeros_to_nan:
        for key, val in values.items():
            values[key] = np.where(val != 0, val, np.nan)

    colors = []
    for idx, (key, val) in enumerate(values.items()):
        colors.append(
            ax[idx // 3][idx % 3].pcolormesh(
                x, v, val, shading="nearest", cmap=kwargs.get("cmap", "viridis")
            )
        )
    tick_size = kwargs.get("tick_size", 2)

    for col in range(ncols):
        ax[-1][col].set_xticks(states_x[::tick_size])
        ax[-1][col].set_xticklabels(
            np.around(states_x[::tick_size], 1), rotation="vertical"
        )
        ax[-1][col].set_xlim(states_x[0], states_x[-1])
        ax[-1][col].set_xlabel("$x$")

    for row in range(nrows):
        ax[row][0].set_yticks(states_v[::tick_size])
        ax[row][0].set_yticklabels(np.around(states_v[::tick_size], 1))
        ax[row][0].set_ylim(states_v[0], states_v[-1])
        ax[row][0].set_ylabel("$v$")

    for idx, key in enumerate(values.keys()):
        ax[idx // 3][idx % 3].set_title(key)
        ax[idx // 3][idx % 3].set_box_aspect(1)

    if shared_cmap:
        min_val = min([np.nanmin(val) for val in values.values()])
        max_val = max([np.nanmax(val) for val in values.values()])

        norm = matplotlib.colors.Normalize(vmin=min_val, vmax=max_val)
        cmap = matplotlib.cm.ScalarMappable(
            norm=norm, cmap=kwargs.get("cmap", "viridis")
        )
        cmap.set_array([])
        fig.colorbar(cmap, ax=ax)
    else:
        for idx, (key, val) in enumerate(values.items()):
            fig.colorbar(colors[idx], ax=ax[idx // 3][idx % 3])
    fig.suptitle(kwargs.get("title", ""), fontsize=25)
    if not shared_cmap:
        plt.subplots_adjust(wspace=0.05, hspace=0.2)

    return plt


def plot_value(xlabel, ylabel, x_val, y_val, xlim, xticks, **kwargs):
    plt.rc(
        "font", size=kwargs.get("fontsize", 15), family="serif", serif="Times New Roman"
    )
    plt.rc("lines", linewidth=kwargs.get("linewidth", 4))
    fig = plt.figure(kwargs.get("title", ""))
    ax = fig.add_subplot(111)
    plt.xlabel(xlabel, fontsize=kwargs.get("fontsize", 15))
    plt.ylabel(ylabel, fontsize=kwargs.get("fontsize", 15))
    plt.title(kwargs.get("title", ""))
    ax.set_xlim(xlim)
    ax.set_xticks(xticks)
    if kwargs.get("yticks", None) is not None:
        ax.set_yticks(kwargs.get("yticks"))
    if kwargs.get("ylim", None) is not None:
        ax.set_ylim(kwargs.get("ylim"))

    for exp in y_val:
        y_iqm, y_cnf = get_iqm_and_conf_parallel(y_val[exp])
        ax.plot(
            x_val,
            y_iqm,
            label=exp,
        )
        ax.fill_between(
            x_val,
            y_cnf[0],
            y_cnf[1],
            alpha=0.3,
        )

    if kwargs.get("sci_x", False):
        plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    if kwargs.get("sci_y", False):
        plt.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    plt.legend(loc="lower right")
    plt.grid()
    plt.tight_layout()

    return plt
