from multiprocessing import Pool

import numpy as np
import scipy.stats


def compute_iqm_conf(array: np.ndarray, n_seeds: int, n_bootstraps: int):
    iqm = scipy.stats.trim_mean(np.sort(array), proportiontocut=0.25)

    bootstrap_iqms = np.zeros(n_bootstraps)
    for i in range(n_bootstraps):
        vals = np.random.choice(array, size=n_seeds)
        bootstrap_iqms[i] = scipy.stats.trim_mean(np.sort(vals), proportiontocut=0.25)

    confs = np.percentile(bootstrap_iqms, [2.5, 97.5])

    return iqm, confs


def get_iqm_and_conf_per_epoch(array: np.ndarray, n_bootstraps: int = 2000):
    n_seeds, n_epochs = array.shape
    if n_seeds == 1:
        return array.reshape(-1), np.stack([array.reshape(-1), array.reshape(-1)])

    with Pool(n_epochs) as pool:
        results = pool.starmap(
            compute_iqm_conf,
            [(array[:, epoch], n_seeds, n_bootstraps) for epoch in range(n_epochs)],
        )

    iqms, confs = zip(*results)

    return np.array(iqms), np.stack(confs, axis=1)
