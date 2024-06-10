import numpy as np
import scipy.stats
from multiprocessing import Pool


def compute_epoch_iqm_conf(array, n_seeds, n_bootstraps):
    iqm = scipy.stats.trim_mean(np.sort(array), proportiontocut=0.25, axis=None)

    bootstrap_iqms = np.zeros(n_bootstraps)
    for i in range(n_bootstraps):
        vals = np.random.choice(array, size=n_seeds)
        bootstrap_iqms[i] = scipy.stats.trim_mean(
            np.sort(vals), proportiontocut=0.25, axis=None
        )

    confs = np.percentile(bootstrap_iqms, [2.5, 97.5])

    return iqm, confs


def get_iqm_and_conf_parallel(array, n_bootstraps=2000):
    n_seeds, n_epochs = array.shape
    if n_seeds == 1:
        return array.reshape(-1), np.stack([array.reshape(-1), array.reshape(-1)])

    with Pool(n_epochs) as pool:
        results = pool.starmap(
            compute_epoch_iqm_conf,
            [(array[:, epoch], n_seeds, n_bootstraps) for epoch in range(n_epochs)],
        )

    iqms, confs = zip(*results)
    iqm = np.array(iqms)
    conf = np.stack(confs, axis=1)

    return iqm, conf
