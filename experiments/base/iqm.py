import numpy as np
import scipy.stats
from multiprocessing import Pool


def compute_epoch_mean_conf(args):
    array, num_seeds, bootstraps, cut_off = args
    mean = scipy.stats.trim_mean(np.sort(array), proportiontocut=0.25, axis=None)

    bootstrap_means = np.zeros(bootstraps)
    for i in range(bootstraps):
        vals = np.random.choice(array, size=num_seeds)
        bootstrap_means[i] = scipy.stats.trim_mean(
            np.sort(vals), proportiontocut=0.25, axis=None
        )

    conf = np.percentile(bootstrap_means, [cut_off, 100 - cut_off])

    return mean, conf


def get_iqm_and_conf_parallel(array, bootstraps=2000, percentile=0.95):
    num_seeds, epochs = array.shape
    if num_seeds == 1:
        return array.reshape(-1), np.stack([array.reshape(-1), array.reshape(-1)])

    cut_off = (1.0 - percentile) / 2
    args = [
        (array[:, epoch], num_seeds, bootstraps, cut_off * 100)
        for epoch in range(epochs)
    ]

    with Pool() as pool:
        results = pool.map(compute_epoch_mean_conf, args)

    means, confs = zip(*results)
    mean = np.array(means)
    conf = np.stack(confs, axis=1)

    return mean, conf
