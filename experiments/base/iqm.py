import numpy as np
import scipy.stats
from multiprocessing import Pool


def get_iqm_and_conf(array, bootstraps=2000, percentile=0.95):
    num_seeds, epochs = array.shape
    mean = np.zeros((epochs))
    conf = np.zeros((2, epochs))

    cut_off = (1 - percentile) / 2.0

    for e in range(epochs):
        # calculate IQM
        sorted_epoch = np.sort(array[:, e])
        mean[e] = scipy.stats.trim_mean(sorted_epoch, proportiontocut=0.25, axis=None)

        # calculate confidence interval
        bootstrap_means = []
        for _ in range(bootstraps):
            vals = np.random.choice(array[:, e], size=num_seeds)
            vals = np.sort(vals)
            bootstrap_iqm_estimate = scipy.stats.trim_mean(
                vals, proportiontocut=0.25, axis=None
            )
            bootstrap_means.append(bootstrap_iqm_estimate)

        bootstrap_means = np.array(bootstrap_means)
        conf[:, e] = np.percentile(
            bootstrap_means, [cut_off * 100, (1.0 - cut_off) * 100]
        )

    return mean, conf


def compute_epoch_mean_conf(args):
    array, num_seeds, bootstraps, cut_off = args

    sorted_array = np.sort(array)
    mean = scipy.stats.trim_mean(sorted_array, proportiontocut=0.25, axis=None)

    bootstrap_means = []
    for _ in range(bootstraps):
        vals = np.random.choice(array, size=num_seeds)
        vals = np.sort(vals)
        bootstrap_iqm_estimate = scipy.stats.trim_mean(
            vals, proportiontocut=0.25, axis=None
        )
        bootstrap_means.append(bootstrap_iqm_estimate)
    bootstrap_means = np.array(bootstrap_means)
    conf = np.percentile(bootstrap_means, [cut_off, 100 - cut_off])

    return mean, conf


def get_iqm_and_conf_parallel(array, bootstraps=2000, percentile=0.95):
    cut_off = (1.0 - percentile) / 2
    num_seeds, epochs = array.shape

    args = [
        (array[:, epoch], num_seeds, bootstraps, cut_off) for epoch in range(epochs)
    ]

    with Pool() as pool:
        results = pool.map(compute_epoch_mean_conf, args)

    means, confs = zip(*results)
    mean = np.array(means)
    conf = np.stack(confs, axis=1)

    return mean, conf
