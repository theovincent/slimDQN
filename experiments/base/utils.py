import numpy as np
from scipy.stats import t as student_variable


def single_confidence_interval(mean, std, n_samples, confidence_level):
    t_crit = np.abs(student_variable.ppf((1 - confidence_level) / 2, n_samples - 1))
    lower_bound = mean - t_crit * (std / np.sqrt(n_samples))
    upper_bound = mean + t_crit * (std / np.sqrt(n_samples))

    return lower_bound, upper_bound


def confidence_interval(means, stds, n_samples, confidence_level=0.95):
    confidence_intervals = np.zeros((2, len(means)))

    if n_samples == 1:
        confidence_intervals[0] = means
        confidence_intervals[1] = means
    else:
        for idx_iteration in range(len(means)):
            (
                confidence_intervals[0, idx_iteration],
                confidence_intervals[1, idx_iteration],
            ) = single_confidence_interval(
                means[idx_iteration], stds[idx_iteration], n_samples, confidence_level
            )

    return confidence_intervals
