import scipy
import jax
import numpy as np


def get_epoch_iqm_and_conf(key, epoch_scores, n_bootstraps=2000):
    epoch_iqm = scipy.stats.trim_mean(epoch_scores[~np.isnan(epoch_scores)], proportiontocut=0.25, axis=None)

    task_keys = jax.random.split(key, epoch_scores.shape[0])
    sampled_epoch_scores = []
    for idx_task in range(epoch_scores.shape[0]):
        nan_free_tasks = epoch_scores[idx_task][~np.isnan(epoch_scores[idx_task])]
        sampled_epoch_scores.append(
            jax.random.choice(
                key=task_keys[idx_task],
                a=nan_free_tasks,
                shape=(len(nan_free_tasks), n_bootstraps),
            )
        )

    sampled_epoch_iqm = scipy.stats.trim_mean(np.vstack(sampled_epoch_scores), proportiontocut=0.25, axis=0)
    lower_bound, upper_bound = np.percentile(sampled_epoch_iqm, q=np.array([2.5, 97.5]))
    return epoch_iqm, lower_bound, upper_bound


def get_iqm_and_conf(scores):
    # scores: n_seeds x n_epochs or n_tasks x n_seeds x n_epochs
    if scores.ndim == 2:
        scores = scores[np.newaxis]

    iqm, ci_lower, ci_upper = np.vectorize(get_epoch_iqm_and_conf, signature="(j),(n,m)-> (),(),()")(
        jax.random.split(jax.random.PRNGKey(0), scores.shape[-1]), scores.transpose((2, 0, 1))
    )

    return iqm, np.stack([ci_lower, ci_upper])
