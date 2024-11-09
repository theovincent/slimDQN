import scipy
import jax
import numpy as np


def get_iqm_and_conf_per_epoch(scores, n_bootstraps=2000):

    n_seeds, n_tasks, n_epochs = scores.shape

    if n_tasks == 1 and n_seeds == 1:
        return scores.reshape(-1), np.stack([scores.reshape(-1), scores.reshape(-1)])

    iqm = np.array(
        [
            scipy.stats.trim_mean(scores[..., epoch][~np.isnan(scores[..., epoch])], proportiontocut=0.25, axis=None)
            for epoch in range(n_epochs)
        ]
    )

    key = jax.random.PRNGKey(seed=0)

    def compute_ci_for_epoch(epoch_scores, key):
        epoch_scores = epoch_scores.reshape(n_seeds, n_tasks)
        epoch_key, key = jax.random.split(key)
        n_successful_seeds_per_task = np.sum(~np.isnan(epoch_scores), axis=0)

        if np.sum(n_successful_seeds_per_task) == 0:
            return np.full(shape=(2,), fill_value=np.nan)

        task_keys = jax.random.split(epoch_key, n_tasks)

        sampled_scores = np.vstack(
            [
                jax.random.choice(
                    key=task_keys[idx_task],
                    a=epoch_scores[..., idx_task],
                    shape=(n_successful_seeds_per_task[idx_task], n_bootstraps),
                    p=np.isfinite(epoch_scores[..., idx_task]),
                )
                for idx_task in range(n_tasks)
                if n_successful_seeds_per_task[idx_task] > 0
            ]
        )

        scores = scipy.stats.trim_mean(sampled_scores, proportiontocut=0.25, axis=0)

        return np.percentile(scores, q=np.array([2.5, 97.5]), axis=0)

    ci_lower, ci_upper = np.apply_along_axis(
        func1d=compute_ci_for_epoch, axis=0, arr=scores.reshape(-1, n_epochs), key=key
    )

    return iqm, np.stack([ci_lower, ci_upper], axis=0)
