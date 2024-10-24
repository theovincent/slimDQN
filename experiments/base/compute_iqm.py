import jax
import jax.numpy as jnp
import scipy.stats
import numpy as np


def get_iqm_and_conf_per_epoch(array: np.ndarray, n_bootstraps: int = 2000):
    n_seeds, n_tasks, n_epochs = array.shape
    if n_tasks == 1 and n_seeds == 1:
        return array.reshape(-1), np.stack([array.reshape(-1), array.reshape(-1)])

    key = jax.random.key(seed=0)
    iqm = jnp.array(
        [scipy.stats.trim_mean(array[..., epoch], proportiontocut=0.25, axis=None) for epoch in range(n_epochs)]
    )

    bootstrap_key, key = jax.random.split(key=key)
    bootstrap_keys = jax.random.split(bootstrap_key, num=n_epochs * n_tasks).reshape((n_epochs, n_tasks))
    bootstrap_samples = jax.vmap(
        lambda k, a: jnp.array(
            [
                jax.random.choice(key=k[idx_task], a=a[..., idx_task], shape=(n_seeds, n_bootstraps))
                for idx_task in range(n_tasks)
            ]
        ),
        in_axes=(0, -1),
    )(bootstrap_keys, array)

    bootstrap_samples = bootstrap_samples.reshape((n_epochs, -1, n_bootstraps))

    bootstrap_samples = scipy.stats.trim_mean(bootstrap_samples, proportiontocut=0.25, axis=1)
    confs = jnp.percentile(bootstrap_samples, q=jnp.array([2.5, 97.5]), axis=1)

    return iqm, confs
