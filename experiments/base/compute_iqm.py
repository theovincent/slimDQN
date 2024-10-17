# from multiprocessing import Pool

import jax
import jax.numpy as jnp
import scipy.stats
import numpy as np


def get_iqm_and_conf_per_epoch(array: jnp.ndarray, n_bootstraps: int = 2000):
    n_seeds, n_epochs = array.shape
    if n_seeds == 1:
        return array.reshape(-1), np.stack([array.reshape(-1), array.reshape(-1)])

    key = jax.random.key(seed=0)
    iqm = scipy.stats.trim_mean(jnp.sort(array, axis=0), proportiontocut=0.25, axis=0)

    bootstrap_key, key = jax.random.split(key=key)
    bootstrap_keys = jax.random.split(bootstrap_key, num=n_epochs)
    bootstrap_samples = jax.vmap(
        lambda k, row: jax.random.choice(k, row, shape=(n_bootstraps, n_seeds)), in_axes=(0, 1)
    )(bootstrap_keys, array)

    bootstrap_samples = scipy.stats.trim_mean(jnp.sort(bootstrap_samples, axis=-1), proportiontocut=0.25, axis=1)
    confs = jnp.percentile(bootstrap_samples, q=jnp.array([2.5, 97.5]), axis=1)

    return np.array(iqm), np.array(confs)
