import jax.numpy as jnp


def test_gpu(capfd):
    assert next(iter(jnp.zeros(8).devices())).platform == "gpu", "The GPU is not being used"

    _, err = capfd.readouterr()
    assert err == "", f"An error message has been raised: {err}"
