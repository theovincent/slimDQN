import numpy as np
import jax.numpy as jnp
import unittest

from experiments.base.compute_iqm import get_iqm_and_conf_per_epoch


class TestIQM(unittest.TestCase):

    def setUp(self):
        self._x = jnp.array(
            [[[1, 2.2]], [[1.1, 2]], [[1, 2.5]], [[0.5, 1.5]], [[1, 2]], [[1.7, 1.1]], [[1, 2]], [[0.9, 2]]]
        )
        self._x_nan = jnp.array(
            [
                [[1, 2.2]],
                [[1.1, jnp.nan]],
                [[1, jnp.nan]],
                [[jnp.nan, 1.5]],
                [[1, jnp.nan]],
                [[jnp.nan, jnp.nan]],
                [[jnp.nan, 2]],
                [[0.9, 2]],
            ]
        )

    def test_one_game_one_seed(self):
        z = np.random.rand(10).reshape(1, 1, 10)
        iqm, conf = get_iqm_and_conf_per_epoch(z)
        np.testing.assert_array_almost_equal(iqm, z.reshape(-1))
        np.testing.assert_array_almost_equal(conf[0], z.reshape(-1))
        np.testing.assert_array_almost_equal(conf[1], z.reshape(-1))

    def test_one_game_many_seeds(self):
        # score matrix defined as n_seeds x n_games x n_epochs
        x, x_nan = self._x.copy(), self._x_nan.copy()

        iqm, conf = get_iqm_and_conf_per_epoch(x)
        np.testing.assert_array_equal(iqm, [1, 2])
        np.testing.assert_array_less(conf[0], [1, 2])
        np.testing.assert_array_less([1, 2], conf[1])

        iqm_nan, conf_nan = get_iqm_and_conf_per_epoch(x_nan)
        np.testing.assert_array_equal(iqm_nan, [1, 2])
        np.testing.assert_array_less(conf_nan[0], [1, 2])
        np.testing.assert_array_less([1, 2], conf_nan[1])

        # tighter CFs in nan case
        np.testing.assert_array_less(conf[0], conf_nan[0])
        np.testing.assert_array_less(conf_nan[1], conf[1])

    def test_one_seed_many_games(self):
        # score matrix defined as n_seeds x n_games x n_epochs

        x, x_nan = self._x.copy().reshape(1, -1, 2), self._x_nan.copy().reshape(1, -1, 2)

        iqm, conf = get_iqm_and_conf_per_epoch(x)
        np.testing.assert_array_equal(iqm, [1, 2])
        np.testing.assert_array_equal(conf[0], [1, 2])
        np.testing.assert_array_equal(conf[1], [1, 2])

        iqm_nan, conf_nan = get_iqm_and_conf_per_epoch(x_nan)
        np.testing.assert_array_equal(iqm_nan, [1, 2])
        np.testing.assert_array_equal(conf_nan[0], [1, 2])
        np.testing.assert_array_equal(conf_nan[1], [1, 2])

    def test_many_seed_many_games(self):

        n_seeds, n_games, n_epochs = 5, 5, 10
        z = np.random.rand(n_seeds * n_games * n_epochs).reshape(n_seeds, n_games, n_epochs)
        z_nan = np.where(
            np.random.rand(n_seeds * n_games * n_epochs).reshape(n_seeds, n_games, n_epochs) < 0.2, np.nan, z
        )

        iqm, conf = get_iqm_and_conf_per_epoch(z)
        np.testing.assert_array_less(conf[0], iqm)
        np.testing.assert_array_less(iqm, conf[1])

        iqm_nan, conf_nan = get_iqm_and_conf_per_epoch(z_nan)
        np.testing.assert_array_less(conf_nan[0], iqm_nan)
        np.testing.assert_array_less(iqm_nan, conf_nan[1])

    def test_all_nans_in_some_epochs(self):

        n_seeds, n_games, n_epochs = 5, 5, 10
        z = np.random.rand(n_seeds * n_games * n_epochs).reshape(n_seeds, n_games, n_epochs)
        z[:, :, 5:9] = np.nan

        iqm, conf = get_iqm_and_conf_per_epoch(z)
        np.testing.assert_array_equal(iqm[5:9], np.nan)
        np.testing.assert_array_equal(conf[0][5:9], np.nan)
        np.testing.assert_array_equal(conf[1][5:9], np.nan)


if __name__ == "__main__":
    unittest.main()
