import numpy as np
import unittest

from experiments.base.compute_iqm import get_iqm_and_conf


class TestIQM(unittest.TestCase):

    def setUp(self):
        # scores: n_seeds x n_epochs or n_tasks x n_seeds x n_epochs
        self.scores = np.array([[20.4, -20.2], [9, 3], [2.2, 4], [11, 24]])
        self.scores_with_nans = np.array([[np.nan, np.nan], [9, 3], [np.nan, 4], [11, np.nan]])
        self.scores_with_tasks_and_nans = np.array(
            [
                [[1.5, np.nan, -2], [np.nan, 5, np.nan]],  # task 0
                [[np.nan, np.nan, 3], [np.nan, np.nan, 1]],  # task 1
                [[0.5, 6, 70], [10, 0.3, np.nan]],  # task 2
                [[-8, np.nan, np.nan], [np.nan, 5, np.nan]],  # task 3
            ]
        )

    def test_one_task_one_seed(self):
        scores = np.random.rand(10).reshape(1, 10)
        iqm, conf = get_iqm_and_conf(scores)
        np.testing.assert_array_almost_equal(iqm, scores.reshape(-1))
        np.testing.assert_array_almost_equal(conf[0], scores.reshape(-1))
        np.testing.assert_array_almost_equal(conf[1], scores.reshape(-1))

    def test_one_task_many_seeds(self):
        iqm, conf = get_iqm_and_conf(self.scores)
        np.testing.assert_array_equal(iqm, [10, 3.5])  # 10 + 10 / 2 | 3 + 4 / 2
        np.testing.assert_array_less(conf[0], [10, 3.5])
        np.testing.assert_array_less([10, 3.5], conf[1])

        iqm_nan, conf_nan = get_iqm_and_conf(self.scores_with_nans)
        np.testing.assert_array_equal(iqm_nan, [10, 3.5])  # 9 + 11 / 2 | 3 + 4 / 2
        np.testing.assert_array_less(conf_nan[0], [10, 3.5])
        np.testing.assert_array_less([10, 3.5], conf_nan[1])

        # tighter CFs in nan case
        np.testing.assert_array_less(conf[0], conf_nan[0])
        np.testing.assert_array_less(conf_nan[1], conf[1])

    def test_many_tasks_many_seeds(self):
        iqm, conf = get_iqm_and_conf(self.scores_with_tasks_and_nans)
        np.testing.assert_array_equal(iqm, [1, 5, 2])  # 0.5 + 1.5 / 2 | 5 + 5 / 2 | 1 + 3 / 2
        np.testing.assert_array_less(conf[0], [1, 5, 2])
        np.testing.assert_array_less([1, 5, 2], conf[1])

    def test_all_nans_in_some_epochs(self):
        self.scores_with_tasks_and_nans[:, :, -1] = np.nan

        iqm, conf = get_iqm_and_conf(self.scores_with_tasks_and_nans)
        np.testing.assert_array_equal(iqm[-1], np.nan)
        np.testing.assert_array_equal(conf[0][-1], np.nan)
        np.testing.assert_array_equal(conf[1][-1], np.nan)
