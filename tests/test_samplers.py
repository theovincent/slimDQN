# coding=utf-8
# Copyright 2024 The Dopamine Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Testing samplers."""

from absl.testing import absltest
from absl.testing import parameterized
from slimdqn.sample_collection import samplers
import numpy as np


class PrioritizedSamplingTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.sampler = samplers.PrioritizedSamplingDistribution(seed=0, max_capacity=10)

    def test_sample(self):
        keys = [0, 1, 2, 3, 4]
        priorities = [1.0, 2.0, 3.0, 4.0, 0.0]

        for key, priority in zip(keys, priorities):
            self.sampler.add(key, priority=priority)

        # test if zero priority absent
        samples = self.sampler.sample(5)
        np.testing.assert_array_less(samples, 4)

        self.sampler.update(keys=np.array([2, 3]), priorities=np.array([0.0, 0.0]))

        # test if priority updated properly
        samples = self.sampler.sample(5)
        np.testing.assert_array_less(samples, 2)

        self.sampler.remove(0)
        samples = self.sampler.sample(5)
        np.testing.assert_array_almost_equal(samples, 1)


if __name__ == "__main__":
    absltest.main()
