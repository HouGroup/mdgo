import os
import unittest

import numpy as np

from mdgo.msd import *


class MyTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.arr1 = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [1, 1, 0], [4, 4, 2]])
        n = 100
        cls.arr2 = np.cumsum(np.random.choice([-1., 0., 1.], size=(n, 3)), axis=0)
        cls.fft = np.array([0.,  8.5,  7., 10.5, 36.])

    def test_msd_straight_forward(self):
        assert np.allclose(self.fft, msd_straight_forward(self.arr1))

    def test_msd_fft(self):
        assert np.allclose(self.fft, msd_fft(self.arr1))
        assert np.allclose(msd_straight_forward(self.arr2), msd_fft(self.arr2))


if __name__ == "__main__":
    unittest.main()
