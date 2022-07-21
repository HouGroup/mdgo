import os
import unittest

import numpy as np
import MDAnalysis

try:
    import tidynamics as td
except ImportError:
    td = None

from mdgo.msd import *


test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_files")


class MyTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.arr1 = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [1, 1, 0], [4, 4, 2]])
        n = 100
        cls.arr2 = np.cumsum(np.random.choice([-1.0, 0.0, 1.0], size=(n, 3)), axis=0)
        cls.fft = np.array([0.0, 8.5, 7.0, 10.5, 36.0])
        cls.gen2 = MDAnalysis.Universe(
            os.path.join(test_dir, "gen2_light", "gen2_mdgo.data"),
            os.path.join(test_dir, "gen2_light", "gen2_mdgo_unwrapped_nvt_main.dcd"),
            format="LAMMPS",
        )
        cls.mda_msd_cation = mda_msd_wrapper(cls.gen2, 0, 100, select="type 3", fft=False)
        cls.mda_msd_anion = mda_msd_wrapper(cls.gen2, 0, 100, select="type 1", fft=False)
        cls.onsager_ii_self = onsager_ii_self(cls.gen2, 0, 100, select="type 3")
        cls.onsager_ii_self_nocom = onsager_ii_self(cls.gen2, 0, 100, select="type 3", center_of_mass=False)
        cls.onsager_ii_self_nofft = onsager_ii_self(cls.gen2, 0, 100, select="type 3", fft=False)

    def test_msd_straight_forward(self):
        assert np.allclose(self.fft, msd_straight_forward(self.arr1))

    def test_msd_fft(self):
        assert np.allclose(self.fft, msd_fft(self.arr1))
        assert np.allclose(msd_straight_forward(self.arr2), msd_fft(self.arr2))

    def test_onsager_ii_self(self):
        self.assertAlmostEqual(32.14254152556588, self.onsager_ii_self[50])
        self.assertAlmostEqual(63.62190983, self.onsager_ii_self[98])
        self.assertAlmostEqual(67.29990019, self.onsager_ii_self[99])
        self.assertAlmostEqual(32.14254152556588, self.onsager_ii_self_nofft[50])
        self.assertAlmostEqual(63.62190983, self.onsager_ii_self_nofft[98])
        self.assertAlmostEqual(67.29990019, self.onsager_ii_self_nofft[99])
        self.assertAlmostEqual(32.338364098424634, self.onsager_ii_self_nocom[50])
        self.assertAlmostEqual(63.52915984813752, self.onsager_ii_self_nocom[98])
        self.assertAlmostEqual(67.29599346166411, self.onsager_ii_self_nocom[99])

    def test_mda_msd_wrapper(self):
        self.assertAlmostEqual(32.338364098424634, self.mda_msd_cation[50])
        self.assertAlmostEqual(63.52915984813752, self.mda_msd_cation[98])
        self.assertAlmostEqual(67.29599346166411, self.mda_msd_cation[99])
        self.assertAlmostEqual(42.69200176568008, self.mda_msd_anion[50])
        self.assertAlmostEqual(86.9209518, self.mda_msd_anion[98])
        self.assertAlmostEqual(89.84668178, self.mda_msd_anion[99])
        assert np.allclose(
            onsager_ii_self(self.gen2, 0, 10, select="type 3", msd_type="x", center_of_mass=False),
            mda_msd_wrapper(self.gen2, 0, 10, select="type 3", msd_type="x", fft=False),
        )
        assert np.allclose(
            onsager_ii_self(self.gen2, 0, 10, select="type 3", msd_type="y", center_of_mass=False),
            mda_msd_wrapper(self.gen2, 0, 10, select="type 3", msd_type="y", fft=False),
        )
        assert np.allclose(
            onsager_ii_self(self.gen2, 0, 10, select="type 3", msd_type="z", center_of_mass=False),
            mda_msd_wrapper(self.gen2, 0, 10, select="type 3", msd_type="z", fft=False),
        )
        assert np.allclose(
            onsager_ii_self(self.gen2, 0, 100, select="type 3", msd_type="xy", center_of_mass=False),
            mda_msd_wrapper(self.gen2, 0, 100, select="type 3", msd_type="xy", fft=False),
        )
        assert np.allclose(
            onsager_ii_self(self.gen2, 0, 100, select="type 3", msd_type="yz", center_of_mass=False),
            mda_msd_wrapper(self.gen2, 0, 100, select="type 3", msd_type="yz", fft=False),
        )
        assert np.allclose(
            onsager_ii_self(self.gen2, 0, 100, select="type 3", msd_type="xz", center_of_mass=False),
            mda_msd_wrapper(self.gen2, 0, 100, select="type 3", msd_type="xz", fft=False),
        )
        if td is not None:
            assert np.allclose(self.mda_msd_cation, mda_msd_wrapper(self.gen2, 0, 100, select="type 3"))


if __name__ == "__main__":
    unittest.main()
