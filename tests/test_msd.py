from __future__ import annotations
import os
import unittest

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
        cls.dims = ["x", "y", "z"]

    def test_msd_straight_forward(self):
        assert np.allclose(self.fft, msd_straight_forward(self.arr1))

    def test_msd_fft(self):
        assert np.allclose(self.fft, msd_fft(self.arr1))
        assert np.allclose(msd_straight_forward(self.arr2), msd_fft(self.arr2))

    def test_create_position_arrays(self):
        assert np.allclose(
            np.array([21.53381769, 14.97501839, -3.87998785]),
            create_position_arrays(self.gen2, 0, 100, select="type 3")[50][2]
        )
        assert np.allclose(
            np.array([-2.78550047, -11.85487624, -17.1221954]),
            create_position_arrays(self.gen2, 0, 100, select="type 3")[99][10]
        )
        assert np.allclose(
            np.array([41.1079216 , 34.95127106, 18.00482368]),
            create_position_arrays(self.gen2, 0, 100, select="type 3", center_of_mass=False)[50][2]
        )
        assert np.allclose(
            np.array([16.98478317,  8.27190208,  5.07116079]),
            create_position_arrays(self.gen2, 0, 100, select="type 3", center_of_mass=False)[99][10]
        )

    def test_parse_msd_type(self):
        xyz = parse_msd_type("xyz")
        self.assertEqual(["x", "y", "z"], self.dims[xyz[0]:xyz[1]:xyz[2]])
        xy = parse_msd_type("xy")
        self.assertEqual(["x", "y"], self.dims[xy[0]:xy[1]:xy[2]])
        yz = parse_msd_type("yz")
        self.assertEqual(["y", "z"], self.dims[yz[0]:yz[1]:yz[2]])
        xz = parse_msd_type("xz")
        self.assertEqual(["x", "z"], self.dims[xz[0]:xz[1]:xz[2]])
        x = parse_msd_type("x")
        self.assertEqual(["x"], self.dims[x[0]:x[1]:x[2]])
        y = parse_msd_type("y")
        self.assertEqual(["y"], self.dims[y[0]:y[1]:y[2]])
        z = parse_msd_type("z")
        self.assertEqual(["z"], self.dims[z[0]:z[1]:z[2]])

    def test_onsager_ii_self(self):
        onsager_ii_self_fft = onsager_ii_self(self.gen2, 0, 100, select="type 3")
        onsager_ii_self_nocom = onsager_ii_self(self.gen2, 0, 100, select="type 3", center_of_mass=False)
        onsager_ii_self_nofft = onsager_ii_self(self.gen2, 0, 100, select="type 3", fft=False)

        self.assertAlmostEqual(32.14254152556588, onsager_ii_self_fft[50])
        self.assertAlmostEqual(63.62190983, onsager_ii_self_fft[98])
        self.assertAlmostEqual(67.29990019, onsager_ii_self_fft[99])
        self.assertAlmostEqual(32.14254152556588, onsager_ii_self_nofft[50])
        self.assertAlmostEqual(63.62190983, onsager_ii_self_nofft[98])
        self.assertAlmostEqual(67.29990019, onsager_ii_self_nofft[99])
        self.assertAlmostEqual(32.338364098424634, onsager_ii_self_nocom[50])
        self.assertAlmostEqual(63.52915984813752, onsager_ii_self_nocom[98])
        self.assertAlmostEqual(67.29599346166411, onsager_ii_self_nocom[99])

    def test_mda_msd_wrapper(self):
        mda_msd_cation = mda_msd_wrapper(self.gen2, 0, 100, select="type 3", fft=False)
        mda_msd_anion = mda_msd_wrapper(self.gen2, 0, 100, select="type 1", fft=False)
        self.assertAlmostEqual(32.338364098424634, mda_msd_cation[50])
        self.assertAlmostEqual(63.52915984813752, mda_msd_cation[98])
        self.assertAlmostEqual(67.29599346166411, mda_msd_cation[99])
        self.assertAlmostEqual(42.69200176568008, mda_msd_anion[50])
        self.assertAlmostEqual(86.9209518, mda_msd_anion[98])
        self.assertAlmostEqual(89.84668178, mda_msd_anion[99])
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
            assert np.allclose(mda_msd_cation, mda_msd_wrapper(self.gen2, 0, 100, select="type 3"))

    def test_total_msd(self):
        total_builtin_cation = total_msd(self.gen2, 0, 100, select="type 3", fft=True, built_in=True)
        total_mda_cation = total_msd(
            self.gen2, 0, 100, select="type 3", fft=False, built_in=False, center_of_mass=False
        )
        self.assertAlmostEqual(32.14254152556588, total_builtin_cation[50])
        self.assertAlmostEqual(32.338364098424634, total_mda_cation[50])
        with self.assertRaises(ValueError):
            total_msd(self.gen2, 0, 100, select="type 3", fft=True, built_in=False, center_of_mass=True)


if __name__ == "__main__":
    unittest.main()
