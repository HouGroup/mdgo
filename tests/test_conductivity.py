import os
import sys
import tempfile
from io import StringIO
import unittest
import MDAnalysis
from mdgo.conductivity import *

test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_files")


class ConductivityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.gen2 = MDAnalysis.Universe(
            os.path.join(test_dir, "gen2_light", "gen2_mdgo.data"),
            os.path.join(test_dir, "gen2_light", "gen2_mdgo_unwrapped_nvt_main.dcd"),
            format="LAMMPS",
        )
        cls.anions = cls.gen2.select_atoms("type 1")
        cls.cations = cls.gen2.select_atoms("type 3")
        cls.cond_array = calc_cond_msd(cls.gen2, cls.anions, cls.cations, 100, 1, -1)
        cls.time_array = np.array([i * 10 for i in range(cls.gen2.trajectory.n_frames - 100)])

    def test_calc_cond_msd(self):
        self.assertEqual(-2.9103830456733704e-11, self.cond_array[0])
        self.assertEqual(112.66080481783138, self.cond_array[1])
        self.assertEqual(236007.76624833583, self.cond_array[-1])

    def test_get_beta(self):
        self.assertEqual(
            (0.8188201425517928, 0.2535110576154693),
            get_beta(self.cond_array, self.time_array, 10, 100),
        )
        self.assertEqual(
            (1.2525648107674503, 1.0120346984003845),
            get_beta(self.cond_array, self.time_array, 1000, 2000),
        )
        self.assertEqual(
            (1.4075552564189142, 1.3748981878979976),
            get_beta(self.cond_array, self.time_array, 1500, 2500),
        )
        self.assertEqual(
            (1.5021915651236932, 51.79451695748163),
            get_beta(self.cond_array, self.time_array, 2000, 4000),
        )


if __name__ == "__main__":
    unittest.main()
