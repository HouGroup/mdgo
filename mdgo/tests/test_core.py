import os
import pathlib
import unittest
from mdgo.core import MdRun


class MdRunTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        gen2_dir = pathlib.Path("../test_files/BN_FEC_LiPF6")
        cls.data_file = gen2_dir / "BN_FEC_LiPF6_split_cation.data"
        cls.unwrapped_dcd = gen2_dir / "data_unwrapped_nvt.dcd"
        select_dict = {"cation": "type 22", "anion": "type 20"}
        cls.md_run = MdRun(cls.data_file, cls.unwrapped_dcd,
                        500, 1000, "BN_FEC_LiPF6", select_dict)

        cls.res_mass_dict = {'Li': 6.94, 'PF6': 144.97,
                             'FEC': 106.05, 'BN': 69.105}
        cls.md_autorun = \
            MdRun.auto_constructor(cls.data_file, cls.unwrapped_dcd,
                                   500, 1000, "BN_FEC_LiPF6", cls.res_mass_dict,
                                   "Li", "PF6", "P", ["BN", "FEC"])

    def test_normal_constructor(self):
        run = self.md_run
        print(run.select_dict)
        print(run.unwrapped_run.atoms.types)
        self.assertEqual(48, len(run.cations))
        self.assertEqual(48, len(run.anion_center))
        self.assertEqual(336, len(run.anions))
        self.assertTrue(set(['anion', 'cation']) ==
                        set(run.select_dict.keys()))

    def test_auto_constructor(self):
        autorun = self.md_autorun
        self.assertEqual(48, len(autorun.cations))
        self.assertEqual(48, len(autorun.anion_center))
        self.assertEqual(336, len(autorun.anions))
        self.assertEqual(2, len(autorun.electrolytes))
        self.assertTrue(set(['anion', 'cation', 'BN', 'FEC']) ==
                        set(autorun.select_dict.keys()))

    def test_get_ion_pairing(self):
        # this function only works with the autorun constructor
        autorun = self.md_autorun
        raw_counts = autorun.get_ion_pairing(700, raw_counts=True)
        self.assertEqual(48, sum(raw_counts.values()))
        print(raw_counts)
        self.assertEqual(34, raw_counts['SSIP'])
        self.assertEqual(14, raw_counts['CIP'])

