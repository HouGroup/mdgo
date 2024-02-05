from __future__ import annotations

import os
import unittest

from numpy.testing import assert_allclose
from pymatgen.core import Molecule

from mdgo.util.volume import molecular_volume

test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_files")


class MyTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ec = Molecule.from_file(filename=os.path.join(test_dir, "EC.xyz"))
        cls.emc = Molecule.from_file(filename=os.path.join(test_dir, "EMC.xyz"))
        cls.dec = Molecule.from_file(filename=os.path.join(test_dir, "DEC.xyz"))
        cls.pf6 = Molecule.from_file(filename=os.path.join(test_dir, "PF6.xyz"))
        cls.tfsi = Molecule.from_file(filename=os.path.join(test_dir, "TFSI.xyz"))
        cls.litfsi = Molecule.from_file(filename=os.path.join(test_dir, "LiTFSI.xyz"))
        cls.lipf6 = Molecule.from_file(filename=os.path.join(test_dir, "LiPF6.xyz"))

    def test_molecular_volume(self) -> None:
        lipf6_volume_1 = molecular_volume(self.lipf6)
        lipf6_volume_2 = molecular_volume(self.lipf6, res=1.0)
        lipf6_volume_3 = molecular_volume(self.lipf6, radii_type="Lange")
        lipf6_volume_4 = molecular_volume(self.lipf6, radii_type="pymatgen")
        lipf6_volume_5 = molecular_volume(self.lipf6, molar_volume=False)
        assert_allclose(lipf6_volume_1, 47.62, atol=0.01)
        assert_allclose(lipf6_volume_2, 43.36, atol=0.01)
        assert_allclose(lipf6_volume_3, 41.49, atol=0.01)
        assert_allclose(lipf6_volume_4, 51.94, atol=0.01)
        assert_allclose(lipf6_volume_5, 79.08, atol=0.01)
        ec_volume_1 = molecular_volume(self.ec)
        ec_volume_2 = molecular_volume(self.ec, exclude_h=False)
        ec_volume_3 = molecular_volume(self.ec, res=1.0)
        ec_volume_4 = molecular_volume(self.ec, radii_type="Lange")
        ec_volume_5 = molecular_volume(self.ec, radii_type="pymatgen")
        ec_volume_6 = molecular_volume(self.ec, molar_volume=False)
        assert_allclose(ec_volume_1, 38.44, atol=0.01)
        assert_allclose(ec_volume_2, 43.17, atol=0.01)
        assert_allclose(ec_volume_3, 40.95, atol=0.01)
        assert_allclose(ec_volume_4, 41.07, atol=0.01)
        assert_allclose(ec_volume_5, 38.44, atol=0.01)
        assert_allclose(ec_volume_6, 63.83, atol=0.01)
        litfsi_volume_1 = molecular_volume(self.litfsi)
        litfsi_volume_2 = molecular_volume(self.litfsi, exclude_h=False)
        litfsi_volume_3 = molecular_volume(self.litfsi, res=1.0)
        litfsi_volume_4 = molecular_volume(self.litfsi, radii_type="Lange")
        litfsi_volume_5 = molecular_volume(self.litfsi, radii_type="pymatgen")
        litfsi_volume_6 = molecular_volume(self.litfsi, molar_volume=False)
        litfsi_volume_7 = molecular_volume(self.litfsi, mode="act", x_size=8, y_size=8, z_size=8)
        assert_allclose(litfsi_volume_1, 100.16, atol=0.01)
        assert_allclose(litfsi_volume_2, 100.16, atol=0.01)
        assert_allclose(litfsi_volume_3, 99.37, atol=0.01)
        assert_allclose(litfsi_volume_4, 90.78, atol=0.01)
        assert_allclose(litfsi_volume_5, 105.31, atol=0.01)
        assert_allclose(litfsi_volume_6, 166.32, atol=0.01)
        assert_allclose(litfsi_volume_7, 124.66, atol=0.01)


if __name__ == "__main__":
    unittest.main()
