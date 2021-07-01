import os
import unittest
from pymatgen.core import Molecule
from mdgo.volume import molecular_volume

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
        self.assertAlmostEqual(lipf6_volume_1, 47.62, places=2)
        self.assertAlmostEqual(lipf6_volume_2, 43.36, places=2)
        self.assertAlmostEqual(lipf6_volume_3, 41.49, places=2)
        self.assertAlmostEqual(lipf6_volume_4, 51.94, places=2)
        self.assertAlmostEqual(lipf6_volume_5, 79.08, places=2)
        ec_volume_1 = molecular_volume(self.ec)
        ec_volume_2 = molecular_volume(self.ec, exclude_h=False)
        ec_volume_3 = molecular_volume(self.ec, res=1.0)
        ec_volume_4 = molecular_volume(self.ec, radii_type="Lange")
        ec_volume_5 = molecular_volume(self.ec, radii_type="pymatgen")
        ec_volume_6 = molecular_volume(self.ec, molar_volume=False)
        self.assertAlmostEqual(ec_volume_1, 38.44, places=2)
        self.assertAlmostEqual(ec_volume_2, 43.17, places=2)
        self.assertAlmostEqual(ec_volume_3, 40.95, places=2)
        self.assertAlmostEqual(ec_volume_4, 41.07, places=2)
        self.assertAlmostEqual(ec_volume_5, 38.44, places=2)
        self.assertAlmostEqual(ec_volume_6, 63.83, places=2)
        litfsi_volume_1 = molecular_volume(self.litfsi)
        litfsi_volume_2 = molecular_volume(self.litfsi, exclude_h=False)
        litfsi_volume_3 = molecular_volume(self.litfsi, res=1.0)
        litfsi_volume_4 = molecular_volume(self.litfsi, radii_type="Lange")
        litfsi_volume_5 = molecular_volume(self.litfsi, radii_type="pymatgen")
        litfsi_volume_6 = molecular_volume(self.litfsi, molar_volume=False)
        self.assertAlmostEqual(litfsi_volume_1, 100.16, places=2)
        self.assertAlmostEqual(litfsi_volume_2, 100.16, places=2)
        self.assertAlmostEqual(litfsi_volume_3, 99.37, places=2)
        self.assertAlmostEqual(litfsi_volume_4, 90.78, places=2)
        self.assertAlmostEqual(litfsi_volume_5, 105.31, places=2)
        self.assertAlmostEqual(litfsi_volume_6, 166.32, places=2)


if __name__ == "__main__":
    unittest.main()
