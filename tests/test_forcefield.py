import os
import sys
import tempfile
from io import StringIO
import unittest
from mdgo.forcefield import *

test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_files")


class FFcrawlerTest(unittest.TestCase):
    def test_chrome(self) -> None:
        with open(os.path.join(test_dir, "EMC.lmp")) as f:
            pdf = f.readlines()
        with open(os.path.join(test_dir, "CCOC(=O)OC.lmp")) as f:
            smiles = f.readlines()
        with open(os.path.join(test_dir, "EMC.lmp.xyz")) as f:
            xyz = f.readlines()
        with open(os.path.join(test_dir, "EMC.gro")) as f:
            gro = f.readlines()
        with open(os.path.join(test_dir, "EMC.itp")) as f:
            itp = f.readlines()

        saved_stdout = sys.stdout
        download_dir = tempfile.mkdtemp()
        try:
            out = StringIO()
            sys.stdout = out

            lpg = FFcrawler(download_dir, xyz=True, gromacs=True)
            lpg.data_from_pdb(os.path.join(test_dir, "EMC.pdb"))
            self.assertEqual(
                out.getvalue(),
                "LigParGen server connected.\n"
                "Structure info uploaded. Rendering force field...\n"
                "Force field file downloaded.\n"
                ".xyz file saved.\n"
                "Force field file saved.\n",
            )
            self.assertTrue(os.path.exists(os.path.join(download_dir, "EMC.lmp")))
            self.assertTrue(os.path.exists(os.path.join(download_dir, "EMC.lmp.xyz")))
            self.assertTrue(os.path.exists(os.path.join(download_dir, "EMC.gro")))
            self.assertTrue(os.path.exists(os.path.join(download_dir, "EMC.itp")))
            with open(os.path.join(download_dir, "EMC.lmp")) as f:
                pdf_actual = f.readlines()
                self.assertListEqual(pdf, pdf_actual)
            with open(os.path.join(download_dir, "EMC.lmp.xyz")) as f:
                xyz_actual = f.readlines()
                self.assertListEqual(xyz, xyz_actual)
            with open(os.path.join(download_dir, "EMC.gro")) as f:
                gro_actual = f.readlines()
                self.assertListEqual(gro, gro_actual)
            with open(os.path.join(download_dir, "EMC.itp")) as f:
                itp_actual = f.readlines()
                self.assertListEqual(itp, itp_actual)
            lpg = FFcrawler(download_dir)
            lpg.data_from_smiles("CCOC(=O)OC")
            with open(os.path.join(download_dir, "CCOC(=O)OC.lmp")) as f:
                smiles_actual = f.readlines()
                self.assertListEqual(smiles[:13], smiles_actual[:13])
                self.assertListEqual(smiles[18:131], smiles_actual[18:131])
                self.assertEqual("     1      1      1 -0.28", smiles_actual[131][:26])
                self.assertEqual("     2      1      2 0.01", smiles_actual[132][:25])
                self.assertEqual("    15      1     15 0.10", smiles_actual[145][:25])
                self.assertListEqual(smiles_actual[146:], smiles[146:])
        finally:
            sys.stdout = saved_stdout
            shutil.rmtree(download_dir)


if __name__ == "__main__":
    unittest.main()
