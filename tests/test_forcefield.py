import os
import sys
import tempfile
import unittest
from io import StringIO

import numpy as np
import pytest

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
            self.assertIn(
                "LigParGen server connected.\n"
                "Structure info uploaded. Rendering force field...\n",
                out.getvalue(),
            )
            self.assertIn(
                "Force field file downloaded.\n"
                ".xyz file saved.\n"
                "Force field file saved.\n",
                out.getvalue(),
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


class AqueousTest(unittest.TestCase):
    def test_get_ion(self):
        """
        Some unit tests for get_ion
        """
        # string input, all lowercase
        cation_ff = Aqueous.get_ion(parameter_set="lm", water_model="opc3", ion="li+")
        assert isinstance(cation_ff, LammpsData)
        assert np.allclose(cation_ff.force_field["Pair Coeffs"]["coeff2"].item(), 2.354, atol=0.001)
        assert np.allclose(cation_ff.force_field["Pair Coeffs"]["coeff1"].item(), 0.0064158, atol=0.0000001)

        # string input, using the default ion parameter set for the water model
        cation_ff = Aqueous.get_ion(parameter_set="lm", water_model="opc3", ion="li+")
        assert isinstance(cation_ff, LammpsData)
        assert np.allclose(cation_ff.force_field["Pair Coeffs"]["coeff2"].item(), 2.354, atol=0.001)
        assert np.allclose(cation_ff.force_field["Pair Coeffs"]["coeff1"].item(), 0.0064158, atol=0.0000001)

        # Ion object input, all lowercase
        li = Ion.from_formula("Li+")
        cation_ff = Aqueous.get_ion(parameter_set="jc", water_model="spce", ion=li)
        assert isinstance(cation_ff, LammpsData)
        assert np.allclose(cation_ff.force_field["Pair Coeffs"]["coeff2"].item(), 1.409, atol=0.001)
        assert np.allclose(cation_ff.force_field["Pair Coeffs"]["coeff1"].item(), 0.3367344, atol=0.0000001)

        # anion
        anion_ff = Aqueous.get_ion(parameter_set="jj", water_model="tip4p", ion="F-")
        assert isinstance(anion_ff, LammpsData)
        assert np.allclose(anion_ff.force_field["Pair Coeffs"]["coeff2"].item(), 3.05, atol=0.001)
        assert np.allclose(anion_ff.force_field["Pair Coeffs"]["coeff1"].item(), 0.71, atol=0.0000001)

        # divalent, uppercase water model with hyphen
        cation_ff = Aqueous.get_ion(parameter_set="lm", water_model="TIP3P-FB", ion="Zn+2")
        assert isinstance(cation_ff, LammpsData)
        assert np.allclose(cation_ff.force_field["Pair Coeffs"]["coeff2"].item(), 2.495, atol=0.001)
        assert np.allclose(cation_ff.force_field["Pair Coeffs"]["coeff1"].item(), 0.01570749, atol=0.0000001)

        # trivalent, with brackets in ion name
        cation_ff = Aqueous.get_ion(parameter_set="lm", water_model="auto", ion="La[3+]")
        assert isinstance(cation_ff, LammpsData)
        assert np.allclose(cation_ff.force_field["Pair Coeffs"]["coeff2"].item(), 3.056, atol=0.001)
        assert np.allclose(cation_ff.force_field["Pair Coeffs"]["coeff1"].item(), 0.1485017, atol=0.0000001)

        # model auto selection
        cation_ff = Aqueous.get_ion(ion="li+")
        assert isinstance(cation_ff, LammpsData)
        assert np.allclose(cation_ff.force_field["Pair Coeffs"]["coeff2"].item(), 1.409, atol=0.001)
        assert np.allclose(cation_ff.force_field["Pair Coeffs"]["coeff1"].item(), 0.3367344, atol=0.0000001)
        cation_ff = Aqueous.get_ion(parameter_set="jj", ion="li+")
        assert isinstance(cation_ff, LammpsData)
        assert np.allclose(cation_ff.force_field["Pair Coeffs"]["coeff2"].item(), 2.87, atol=0.001)
        assert np.allclose(cation_ff.force_field["Pair Coeffs"]["coeff1"].item(), 0.0005, atol=0.0000001)
        cation_ff = Aqueous.get_ion(water_model="opc3", ion="li+")
        assert isinstance(cation_ff, LammpsData)
        assert np.allclose(cation_ff.force_field["Pair Coeffs"]["coeff2"].item(), 2.3537544133267763, atol=0.001)
        assert np.allclose(cation_ff.force_field["Pair Coeffs"]["coeff1"].item(), 0.0064158, atol=0.0000001)

        # ion not found
        with pytest.raises(ValueError, match="not found in database"):
            cation_ff = Aqueous.get_ion(parameter_set="jj", water_model="opc3", ion="Cu+3")

        # parameter set not found
        with pytest.raises(ValueError, match="No jensen_jorgensen parameters for water model opc3 for ion"):
            cation_ff = Aqueous.get_ion(parameter_set="jj", water_model="opc3", ion="Cu+")

        # water model not found
        with pytest.raises(ValueError, match="No ryan parameters for water model tip8p for ion"):
            cation_ff = Aqueous.get_ion(parameter_set="ryan", water_model="tip8p", ion="Cu+")

        # mixing rule
        cation_ff = Aqueous.get_ion(parameter_set="jc", water_model="spce", ion="li+", mixing_rule="LB")
        assert isinstance(cation_ff, LammpsData)
        assert np.allclose(cation_ff.force_field["Pair Coeffs"]["coeff2"].item(), 1.409, atol=0.001)
        assert np.allclose(cation_ff.force_field["Pair Coeffs"]["coeff1"].item(), 0.3367344, atol=0.0000001)
        cation_ff = Aqueous.get_ion(parameter_set="jj", water_model="tip4p", ion="li+", mixing_rule="arithmetic")
        assert isinstance(cation_ff, LammpsData)
        assert np.allclose(cation_ff.force_field["Pair Coeffs"]["coeff2"].item(), 2.863, atol=0.001)
        assert np.allclose(cation_ff.force_field["Pair Coeffs"]["coeff1"].item(), 0.0005, atol=0.0000001)
        cation_ff = Aqueous.get_ion(
            parameter_set="lm", water_model="tip3pfb", ion="li+", mixing_rule="lorentz-berthelot"
        )
        assert isinstance(cation_ff, LammpsData)
        assert np.allclose(cation_ff.force_field["Pair Coeffs"]["coeff2"].item(), 2.352, atol=0.001)
        assert np.allclose(cation_ff.force_field["Pair Coeffs"]["coeff1"].item(), 0.00633615, atol=0.0000001)
        cation_ff = Aqueous.get_ion(parameter_set="jc", water_model="spce", ion="li+", mixing_rule="geometric")
        assert isinstance(cation_ff, LammpsData)
        assert np.allclose(cation_ff.force_field["Pair Coeffs"]["coeff2"].item(), 1.653, atol=0.001)
        assert np.allclose(cation_ff.force_field["Pair Coeffs"]["coeff1"].item(), 0.3367344, atol=0.0000001)
        cation_ff = Aqueous.get_ion(parameter_set="jc", water_model="tip4pew", ion="na+", mixing_rule="geometric")
        assert isinstance(cation_ff, LammpsData)
        assert np.allclose(cation_ff.force_field["Pair Coeffs"]["coeff2"].item(), 2.260, atol=0.001)
        assert np.allclose(cation_ff.force_field["Pair Coeffs"]["coeff1"].item(), 0.1684375, atol=0.0000001)


if __name__ == "__main__":
    unittest.main()
