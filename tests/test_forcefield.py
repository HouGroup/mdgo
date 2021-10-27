import os
import sys
import tempfile
import unittest
from io import StringIO

import pytest

from mdgo.forcefield import *

test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_files")


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
        cation_ff = Aqueous.get_ion(parameter_set=None, water_model="opc3", ion="li+")
        assert isinstance(cation_ff, LammpsData)
        assert np.allclose(cation_ff.force_field["Pair Coeffs"]["coeff2"].item(), 2.354, atol=0.001)
        assert np.allclose(cation_ff.force_field["Pair Coeffs"]["coeff1"].item(), 0.0064158, atol=0.0000001)

        # Ion object input, all lowercase
        li = Ion.from_formula('Li+')
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
        cation_ff = Aqueous.get_ion(parameter_set="lm", water_model="tip4p-fb", ion="La[3+]")
        assert isinstance(cation_ff, LammpsData)
        assert np.allclose(cation_ff.force_field["Pair Coeffs"]["coeff2"].item(), 3.056, atol=0.001)
        assert np.allclose(cation_ff.force_field["Pair Coeffs"]["coeff1"].item(), 0.1485017, atol=0.0000001)

        # ion not found
        with pytest.raises(ValueError, match="not found in database"):
            cation_ff = Aqueous.get_ion(parameter_set="jj", water_model="opc3", ion="Cu+3")

        # parameter set not found
        with pytest.raises(ValueError, match="No jensen_jorgensen parameters for water model opc3 for ion"):
            cation_ff = Aqueous.get_ion(parameter_set="jj", water_model="opc3", ion="Cu+")

        # water model not found
        with pytest.raises(ValueError, match="No ryan parameters for water model tip8p for ion"):
            cation_ff = Aqueous.get_ion(parameter_set="ryan", water_model="tip8p", ion="Cu+")


if __name__ == "__main__":
    unittest.main()
