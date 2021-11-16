import os
import tempfile
from pathlib import Path
from subprocess import TimeoutExpired

import pytest
import numpy as np
from pymatgen.core import Molecule

from mdgo.mdgopackmol import PackmolWrapper

test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_files")


@pytest.fixture
def ethanol():
    """
    Returns a Molecule of ethanol
    """
    ethanol_coords = [
        [0.00720, -0.56870, 0.00000],
        [-1.28540, 0.24990, 0.00000],
        [1.13040, 0.31470, 0.00000],
        [0.03920, -1.19720, 0.89000],
        [0.03920, -1.19720, -0.89000],
        [-1.31750, 0.87840, 0.89000],
        [-1.31750, 0.87840, -0.89000],
        [-2.14220, -0.42390, -0.00000],
        [1.98570, -0.13650, -0.00000],
    ]
    ethanol_atoms = ["C", "C", "O", "H", "H", "H", "H", "H", "H"]

    return Molecule(ethanol_atoms, ethanol_coords)


@pytest.fixture
def water():
    """
    Returns a Molecule of water
    """
    water_coords = [
        [9.626, 6.787, 12.673],
        [9.626, 8.420, 12.673],
        [10.203, 7.604, 12.673],
    ]
    water_atoms = ["H", "H", "O"]

    return Molecule(water_atoms, water_coords)


class TestPackmolWrapper:
    def test_packmol_with_molecule(self, water, ethanol):
        """
        Test coords input as Molecule
        """
        with tempfile.TemporaryDirectory() as scratch_dir:
            pw = PackmolWrapper(
                scratch_dir,
                molecules=[
                    {"name": "water", "number": 10, "coords": water},
                    {"name": "ethanol", "number": 20, "coords": ethanol},
                ],
            )
            pw.make_packmol_input()
            pw.run_packmol()
            assert os.path.exists(os.path.join(scratch_dir, "packmol_out.xyz"))
            out = Molecule.from_file(os.path.join(scratch_dir, "packmol_out.xyz"))
            assert out.composition.num_atoms == 10 * 3 + 20 * 9

    def test_packmol_with_str(self):
        """
        Test coords input as strings
        """
        with tempfile.TemporaryDirectory() as scratch_dir:
            pw = PackmolWrapper(
                scratch_dir,
                molecules=[
                    {"name": "EMC", "number": 10, "coords": os.path.join(test_dir, "subdir with spaces", "EMC.xyz")},
                    {"name": "LiTFSi", "number": 20, "coords": os.path.join(test_dir, "LiTFSi.xyz")},
                ],
            )
            pw.make_packmol_input()
            pw.run_packmol()
            assert os.path.exists(os.path.join(scratch_dir, "packmol_out.xyz"))
            out = Molecule.from_file(os.path.join(scratch_dir, "packmol_out.xyz"))
            assert out.composition.num_atoms == 10 * 15 + 20 * 16

    def test_packmol_with_path(self):
        """
        Test coords input as Path. Use a subdirectory with spaces.
        """
        p1 = Path(os.path.join(test_dir, "subdir with spaces", "EMC.xyz"))
        p2 = Path(os.path.join(test_dir, "LiTFSi.xyz"))
        with tempfile.TemporaryDirectory() as scratch_dir:
            pw = PackmolWrapper(
                scratch_dir,
                molecules=[
                    {"name": "EMC", "number": 10, "coords": p1},
                    {"name": "LiTFSi", "number": 20, "coords": p2},
                ],
            )
            pw.make_packmol_input()
            pw.run_packmol()
            assert os.path.exists(os.path.join(scratch_dir, "packmol_out.xyz"))
            out = Molecule.from_file(os.path.join(scratch_dir, "packmol_out.xyz"))
            assert out.composition.num_atoms == 10 * 15 + 20 * 16

    def test_control_params(self, water, ethanol):
        """
        Check that custom control_params work and that ValueError
        is raised when 'ERROR' appears in stdout (even if return code is 0)
        """
        with tempfile.TemporaryDirectory() as scratch_dir:
            pw = PackmolWrapper(
                scratch_dir,
                molecules=[
                    {"name": "water", "number": 1000, "coords": water},
                    {"name": "ethanol", "number": 2000, "coords": ethanol},
                ],
                control_params={"maxit": 0, "nloop": 0},
            )
            pw.make_packmol_input()
            with open(os.path.join(scratch_dir, "packmol.inp"), "r") as f:
                input_string = f.read()
                assert "maxit 0" in input_string
                assert "nloop 0" in input_string
            with pytest.raises(ValueError):
                pw.run_packmol()

    def test_timeout(self, water, ethanol):
        """
        Check that the timeout works
        """
        with tempfile.TemporaryDirectory() as scratch_dir:
            pw = PackmolWrapper(
                scratch_dir,
                molecules=[
                    {"name": "water", "number": 1000, "coords": water},
                    {"name": "ethanol", "number": 2000, "coords": ethanol},
                ],
            )
            pw.make_packmol_input()
            with pytest.raises(TimeoutExpired):
                pw.run_packmol(1)

    def test_no_return_and_box(self, water, ethanol):
        """
        Make sure the code raises an error if packmol doesn't
        exit cleanly. Also verify the box arg works properly.
        """
        with tempfile.TemporaryDirectory() as scratch_dir:
            pw = PackmolWrapper(
                scratch_dir,
                molecules=[
                    {"name": "water", "number": 1000, "coords": water},
                    {"name": "ethanol", "number": 2000, "coords": ethanol},
                ],
                box=[0, 0, 0, 2, 2, 2],
            )
            pw.make_packmol_input()
            with open(os.path.join(scratch_dir, "packmol.inp"), "r") as f:
                input_string = f.read()
                assert "inside box 0 0 0 2 2 2" in input_string
            with pytest.raises(ValueError):
                pw.run_packmol()

    def test_random_seed(self, water, ethanol):
        """
        Confirm that seed = -1 generates random structures
        while seed = 1 is deterministic
        """
        # deterministic output
        with tempfile.TemporaryDirectory() as scratch_dir:
            pw = PackmolWrapper(
                scratch_dir,
                molecules=[
                    {"name": "water", "number": 10, "coords": water},
                    {"name": "ethanol", "number": 20, "coords": ethanol},
                ],
                seed=1,
                inputfile="input.in",
                outputfile="output.xyz",
            )
            pw.make_packmol_input()
            pw.run_packmol()
            out1 = Molecule.from_file(os.path.join(scratch_dir, "output.xyz"))
            pw.run_packmol()
            out2 = Molecule.from_file(os.path.join(scratch_dir, "output.xyz"))
            assert np.array_equal(out1.cart_coords, out2.cart_coords)

        # randomly generated structures
        with tempfile.TemporaryDirectory() as scratch_dir:
            pw = PackmolWrapper(
                scratch_dir,
                molecules=[
                    {"name": "water", "number": 10, "coords": water},
                    {"name": "ethanol", "number": 20, "coords": ethanol},
                ],
                seed=-1,
                inputfile="input.in",
                outputfile="output.xyz",
            )
            pw.make_packmol_input()
            pw.run_packmol()
            out1 = Molecule.from_file(os.path.join(scratch_dir, "output.xyz"))
            pw.run_packmol()
            out2 = Molecule.from_file(os.path.join(scratch_dir, "output.xyz"))
            assert not np.array_equal(out1.cart_coords, out2.cart_coords)

    def test_arbitrary_filenames(self, water, ethanol):
        """
        Make sure custom input and output filenames work.
        Use a subdirectory with spaces.
        """
        with tempfile.TemporaryDirectory() as scratch_dir:
            os.mkdir(os.path.join(scratch_dir, "subdirectory with spaces"))
            pw = PackmolWrapper(
                os.path.join(scratch_dir, "subdirectory with spaces"),
                molecules=[
                    {"name": "water", "number": 10, "coords": water},
                    {"name": "ethanol", "number": 20, "coords": ethanol},
                ],
                inputfile="input.in",
                outputfile="output.xyz",
            )
            pw.make_packmol_input()
            assert os.path.exists(os.path.join(scratch_dir, "subdirectory with spaces", "input.in"))
            pw.run_packmol()
            assert os.path.exists(os.path.join(scratch_dir, "subdirectory with spaces", "output.xyz"))
            out = Molecule.from_file(os.path.join(scratch_dir, "subdirectory with spaces", "output.xyz"))
            assert out.composition.num_atoms == 10 * 3 + 20 * 9
