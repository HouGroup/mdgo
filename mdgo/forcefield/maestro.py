# Copyright (c) Tingzheng Hou.
# Distributed under the terms of the MIT License.

"""
This module implements a core class MaestroRunner for generating
LAMMPS/GROMACS data files from molecule structure using Maestro.

For using the MaestroRunner class:

  * Download a free Maestro via https://www.schrodinger.com/freemaestro

  * Install the package and set the environment variable $SCHRODINGER
    (e.g. 'export SCHRODINGER=/opt/schrodinger/suites2021-4', please
    check https://www.schrodinger.com/kb/446296 or
    https://www.schrodinger.com/kb/1842 for details.

"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from string import Template
from typing import Final

from mdgo.util.reformat import ff_parser

MAESTRO: Final[str] = "$SCHRODINGER/maestro -console -nosplash"
FFLD: Final[str] = "$SCHRODINGER/utilities/ffld_server -imae {} -version 14 -print_parameters -out_file {}"
MODULE_DIR: Final[str] = os.path.dirname(os.path.abspath(__file__))


class MaestroRunner:
    """
    Wrapper for the Maestro software that can be used to generate the OPLS_2005
    force field parameter for a molecule.

    Args:
        structure_dir: Path to the structure file.
            Supported input format please check
            https://www.schrodinger.com/kb/1278
        working_dir: Directory for writing intermediate
            and final output.
        out: Force field output form. Default to "lmp",
            the data file for LAMMPS. Other supported formats
            are under development.
        cmd_template: String template for input script
            with placeholders. Default to None, i.e., using
            the default template.
        assign_bond: Whether to assign bond to the input
            structure. Default to None.

    Supported input format please check https://www.schrodinger.com/kb/1278

    The OPLS_2005 parameters are described in

    Banks, J.L.; Beard, H.S.; Cao, Y.; Cho, A.E.; Damm, W.; Farid, R.;
    Felts, A.K.; Halgren, T.A.; Mainz, D.T.; Maple, J.R.; Murphy, R.;
    Philipp, D.M.; Repasky, M.P.; Zhang, L.Y.; Berne, B.J.; Friesner, R.A.;
    Gallicchio, E.; Levy. R.M. Integrated Modeling Program, Applied Chemical
    Theory (IMPACT). J. Comp. Chem. 2005, 26, 1752.

    The OPLS_2005 parameters are located in

    $SCHRODINGER/mmshare-vversion/data/f14/

    Examples:
        >>> mr = MaestroRunner('/path/to/structure', '/path/to/working/dir')
        >>> mr.get_mae()
        >>> mr.get_ff()
    """

    template_assignbond = os.path.join(MODULE_DIR, "..", "templates", "mae_cmd_assignbond.txt")

    template_noassignbond = os.path.join(MODULE_DIR, "..", "templates", "mae_cmd_noassignbond.txt")

    def __init__(
        self,
        structure_dir: str,
        working_dir: str,
        out: str = "lmp",
        cmd_template: str | None = None,
        assign_bond: bool = False,
    ):
        """Base constructor."""
        self.structure = structure_dir
        self.out = out
        self.structure_format = os.path.splitext(self.structure)[1][1:]
        self.name = os.path.splitext(os.path.split(self.structure)[-1])[0]
        print("Input format:", self.structure_format)
        self.work = working_dir
        self.cmd = os.path.join(self.work, "maetro_script.cmd")
        self.mae = os.path.join(self.work, self.name)
        self.ff = os.path.join(self.work, self.name + ".out")
        self.xyz = os.path.join(self.work, self.name + ".xyz")
        if cmd_template:
            self.cmd_template = cmd_template
        else:
            if assign_bond:
                with open(self.template_assignbond) as f:
                    cmd_template = f.read()
                self.cmd_template = cmd_template
            else:
                with open(self.template_noassignbond) as f:
                    cmd_template = f.read()
                self.cmd_template = cmd_template

    def get_mae(self, wait: float = 30):
        """Write a Maestro command script and execute it to generate a
        maestro file containing all the info needed.

        Args:
            wait: The time waiting for Maestro execution in seconds. Default to 30.
        """
        with open(self.cmd, "w") as f:
            cmd_template = Template(self.cmd_template)
            cmd_script = cmd_template.substitute(file=self.structure, mae=self.mae, xyz=self.xyz)
            f.write(cmd_script)
        try:
            p = subprocess.Popen(  # pylint: disable=consider-using-with
                f"{MAESTRO} -c {self.cmd}",
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
        except subprocess.CalledProcessError as e:
            raise ValueError(f"Maestro failed with errorcode {e.returncode}  and stderr: {e.stderr}") from e

        counter = 0
        while not os.path.isfile(self.mae + ".mae"):
            time.sleep(1)
            counter += 1
            if counter > wait:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                raise TimeoutError(f"Failed to generate Maestro file in {wait} secs!")
        print("Maestro file generated!")
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)

    def get_ff(self):
        """Read the Maestro file and save the force field as LAMMPS data file."""
        try:
            subprocess.run(
                FFLD.format(self.mae + ".mae", self.ff),
                check=True,
                shell=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as e:
            raise ValueError(f"Maestro failed with errorcode {e.returncode} and stderr: {e.stderr}") from e
        print("Maestro force field file generated.")
        if self.out:
            if self.out == "lmp":
                with open(os.path.join(self.work, self.name + "." + self.out), "w") as f:
                    f.write(ff_parser(self.ff, self.xyz))
                print("LAMMPS data file generated.")
            else:
                print("Output format not supported, ff format not converted.")
