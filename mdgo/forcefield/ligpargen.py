
"""
This module implements a core class LigpargenRunner for generating
LAMMPS/GROMACS data files from molecule structure using LigParGen 2.1 
and BOSS 5.0.
"""

import subprocess
import os
from pymatgen.io.lammps.data import LammpsData
from mdgo.util.dict_utils import lmp_mass_to_name

class LigpargenRunner:

    def __init__(
        self,
        structure_name: str,
        structure_dir: str,
        working_dir: str= "boss_files",
        out: str = "lmp",
        charge: int = 0,
        opt: int = 0,
        xyz: bool = False,        
    ):
        """Base constructor."""
        self.structure = structure_dir + "/" + structure_name
        self.name = os.path.splitext(structure_name)[0]
        self.structure_format = os.path.splitext(structure_name)[1][1:]
        print("Input format:", self.structure_format)
        self.structure_dir = structure_dir
        self.work = working_dir
        self.out = out
        self.charge = charge
        self.opt = opt
        self.xyz = xyz
    

    def data_from_structure(self, wait: float = 30):

        try:
            cmd = f"ligpargen -i {self.structure} -n {self.name} -p {self.work} -c {self.charge} -o {self.opt}"
            subprocess.run(cmd, shell=True)
        except subprocess.CalledProcessError as e:
            raise ValueError(f"LigParGen failed with errorcode {e.returncode}  and stderr: {e.stderr}") from e
        
        if self.out == "lmp":
            lmp_name = f"{self.name}.lmp"
            lmp_file = f"{self.structure_dir}/{self.name}.lmp"
            cp_lmp_data = f"cp {self.work}/{self.name}.lammps.lmp {lmp_file}"
            subprocess.run(cp_lmp_data, shell=True)

        if self.xyz:
            lmp_file = f"{self.structure_dir}/{self.name}.lmp"
            data_obj = LammpsData.from_file(lmp_file)
            element_id_dict = lmp_mass_to_name(data_obj.masses)
            coords = data_obj.atoms[["type", "x", "y", "z"]]
            lines = []
            lines.append(str(len(coords.index)))
            lines.append("")
            for _, r in coords.iterrows():
                element_name = element_id_dict.get(int(r["type"]))
                assert element_name is not None
                line = element_name + " " + " ".join(str(r[loc]) for loc in ["x", "y", "z"])
                lines.append(line)

            with open(os.path.join(self.structure_dir, lmp_name + ".xyz"), "w") as xyz_file:
                xyz_file.write("\n".join(lines))
            print(".xyz file saved.")
        
    def data_from_smiles(self, wait: float = 30):
        try:
            cmd = f"ligpargen -s {self.name} -n {self.name} -p {self.work} -c {self.charge} -o {self.opt}"
            subprocess.run(cmd, shell=True)
        except subprocess.CalledProcessError as e:
            raise ValueError(f"LigParGen failed with errorcode {e.returncode}  and stderr: {e.stderr}") from e
        
        if self.out == "lmp":
            lmp_name = f"{self.name}.lmp"
            lmp_file = f"{self.structure_dir}/{self.name}.lmp"
            cp_lmp_data = f"cp {self.work}/{self.name}.lammps.lmp {lmp_file}"
            subprocess.run(cp_lmp_data, shell=True)

        if self.xyz:
            lmp_file = f"{self.structure_dir}/{self.name}.lmp"
            data_obj = LammpsData.from_file(lmp_file)
            element_id_dict = lmp_mass_to_name(data_obj.masses)
            coords = data_obj.atoms[["type", "x", "y", "z"]]
            lines = []
            lines.append(str(len(coords.index)))
            lines.append("")
            for _, r in coords.iterrows():
                element_name = element_id_dict.get(int(r["type"]))
                assert element_name is not None
                line = element_name + " " + " ".join(str(r[loc]) for loc in ["x", "y", "z"])
                lines.append(line)

            with open(os.path.join(self.structure_dir, lmp_name + ".xyz"), "w") as xyz_file:
                xyz_file.write("\n".join(lines))
            print(".xyz file saved.")
