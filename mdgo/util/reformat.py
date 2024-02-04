# Copyright (c) Tingzheng Hou.
# Distributed under the terms of the MIT License.

"""
Utilities for converting data file formats.
"""

from __future__ import annotations

import re
from io import StringIO
from typing import Any, Final

import pandas as pd

from mdgo.util.dict_utils import MM_of_Elements

from . import __author__

SECTION_SORTER: Final[dict[str, dict[str, Any]]] = {
    "atoms": {
        "in_kw": None,
        "in_header": ["atom", "charge", "sigma", "epsilon"],
        "sec_number": None,
        "desired_split": None,
        "desired_cols": None,
        "out_kw": None,
        "ff_header": ["epsilon", "sigma"],
        "topo_header": ["mol-id", "type", "charge", "x", "y", "z"],
    },
    "bonds": {
        "in_kw": "Stretch",
        "in_header": ["atom1", "atom2", "k", "r0"],
        "sec_number": 5,
        "desired_split": 2,
        "desired_cols": 4,
        "out_kw": ["Bond Coeffs", "Bonds"],
        "ff_header": ["k", "r0"],
        "topo_header": ["type", "atom1", "atom2"],
    },
    "angles": {
        "in_kw": "Bending",
        "in_header": ["atom1", "atom2", "atom3", "k", "theta0"],
        "sec_number": 6,
        "desired_split": 1,
        "desired_cols": 5,
        "out_kw": ["Angle Coeffs", "Angles"],
        "ff_header": ["k", "theta0"],
        "topo_header": ["type", "atom1", "atom2", "atom3"],
    },
    "dihedrals": {
        "in_kw": "proper Torsion",
        "in_header": ["atom1", "atom2", "atom3", "atom4", "v1", "v2", "v3", "v4"],
        "sec_number": 7,
        "desired_split": 1,
        "desired_cols": 8,
        "out_kw": ["Dihedral Coeffs", "Dihedrals"],
        "ff_header": ["v1", "v2", "v3", "v4"],
        "topo_header": ["type", "atom1", "atom2", "atom3", "atom4"],
    },
    "impropers": {
        "in_kw": "improper Torsion",
        "in_header": ["atom1", "atom2", "atom3", "atom4", "v2"],
        "sec_number": 8,
        "desired_split": 1,
        "desired_cols": 5,
        "out_kw": ["Improper Coeffs", "Impropers"],
        "ff_header": ["v1", "v2", "v3"],
        "topo_header": ["type", "atom1", "atom2", "atom3", "atom4"],
    },
}

BOX: Final[
    str
] = """{0:6f} {1:6f} xlo xhi
{0:6f} {1:6f} ylo yhi
{0:6f} {1:6f} zlo zhi"""


def ff_parser(ff_dir: str, xyz_dir: str) -> str:
    """
    A parser to convert a force field field from Maestro format
    to LAMMPS data format.

    Args:
        ff_dir: The path to the Maestro force field file.
        xyz_dir: The path to the xyz structure file.

    Return:
        The output LAMMPS data string.
    """
    with open(xyz_dir) as f_xyz:
        molecule = pd.read_table(f_xyz, skiprows=2, delim_whitespace=True, names=["atom", "x", "y", "z"])
        coordinates = molecule[["x", "y", "z"]]
        lo = coordinates.min().min() - 0.5
        hi = coordinates.max().max() + 0.5
    with open(ff_dir) as f:
        lines_org = f.read()
        lines = lines_org.split("\n\n")
        atoms = "\n".join(lines[4].split("\n", 4)[4].split("\n")[:-1])
        dfs = {}
        dfs["atoms"] = pd.read_csv(
            StringIO(atoms),
            names=SECTION_SORTER.get("atoms", {}).get("in_header"),
            delim_whitespace=True,
            usecols=[0, 4, 5, 6],
        )
        dfs["atoms"] = pd.concat([dfs["atoms"], coordinates], axis=1)
        dfs["atoms"].index += 1
        dfs["atoms"].index.name = "type"
        dfs["atoms"] = dfs["atoms"].reset_index()
        dfs["atoms"].index += 1
        types = dfs["atoms"].copy().reset_index().set_index("atom")["type"]
        replace_dict = {
            "atom1": dict(types),
            "atom2": dict(types),
            "atom3": dict(types),
            "atom4": dict(types),
        }
        counts = {}
        counts["atoms"] = len(dfs["atoms"].index)
        mass_list = []
        for index, row in dfs["atoms"].iterrows():
            mass_list.append(MM_of_Elements.get(re.split(r"(\d+)", row["atom"])[0]))
        mass_df = pd.DataFrame(mass_list)
        mass_df.index += 1
        mass_string = mass_df.to_string(header=False, index_names=False, float_format="{:.3f}".format)
        masses = ["Masses", mass_string]
        ff = ["Pair Coeffs"]
        dfs["atoms"]["mol-id"] = 1
        atom_ff_string = dfs["atoms"][SECTION_SORTER["atoms"]["ff_header"]].to_string(header=False, index_names=False)
        ff.append(atom_ff_string)
        topo = ["Atoms"]
        atom_topo_string = dfs["atoms"][SECTION_SORTER["atoms"]["topo_header"]].to_string(
            header=False, index_names=False
        )
        topo.append(atom_topo_string)
        for section in list(SECTION_SORTER.keys())[1:]:
            if SECTION_SORTER[section]["in_kw"] in lines_org:
                a, b, c, d = (
                    SECTION_SORTER[section]["sec_number"],
                    SECTION_SORTER[section]["desired_split"],
                    SECTION_SORTER[section]["desired_cols"],
                    SECTION_SORTER[section]["in_header"],
                )
                section_str = lines[a].split("\n", b)[b]
                dfs[section] = pd.read_csv(
                    StringIO(section_str),
                    names=d,
                    delim_whitespace=True,
                    usecols=list(range(c)),
                )

                dfs[section].index += 1
                dfs[section].index.name = "type"
                dfs[section] = dfs[section].replace(replace_dict)
                dfs[section] = dfs[section].reset_index()
                dfs[section].index += 1
                if section == "impropers":
                    dfs[section]["v1"] = dfs[section]["v2"] / 2
                    dfs[section]["v2"] = -1
                    dfs[section]["v3"] = 2
                ff_string = dfs[section][SECTION_SORTER[section]["ff_header"]].to_string(
                    header=False, index_names=False
                )
                ff.append(SECTION_SORTER[section]["out_kw"][0])
                ff.append(ff_string)
                topo_string = dfs[section][SECTION_SORTER[section]["topo_header"]].to_string(
                    header=False, index_names=False
                )
                topo.append(SECTION_SORTER[section]["out_kw"][1])
                topo.append(topo_string)
                counts[section] = len(dfs[section].index)
        max_stats = len(str(max(list(counts.values()))))
        stats_template = "{:>" + str(max_stats) + "}  {}"
        count_lines = [stats_template.format(v, k) for k, v in counts.items()]
        type_lines = [stats_template.format(v, k[:-1] + " types") for k, v in counts.items()]
        stats = "\n".join(count_lines + [""] + type_lines)
        header = [
            f"LAMMPS data file created by mdgo (by {__author__})\n"
            "# OPLS force field: harmonic, harmonic, opls, cvff",
            stats,
            BOX.format(lo, hi),
        ]
        data_string = "\n\n".join(header + masses + ff + topo) + "\n"
        return data_string


def sdf_to_pdb(
    sdf_file: str,
    pdb_file: str,
    write_title: bool = True,
    version: bool = True,
    credit: bool = True,
    pubchem: bool = True,
):
    """
    Convert SDF file to PDB file.

    Args:
        sdf_file: Path to the input sdf file.
        pdb_file: Path to the output pdb file.
        write_title: Whether to write title in the pdb file. Default to True.
        version: Whether to version line (remark 4) line in the pdb file. Default to True.
        credit: Whether to credit line (remark 888) in the pdb file. Default to True.
        pubchem: Whether the sdf is downloaded from PubChem. Default to True.
    """
    # parse sdf file file
    with open(sdf_file) as inp:
        sdf_lines = inp.readlines()
        sdf = list(map(str.strip, sdf_lines))
    if pubchem:
        title = "cid_"
    else:
        title = ""
    pdb_atoms: list[dict[str, Any]] = []
    # create pdb list of dictionaries
    atoms = 0
    bonds = 0
    atom1s = []
    atom2s = []
    orders = []
    for i, line in enumerate(sdf):
        if i == 0:
            title += line.strip() + " "
        elif i in [1, 2]:
            pass
        elif i == 3:
            line_split = line.split()
            atoms = int(line_split[0])
            bonds = int(line_split[1])
        elif line.startswith("M  END"):
            break
        elif i in list(range(4, 4 + atoms)):
            line_split = line.split()
            newline = {
                "ATOM": "HETATM",
                "serial": int(i - 3),
                "name": str(line_split[3]),
                "resName": "UNK",
                "resSeq": 900,
                "x": float(line_split[0]),
                "y": float(line_split[1]),
                "z": float(line_split[2]),
                "occupancy": 1.00,
                "tempFactor": 0.00,
                "altLoc": "",
                "chainID": "",
                "iCode": "",
                "element": str(line_split[3]),
                "charge": "",
                "segment": "",
            }
            pdb_atoms.append(newline)
        elif i in list(range(4 + atoms, 4 + atoms + bonds)):
            line_split = line.split()
            atom1 = int(line_split[0])
            atom2 = int(line_split[1])
            order = int(line_split[2])
            atom1s.append(atom1)
            atom2s.append(atom2)
            while order > 1:
                orders.append([atom1, atom2])
                orders.append([atom2, atom1])
                order -= 1
        else:
            pass

    # write pdb file
    with open(pdb_file, "w") as outp:
        if write_title:
            outp.write(f"TITLE     {title:70s}\n")
        if version:
            outp.write("REMARK   4      COMPLIES WITH FORMAT V. 3.3, 21-NOV-2012\n")
        if credit:
            outp.write("REMARK 888\nREMARK 888 WRITTEN BY MDGO (CREATED BY TINGZHENG HOU)\n")
        for n in range(atoms):
            line_dict = pdb_atoms[n].copy()
            if len(line_dict["name"]) == 3:
                line_dict["name"] = " " + line_dict["name"]
            # format pdb
            formatted_line = (
                "{:<6s}{:>5d} {:^4s}{:1s}{:>3s} {:1s}{:>4.4}{:1s}   "  # pylint: disable=C0209
                "{:>8.3f}{:>8.3f}{:>8.3f}{:>6.2f}{:>6.2f}      "
                "{:<4s}{:>2s}{:<2s}"
            ).format(
                line_dict["ATOM"],
                line_dict["serial"],
                line_dict["name"],
                line_dict["altLoc"],
                line_dict["resName"],
                line_dict["chainID"],
                str(line_dict["resSeq"]),
                line_dict["iCode"],
                line_dict["x"],
                line_dict["y"],
                line_dict["z"],
                line_dict["occupancy"],
                line_dict["tempFactor"],
                line_dict["segment"],
                line_dict["element"],
                line_dict["charge"],
            )
            # write
            outp.write(formatted_line + "\n")

        bond_lines = [[i] for i in range(atoms + 1)]
        for i, atom in enumerate(atom1s):
            bond_lines[atom].append(atom2s[i])
        for i, atom in enumerate(atom2s):
            bond_lines[atom].append(atom1s[i])
        for i, odr in enumerate(orders):
            for j, ln in enumerate(bond_lines):
                if ln[0] == odr[0]:
                    bond_lines.insert(j + 1, odr)
                    break
        for i in range(1, len(bond_lines)):
            bond_lines[i][1:] = sorted(bond_lines[i][1:])
        for i in range(1, len(bond_lines)):
            outp.write("CONECT" + "".join(f"{num:>5d}" for num in bond_lines[i]) + "\n")
        outp.write("END\n")  # final 'END'
