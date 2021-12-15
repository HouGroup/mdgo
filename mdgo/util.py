# coding: utf-8
# Copyright (c) Tingzheng Hou.
# Distributed under the terms of the MIT License.

"""
This module implements utility functions for other modules in the package.
"""


import string
from io import StringIO
import os
import re
import math
import sys
from typing import List, Dict, Union, Tuple, Optional, Any
from typing_extensions import Final
import numpy as np
import pandas as pd

from pymatgen.core import Molecule
from pymatgen.io.lammps.data import CombinedData

from MDAnalysis import Universe
from MDAnalysis.core.groups import Atom, Residue, AtomGroup

from mdgo.volume import molecular_volume

__author__ = "Tingzheng Hou"
__version__ = "1.0"
__maintainer__ = "Tingzheng Hou"
__email__ = "tingzheng_hou@berkeley.edu"
__date__ = "Feb 9, 2021"

MM_of_Elements: Final[Dict[str, float]] = {
    "H": 1.00794,
    "He": 4.002602,
    "Li": 6.941,
    "Be": 9.012182,
    "B": 10.811,
    "C": 12.0107,
    "N": 14.0067,
    "O": 15.9994,
    "F": 18.9984032,
    "Ne": 20.1797,
    "Na": 22.98976928,
    "Mg": 24.305,
    "Al": 26.9815386,
    "Si": 28.0855,
    "P": 30.973762,
    "S": 32.065,
    "Cl": 35.453,
    "Ar": 39.948,
    "K": 39.0983,
    "Ca": 40.078,
    "Sc": 44.955912,
    "Ti": 47.867,
    "V": 50.9415,
    "Cr": 51.9961,
    "Mn": 54.938045,
    "Fe": 55.845,
    "Co": 58.933195,
    "Ni": 58.6934,
    "Cu": 63.546,
    "Zn": 65.409,
    "Ga": 69.723,
    "Ge": 72.64,
    "As": 74.9216,
    "Se": 78.96,
    "Br": 79.904,
    "Kr": 83.798,
    "Rb": 85.4678,
    "Sr": 87.62,
    "Y": 88.90585,
    "Zr": 91.224,
    "Nb": 92.90638,
    "Mo": 95.94,
    "Tc": 98.9063,
    "Ru": 101.07,
    "Rh": 102.9055,
    "Pd": 106.42,
    "Ag": 107.8682,
    "Cd": 112.411,
    "In": 114.818,
    "Sn": 118.71,
    "Sb": 121.760,
    "Te": 127.6,
    "I": 126.90447,
    "Xe": 131.293,
    "Cs": 132.9054519,
    "Ba": 137.327,
    "La": 138.90547,
    "Ce": 140.116,
    "Pr": 140.90465,
    "Nd": 144.242,
    "Pm": 146.9151,
    "Sm": 150.36,
    "Eu": 151.964,
    "Gd": 157.25,
    "Tb": 158.92535,
    "Dy": 162.5,
    "Ho": 164.93032,
    "Er": 167.259,
    "Tm": 168.93421,
    "Yb": 173.04,
    "Lu": 174.967,
    "Hf": 178.49,
    "Ta": 180.9479,
    "W": 183.84,
    "Re": 186.207,
    "Os": 190.23,
    "Ir": 192.217,
    "Pt": 195.084,
    "Au": 196.966569,
    "Hg": 200.59,
    "Tl": 204.3833,
    "Pb": 207.2,
    "Bi": 208.9804,
    "Po": 208.9824,
    "At": 209.9871,
    "Rn": 222.0176,
    "Fr": 223.0197,
    "Ra": 226.0254,
    "Ac": 227.0278,
    "Th": 232.03806,
    "Pa": 231.03588,
    "U": 238.02891,
    "Np": 237.0482,
    "Pu": 244.0642,
    "Am": 243.0614,
    "Cm": 247.0703,
    "Bk": 247.0703,
    "Cf": 251.0796,
    "Es": 252.0829,
    "Fm": 257.0951,
    "Md": 258.0951,
    "No": 259.1009,
    "Lr": 262,
    "Rf": 267,
    "Db": 268,
    "Sg": 271,
    "Bh": 270,
    "Hs": 269,
    "Mt": 278,
    "Ds": 281,
    "Rg": 281,
    "Cn": 285,
    "Nh": 284,
    "Fl": 289,
    "Mc": 289,
    "Lv": 292,
    "Ts": 294,
    "Og": 294,
    "ZERO": 0,
}

SECTION_SORTER: Final[Dict[str, Dict[str, Any]]] = {
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

MOLAR_VOLUME: Final[Dict[str, float]] = {"lipf6": 18, "litfsi": 100}  # empirical value

ALIAS: Final[Dict[str, str]] = {
    "ethylene carbonate": "ec",
    "ec": "ec",
    "propylene carbonate": "pc",
    "pc": "pc",
    "dimethyl carbonate": "dmc",
    "dmc": "dmc",
    "diethyl carbonate": "dec",
    "dec": "dec",
    "ethyl methyl carbonate": "emc",
    "emc": "emc",
    "fluoroethylene carbonate": "fec",
    "fec": "fec",
    "vinyl carbonate": "vc",
    "vinylene carbonate": "vc",
    "vc": "vc",
    "1,3-dioxolane": "dol",
    "dioxolane": "dol",
    "dol": "dol",
    "ethylene glycol monomethyl ether": "egme",
    "2-methoxyethanol": "egme",
    "egme": "egme",
    "dme": "dme",
    "1,2-dimethoxyethane": "dme",
    "glyme": "dme",
    "monoglyme": "dme",
    "2-methoxyethyl ether": "diglyme",
    "diglyme": "diglyme",
    "triglyme": "triglyme",
    "tetraglyme": "tetraglyme",
    "acetonitrile": "acn",
    "acn": "acn",
    "water": "water",
    "h2o": "water",
}

# From PubChem
MOLAR_MASS: Final[Dict[str, float]] = {
    "ec": 88.06,
    "pc": 102.09,
    "dec": 118.13,
    "dmc": 90.08,
    "emc": 104.05,
    "fec": 106.05,
    "vc": 86.05,
    "dol": 74.08,
    "egme": 76.09,
    "dme": 90.12,
    "diglyme": 134.17,
    "triglyme": 178.23,
    "tetraglyme": 222.28,
    "acn": 41.05,
    "water": 18.01528,
}

# from Sigma-Aldrich
DENSITY: Final[Dict[str, float]] = {
    "ec": 1.321,
    "pc": 1.204,
    "dec": 0.975,
    "dmc": 1.069,
    "emc": 1.006,
    "fec": 1.454,  # from qm-ht.com
    "vc": 1.355,
    "dol": 1.06,
    "dme": 0.867,
    "egme": 0.965,
    "diglyme": 0.939,
    "triglyme": 0.986,
    "tetraglyme": 1.009,
    "acn": 0.786,
    "water": 0.99707,
}


def atom_vec(atom1: Atom, atom2: Atom, dimension: np.ndarray) -> np.ndarray:
    """
    Calculate the vector of the positions from atom2 to atom1.

    Args:
        atom1: Atom obj 1.
        atom2: Atom obj 2.
        dimension: box dimension.

    Return:
        The obtained vector
    """
    vec = [0, 0, 0]
    for i in range(3):
        diff = atom1.position[i] - atom2.position[i]
        if diff > dimension[i] / 2:
            vec[i] = diff - dimension[i]
        elif diff < -dimension[i] / 2:
            vec[i] = diff + dimension[i]
        else:
            vec[i] = diff
    return np.array(vec)


def position_vec(
    pos1: Union[List[float], np.ndarray],
    pos2: Union[List[float], np.ndarray],
    dimension: Union[List[float], np.ndarray],
) -> np.ndarray:
    """
    Calculate the vector from pos2 to pos2.

    Args:
        pos1: Array of 3d coordinates 1.
        pos2: Array of 3d coordinates 2.
        dimension: box dimension.

    Return:
        The obtained vector.
    """
    vec: List[Union[int, float, np.floating]] = [0, 0, 0]
    for i in range(3):
        diff = pos1[i] - pos2[i]
        if diff > dimension[i] / 2:
            vec[i] = diff - dimension[i]
        elif diff < -dimension[i] / 2:
            vec[i] = diff + dimension[i]
        else:
            vec[i] = diff
    return np.array(vec)


def angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.floating:
    """
    Calculate the angle between three atoms.

    Args:
        a: Coordinates of atom A.
        b: Coordinates of atom B.
        c: Coordinates of atom C.

    Returns:
        The degree A-B-C.
    """
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle_in_radian = np.arccos(cosine_angle)
    return np.degrees(angle_in_radian)


def mass_to_name(masses: np.ndarray) -> np.ndarray:
    """
    Map atom names to element names.

    Args:
        masses: The masses array of atoms in an ``Universe``.

    Return:
        The element name array.
    """
    names = []
    for mass in masses:
        for item in MM_of_Elements.items():
            if math.isclose(mass, item[1], abs_tol=0.1):
                names.append(item[0])
    assert len(masses) == len(names), "Invalid mass found."
    return np.array(names)


def lmp_mass_to_name(df: pd.DataFrame) -> Dict[int, str]:
    """
    Create a dict for mapping atom type id to element from the mass information.
    Args:
        df: The masses attribute from LammpsData object
    Return:
        The element dict.
    """
    atoms = {}
    for row in df.index:
        for item in MM_of_Elements.items():
            if math.isclose(df["mass"][row], item[1], abs_tol=0.01):
                atoms[int(row)] = item[0]
    return atoms


def assign_name(u: Universe, names: np.ndarray):
    """
    Assign resnames to residues in a MDAnalysis.universe object. The function will not overwrite existing names.

    Args:
        u: The universe object to assign resnames to.
        names: The element name array.
    """
    u.add_TopologyAttr("name", values=names)


def assign_resname(u: Universe, res_dict: Dict[str, str]):
    """
    Assign resnames to residues in a MDAnalysis.universe object. The function will not overwrite existing resnames.

    Args:
        u: The universe object to assign resnames to.
        res_dict: A dictionary of resnames, where each resname is a key
            and the corresponding values are the selection language.
    """
    u.add_TopologyAttr("resname")
    for key, val in res_dict.items():
        res_group = u.select_atoms(val)
        res_names = res_group.residues.resnames
        res_names[res_names == ""] = key
        res_group.residues.resnames = res_names


def res_dict_from_select_dict(u: Universe, select_dict: Dict[str, str]) -> Dict[str, str]:
    """
    Infer res_dict (residue selection) from select_dict (atom selection) in a MDAnalysis.universe object.

    Args:
        u: The universe object to assign resnames to.
        select_dict: A dictionary of atom species, where each atom species name is a key
                and the corresponding values are the selection language.

    return:
        A dictionary of resnames.
    """
    saved_select = []
    res_dict = {}
    for key, val in select_dict.items():
        res_select = "same resid as (" + val + ")"
        res_group = u.select_atoms(res_select)
        if key in ["cation", "anion"] or res_group not in saved_select:
            saved_select.append(res_group)
            res_dict[key] = res_select
    if (
        "cation" in res_dict
        and "anion" in res_dict
        and u.select_atoms(res_dict.get("cation")) == u.select_atoms(res_dict.get("anion"))
    ):
        res_dict.pop("anion")
        res_dict["salt"] = res_dict.pop("cation")
    return res_dict


def res_dict_from_datafile(filename: str) -> Dict[str, str]:
    """
    Infer res_dict (residue selection) from a LAMMPS data file.

    Args:
        filename: Path to the data file. The data file must be generated by a CombinedData object.

    return:
        A dictionary of resnames.
    """
    res_dict = {}
    with open(filename, "r") as f:
        lines = f.readlines()
        if lines[0] == "Generated by pymatgen.io.lammps.data.LammpsData\n" and lines[1].startswith("#"):
            elyte_info = re.findall(r"\w+", lines[1])
            it = iter(elyte_info)
            idx = 1
            for num in it:
                name = next(it)
                if name.isnumeric():
                    frag = int(name)
                    name = next(it)
                    names = [name + c for c in string.ascii_lowercase[0:frag]]
                    start = idx
                    idx += int(num) * frag
                    for i, n in enumerate(names):
                        res_dict[n] = "same mass as resid " + str(start + i)
                else:
                    start = idx
                    idx += int(num)
                    end = idx
                    res_dict[name] = "resid " + str(start) + "-" + str(end - 1)
            return res_dict
        raise ValueError("The LAMMPS data file should be generated by pymatgen.io.lammps.data.")


def res_dict_from_lammpsdata(lammps_data: CombinedData) -> Dict[str, str]:
    """
    Infer res_dict (residue selection) from a LAMMPS data file.

    Args:
        lammps_data: A CombinedData object.

    return:
        A dictionary of resnames.
    """
    assert isinstance(lammps_data, CombinedData)
    idx = 1
    res_dict = {}

    if hasattr(lammps_data, "frags"):
        for name, num, frag in zip(lammps_data.names, lammps_data.nums, lammps_data.frags):
            if frag == 1:
                start = idx
                idx += num
                end = idx
                res_dict[name] = "resid " + str(start) + "-" + str(end - 1)
            else:
                names = [name + c for c in string.ascii_lowercase[0:frag]]
                start = idx
                idx += int(num) * frag
                for i, n in enumerate(names):
                    res_dict[n] = "same mass as resid " + str(start + i)
    else:
        for name, num in zip(lammps_data.names, lammps_data.nums):
            start = idx
            idx += num
            end = idx
            res_dict[name] = "resid " + str(start) + "-" + str(end - 1)
    return res_dict


def select_dict_from_resname(u: Universe) -> Dict[str, str]:
    """
    Infer select_dict (possibly interested atom species selection) from resnames in a MDAnalysis.universe object.
    The resname must be pre-assigned already.

    Args:
        u: The universe object to work with.

    return:
        A dictionary of atom species.
    """
    select_dict: Dict[str, str] = {}
    resnames = np.unique(u.residues.resnames)
    for resname in resnames:
        if resname == "":
            continue
        residue = u.select_atoms("resname " + resname).residues[0]
        if np.isclose(residue.charge, 0, atol=1e-5):  # np.sum(residue.atoms.charges)
            if len(residue.atoms.fragments) == 2:
                for i, frag in enumerate(residue.atoms.fragments):
                    charge = np.sum(frag.charges)
                    if charge > 0.001:
                        extract_atom_from_ion(True, frag, select_dict)
                    elif charge < -0.001:
                        extract_atom_from_ion(False, frag, select_dict)
                    else:
                        extract_atom_from_molecule(resname, frag, select_dict, number=i + 1)
            elif len(residue.atoms.fragments) >= 2:
                cation_number = 1
                anion_number = 1
                molecule_number = 1
                for frag in residue.atoms.fragments:
                    charge = np.sum(frag.charges)
                    if charge > 0.001:
                        extract_atom_from_ion(True, frag, select_dict, cation_number)
                        cation_number += 1
                    elif charge < -0.001:
                        extract_atom_from_ion(False, frag, select_dict, anion_number)
                        anion_number += 1
                    else:
                        extract_atom_from_molecule(resname, frag, select_dict, molecule_number)
                        molecule_number += 1
            else:
                extract_atom_from_molecule(resname, residue, select_dict)
        elif residue.charge > 0:
            extract_atom_from_ion(True, residue, select_dict)
        else:
            extract_atom_from_ion(False, residue, select_dict)
    return select_dict


def extract_atom_from_ion(positive: bool, ion: Union[Residue, AtomGroup], select_dict: Dict[str, str], number: int = 0):
    """
    Assign the most most charged atom and/or one unique atom in the ion into select_dict.

    Args:
        positive: Whether the charge of ion is positive. Otherwise negative. Default to True.
        ion: Residue or AtomGroup
        select_dict: A dictionary of atom species, where each atom species name is a key
            and the corresponding values are the selection language.
        number: The serial number of the ion.
    """
    if positive:
        if number == 0:
            cation_name = "cation"
        else:
            cation_name = "cation_" + str(number)
        if len(ion.atoms.types) == 1:
            select_dict[cation_name] = "type " + ion.atoms.types[0]
        else:
            # The most positively charged atom in the cation
            pos_center = ion.atoms[np.argmax(ion.atoms.charges)]
            unique_types = np.unique(ion.atoms.types, return_counts=True)
            # One unique atom in the cation
            uni_center = unique_types[0][np.argmin(unique_types[1])]
            if pos_center.type == uni_center:
                select_dict[cation_name] = "type " + uni_center
            else:
                select_dict[cation_name + "_" + pos_center.name + pos_center.type] = "type " + pos_center.type
                select_dict[cation_name] = "type " + uni_center
    else:
        if number == 0:
            anion_name = "anion"
        else:
            anion_name = "anion_" + str(number)
        if len(ion.atoms.types) == 1:
            select_dict[anion_name] = "type " + ion.atoms.types[0]
        else:
            # The most negatively charged atom in the anion
            neg_center = ion.atoms[np.argmin(ion.atoms.charges)]
            unique_types = np.unique(ion.atoms.types, return_counts=True)
            # One unique atom in the anion
            uni_center = unique_types[0][np.argmin(unique_types[1])]
            if neg_center.type == uni_center:
                select_dict[anion_name] = "type " + uni_center
            else:
                select_dict[anion_name + "_" + neg_center.name + neg_center.type] = "type " + neg_center.type
                select_dict[anion_name] = "type " + uni_center


def extract_atom_from_molecule(
    resname: str, molecule: Union[Residue, AtomGroup], select_dict: Dict[str, str], number: int = 0
):
    """
    Assign the most negatively charged atom in the molecule into select_dict

    Args:
        resname: The name of the molecule
        molecule: The Residue or AtomGroup obj of the molecule.
        select_dict: A dictionary of atom species, where each atom species name is a key
            and the corresponding values are the selection language.
        number: The serial number of the molecule under the name of resname.
    """
    # neg_center = residue.atoms[np.argmin(residue.atoms.charges)]
    # select_dict[resname + "-" + neg_center.name + neg_center.type] = "type " + neg_center.type
    # pos_center = residue.atoms[np.argmax(residue.atoms.charges)]
    # select_dict[resname + "+" + pos_center.name + pos_center.type] = "type " + pos_center.type

    # The most negatively charged atom in the anion
    if number > 0:
        resname = resname + "_" + str(number)
    neg_center = molecule.atoms[np.argmin(molecule.atoms.charges)]
    select_dict[resname] = "type " + neg_center.type


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
    with open(xyz_dir, "r") as f_xyz:
        molecule = pd.read_table(f_xyz, skiprows=2, delim_whitespace=True, names=["atom", "x", "y", "z"])
        coordinates = molecule[["x", "y", "z"]]
        lo = coordinates.min().min() - 0.5
        hi = coordinates.max().max() + 0.5
    with open(ff_dir, "r") as f:
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


def concentration_matcher(
    concentration: float,
    salt: Union[float, int, str, Molecule],
    solvents: List[Union[str, Dict[str, float]]],
    solv_ratio: List[float],
    num_salt: int = 100,
    mode: str = "v",
    radii_type: str = "Bondi",
) -> Tuple[List, float]:
    """
    Estimate the number of molecules of each species in a box,
    given the salt concentration, salt type, solvent molecular weight,
    solvent density, solvent ratio and total number of salt.
    TODO: Auto box size according to Debye screening length

    Args:
        concentration: Salt concentration in mol/L.
        salt: Four types of input are accepted:
              1. The salt name in string ('lipf6' or 'litfsi')
              2. Salt molar volume in as a float/int (cm^3/mol)
              3. A pymatgen Molecule object of the salt structure
              4. The path to the salt structure xyz file

            Valid names are listed in the MOLAR_VOLUME dictionary at the beginning
            of this file and currently include only 'lipf6' or 'litfsi'

            If a Molecule or structure file is provided, mdgo will estimate
            the molar volume according to the VdW radii of the atoms. The
            specific radii used depend on the value of the 'radii_type' kwarg
            (see below).
        solvents: A list of solvent molecules. A molecule could either be
            a name (e.g. "water" or "ethylene carbonate") or a dict containing
            two keys "mass" and "density" in g/mol and g/mL, respectively.

            Valid names are listed in the ALIAS dictionary at the beginning
            of this file.
        solv_ratio: A list of relative weights or volumes of solvents. Must be the
            same length as solvents. For example, for a 30% / 70% (w/w) mix of
            two solvent, pass [0.3, 0.7] or [30, 70]. The sum of weights / volumes
            does not need to be normalized.
        num_salt: The number of salt in the box.
        mode: Weight mode (Weight/weight/W/w/W./w.) or volume mode
            (Volume/volume/V/v/V./v.) for interpreting the ratio of solvents.
        radii_type: "Bondi", "Lange", or "pymatgen". Bondi and Lange vdW radii
            are compiled in this package for H, B, C, N, O, F, Si, P, S, Cl, Br,
            and I. Choose 'pymatgen' to use the vdW radii from pymatgen.Element,
            which are available for most elements and reflect the latest values in
            the CRC handbook.

    Returns:
        A list of number of molecules in the simulation box, starting with
        the salt and followed by each solvent in 'solvents'.
        The list is followed by a float of the approximate length of one side of the box in Å.

    """
    n_solvent = []
    n = len(solv_ratio)
    if n != len(solvents):
        raise ValueError("solvents and solv_ratio must be the same length!")

    if isinstance(salt, (float, int)):
        salt_molar_volume = salt
    elif isinstance(salt, Molecule):
        salt_molar_volume = molecular_volume(salt, salt.composition.reduced_formula, radii_type=radii_type)
    elif isinstance(salt, str):
        if MOLAR_VOLUME.get(salt.lower()):
            salt_molar_volume = MOLAR_VOLUME.get(salt.lower(), 0)
        else:
            if not os.path.exists(salt):
                print(f"\nError: Input file '{salt}' not found.\n")
                sys.exit(1)
            name = os.path.splitext(os.path.split(salt)[-1])[0]
            ext = os.path.splitext(os.path.split(salt)[-1])[1]
            if not ext == ".xyz":
                print("Error: Wrong file format, please use a .xyz file.\n")
                sys.exit(1)
            salt_molar_volume = molecular_volume(salt, name, radii_type=radii_type)
    else:
        raise ValueError("Invalid salt type! Salt must be a number, string, or Molecule.")

    solv_mass = []
    solv_density = []
    for solv in solvents:
        if isinstance(solv, dict):
            solv_mass.append(solv.get("mass"))
            solv_density.append(solv.get("density"))
        else:
            solv_mass.append(MOLAR_MASS[ALIAS[solv.lower()]])
            solv_density.append(DENSITY[ALIAS[solv.lower()]])
    if mode.lower().startswith("v"):
        for i in range(n):
            n_solvent.append(solv_ratio[i] * solv_density[i] / solv_mass[i])  # type: ignore
        v_solv = sum(solv_ratio)
        n_salt = v_solv / (1000 / concentration - salt_molar_volume)
        n_all = [int(m / n_salt * num_salt) for m in n_solvent]
        n_all.insert(0, num_salt)
        volume = ((v_solv + salt_molar_volume * n_salt) / n_salt * num_salt) / 6.022e23
        return n_all, volume ** (1 / 3) * 1e8
    if mode.lower().startswith("w"):
        for i in range(n):
            n_solvent.append(solv_ratio[i] / solv_mass[i])  # type: ignore
        v_solv = np.divide(solv_ratio, solv_density).sum()
        n_salt = v_solv / (1000 / concentration - salt_molar_volume)
        n_all = [int(m / n_salt * num_salt) for m in n_solvent]
        n_all.insert(0, num_salt)
        volume = ((v_solv + salt_molar_volume * n_salt) / n_salt * num_salt) / 6.022e23
        return n_all, volume ** (1 / 3) * 1e8
    mode = input("Volume or weight ratio? (w or v): ")
    return concentration_matcher(concentration, salt_molar_volume, solvents, solv_ratio, num_salt=num_salt, mode=mode)


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
    with open(sdf_file, "r") as inp:
        sdf_lines = inp.readlines()
        sdf = list(map(str.strip, sdf_lines))
    if pubchem:
        title = "cid_"
    else:
        title = ""
    pdb_atoms: List[Dict[str, Any]] = []
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
                "altLoc": str(""),
                "chainID": str(""),
                "iCode": str(""),
                "element": str(line_split[3]),
                "charge": str(""),
                "segment": str(""),
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
    with open(pdb_file, "wt") as outp:
        if write_title:
            outp.write(f"TITLE     {title:70s}\n")
        if version:
            outp.write("REMARK   4      COMPLIES WITH FORMAT V. 3.3, 21-NOV-2012\n")
        if credit:
            outp.write("REMARK 888\n" "REMARK 888 WRITTEN BY MDGO (CREATED BY TINGZHENG HOU)\n")
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


def strip_zeros(items: Union[List[Union[str, float, int]], str]) -> Optional[List[int]]:
    """
    Strip the trailing zeros of a sequence.

    Args:
        items: The sequence.

    Return:
        A new list of numbers.
    """
    new_items = [int(i) for i in items]
    while new_items[-1] == 0:
        new_items.pop()
    while new_items[0] == 0:
        new_items.pop(0)
    if len(new_items) == 0:
        return None
    return new_items


if __name__ == "__main__":
    """
    litfsi = Molecule.from_file(
        "/Users/th/Downloads/package/packmol-17.163/LiTFSI.xyz"
    )
    mols, box_len = concentration_matcher(1.083,
                                          "litfsi",
                                          ["ec", "emc"],
                                          [0.3, 0.7],
                                          num_salt=166,
                                          mode="w")
    print(mols)
    print(box_len)
    """
    sdf_to_pdb(
        "/Users/th/Downloads/test_mdgo/EC_7303.sdf",
        "/Users/th/Downloads/test_mdgo/test_util.pdb",
    )
