# coding: utf-8
# Copyright (c) Tingzheng Hou.
# Distributed under the terms of the MIT License.

import numpy as np
from io import StringIO
import re
import pandas as pd
import math

__author__ = "Tingzheng Hou"
__version__ = "1.0"
__maintainer__ = "Tingzheng Hou"
__email__ = "tingzheng_hou@berkeley.edu"
__date__ = "Feb 9, 2021"

MM_of_Elements = {'H': 1.00794, 'He': 4.002602, 'Li': 6.941, 'Be': 9.012182,
                  'B': 10.811, 'C': 12.0107, 'N': 14.0067, 'O': 15.9994,
                  'F': 18.9984032, 'Ne': 20.1797, 'Na': 22.98976928,
                  'Mg': 24.305, 'Al': 26.9815386, 'Si': 28.0855, 'P': 30.973762,
                  'S': 32.065, 'Cl': 35.453, 'Ar': 39.948, 'K': 39.0983,
                  'Ca': 40.078, 'Sc': 44.955912, 'Ti': 47.867, 'V': 50.9415,
                  'Cr': 51.9961, 'Mn': 54.938045, 'Fe': 55.845, 'Co': 58.933195,
                  'Ni': 58.6934, 'Cu': 63.546, 'Zn': 65.409, 'Ga': 69.723,
                  'Ge': 72.64, 'As': 74.9216, 'Se': 78.96, 'Br': 79.904,
                  'Kr': 83.798, 'Rb': 85.4678, 'Sr': 87.62, 'Y': 88.90585,
                  'Zr': 91.224, 'Nb': 92.90638, 'Mo': 95.94, 'Tc': 98.9063,
                  'Ru': 101.07, 'Rh': 102.9055, 'Pd': 106.42, 'Ag': 107.8682,
                  'Cd': 112.411, 'In': 114.818, 'Sn': 118.71, 'Sb': 121.760,
                  'Te': 127.6, 'I': 126.90447, 'Xe': 131.293, 'Cs': 132.9054519,
                  'Ba': 137.327, 'La': 138.90547, 'Ce': 140.116,
                  'Pr': 140.90465, 'Nd': 144.242, 'Pm': 146.9151, 'Sm': 150.36,
                  'Eu': 151.964, 'Gd': 157.25, 'Tb': 158.92535, 'Dy': 162.5,
                  'Ho': 164.93032, 'Er': 167.259, 'Tm': 168.93421, 'Yb': 173.04,
                  'Lu': 174.967, 'Hf': 178.49, 'Ta': 180.9479, 'W': 183.84,
                  'Re': 186.207, 'Os': 190.23, 'Ir': 192.217, 'Pt': 195.084,
                  'Au': 196.966569, 'Hg': 200.59, 'Tl': 204.3833, 'Pb': 207.2,
                  'Bi': 208.9804, 'Po': 208.9824, 'At': 209.9871,
                  'Rn': 222.0176, 'Fr': 223.0197, 'Ra': 226.0254,
                  'Ac': 227.0278, 'Th': 232.03806, 'Pa': 231.03588,
                  'U': 238.02891, 'Np': 237.0482, 'Pu': 244.0642,
                  'Am': 243.0614, 'Cm': 247.0703, 'Bk': 247.0703,
                  'Cf': 251.0796, 'Es': 252.0829, 'Fm': 257.0951,
                  'Md': 258.0951, 'No': 259.1009, 'Lr': 262, 'Rf': 267,
                  'Db': 268, 'Sg': 271, 'Bh': 270, 'Hs': 269, 'Mt': 278,
                  'Ds': 281, 'Rg': 281, 'Cn': 285, 'Nh': 284, 'Fl': 289,
                  'Mc': 289, 'Lv': 292, 'Ts': 294, 'Og': 294, 'ZERO': 0}

SECTION_SORTER = {
    "atoms": {
        "in_kw": None,
        "in_header": ["atom", "charge", "sigma", "epsilon"],
        "sec_number": None,
        "desired_split": None,
        "desired_cols": None,
        "out_kw": None,
        "ff_header": ["epsilon", "sigma"],
        "topo_header": ["mol-id", "type", "charge", "x", "y", "z"]
    },
    "bonds": {
        "in_kw": "Stretch",
        "in_header": ["atom1", "atom2", "k", "r0"],
        "sec_number": 5,
        "desired_split": 2,
        "desired_cols": 4,
        "out_kw": ["Bond Coeffs", "Bonds"],
        "ff_header": ["k", "r0"],
        "topo_header": ["type", "atom1", "atom2"]
    },
    "angles": {
        "in_kw": "Bending",
        "in_header": ["atom1", "atom2", "atom3", "k", "theta0"],
        "sec_number": 6,
        "desired_split": 1,
        "desired_cols": 5,
        "out_kw": ["Angle Coeffs", "Angles"],
        "ff_header": ["k", "theta0"],
        "topo_header": ["type", "atom1", "atom2", "atom3"]
    },
    "dihedrals": {
        "in_kw": "proper Torsion",
        "in_header": [
            "atom1", "atom2", "atom3", "atom4", "v1", "v2", "v3", "v4"
        ],
        "sec_number": 7,
        "desired_split": 1,
        "desired_cols": 8,
        "out_kw": ["Dihedral Coeffs", "Dihedrals"],
        "ff_header": ["v1", "v2", "v3", "v4"],
        "topo_header": ["type", "atom1", "atom2", "atom3", "atom4"]
    },
    "impropers": {
        "in_kw": "improper Torsion",
        "in_header": ["atom1", "atom2", "atom3", "atom4", "v2"],
        "sec_number": 8,
        "desired_split": 1,
        "desired_cols": 5,
        "out_kw": ["Improper Coeffs", "Impropers"],
        "ff_header": ["v1", "v2", "v3", "v4"],
        "topo_header": ["type", "atom1", "atom2", "atom3", "atom4"]
    },
}

BOX = """{0:6f} {1:6f} xlo xhi
{0:6f} {1:6f} ylo yhi
{0:6f} {1:6f} zlo zhi"""


def atom_vec(atom1, atom2, dimension):
    """
    Calculate the vector of the positions from atom2 to atom1.
    """
    vec = [0, 0, 0]
    for i in range(3):
        diff = atom1.position[i]-atom2.position[i]
        if diff > dimension[i]/2:
            vec[i] = diff - dimension[i]
        elif diff < - dimension[i]/2:
            vec[i] = diff + dimension[i]
        else:
            vec[i] = diff
    return np.array(vec)


def position_vec(pos1, pos2, dimension):
    """
    Calculate the vector from pos2 to pos2.
    """
    vec = [0, 0, 0]
    for i in range(3):
        diff = pos1[i]-pos2[i]
        if diff > dimension[i]/2:
            vec[i] = diff - dimension[i]
        elif diff < - dimension[i]/2:
            vec[i] = diff + dimension[i]
        else:
            vec[i] = diff
    return np.array(vec)


def mass_to_name(df):
    """
    Create a dict for mapping atom type id to element from the mass information.

    Args:
        df (pandas.DataFrame): The masses attribute from LammpsData object
    Return:
        dict: The element dict.
    """
    atoms = {}
    for row in df.index:
        for item in MM_of_Elements.items():
            if math.isclose(df["mass"][row], item[1], abs_tol=0.01):
                atoms[row] = item[0]
    return atoms


def ff_parser(ff_dir, xyz_dir):
    """
    A parser to convert a force field field from Maestro format
    to LAMMPS data format.

    Args:
        ff_dir (str): The path to the Maestro force field file.
        xyz_dir (str): The path to the xyz structure file.
    Return:
        str: The output LAMMPS data string.
    """
    with open(xyz_dir, 'r') as f_xyz:
        molecule = pd.read_table(
            f_xyz,
            skiprows=2,
            delim_whitespace=True,
            names=['atom', 'x', 'y', 'z']
        )
        coordinates = molecule[["x", "y", "z"]]
        lo = coordinates.min().min() - 0.5
        hi = coordinates.max().max() + 0.5
    with open(ff_dir, 'r') as f:
        lines_org = f.read()
        lines = lines_org.split("\n\n")
        atoms = "\n".join(lines[4].split("\n", 4)[4].split("\n")[:-1])
        dfs = dict()
        dfs["atoms"] = pd.read_csv(
            StringIO(atoms),
            names=SECTION_SORTER.get("atoms").get("in_header"),
            delim_whitespace=True,
            usecols=[0, 4, 5, 6]
        )
        dfs["atoms"] = pd.concat(
            [dfs["atoms"], coordinates],
            axis=1
        )
        dfs["atoms"].index += 1
        dfs["atoms"].index.name = "type"
        dfs["atoms"] = dfs["atoms"].reset_index()
        dfs["atoms"].index += 1
        types = dfs["atoms"].copy().reset_index().set_index('atom')['type']
        replace_dict = {
            "atom1": dict(types),
            "atom2": dict(types),
            "atom3": dict(types),
            "atom4": dict(types)
        }
        counts = dict()
        counts["atoms"] = len(dfs["atoms"].index)
        mass_list = list()
        for index, row in dfs["atoms"].iterrows():
            mass_list.append(
                MM_of_Elements.get(re.split(r'(\d+)', row['atom'])[0])
            )
        mass_df = pd.DataFrame(mass_list)
        mass_df.index += 1
        mass_string = mass_df.to_string(
            header=False,
            index_names=False,
            float_format="{:.3f}".format
        )
        masses = ["Masses", mass_string]
        ff = ["Pair Coeffs"]
        dfs["atoms"]["mol-id"] = 1
        atom_ff_string = dfs["atoms"][
            SECTION_SORTER["atoms"]["ff_header"]
        ].to_string(
            header=False,
            index_names=False
        )
        ff.append(atom_ff_string)
        topo = ["Atoms"]
        atom_topo_string = dfs["atoms"][
            SECTION_SORTER["atoms"]["topo_header"]
        ].to_string(
            header=False,
            index_names=False
        )
        topo.append(atom_topo_string)
        for section in list(SECTION_SORTER.keys())[1:]:
            if SECTION_SORTER[section]["in_kw"] in lines_org:
                a, b, c, d = SECTION_SORTER[section]["sec_number"],\
                             SECTION_SORTER[section]["desired_split"],\
                             SECTION_SORTER[section]["desired_cols"],\
                             SECTION_SORTER[section]["in_header"]
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
                    dfs[section]["v1"] = 0.0
                    dfs[section]["v3"] = 0.0
                    dfs[section]["v4"] = 0.0
                ff_string = dfs[section][
                    SECTION_SORTER[section]["ff_header"]
                ].to_string(
                    header=False,
                    index_names=False
                )
                ff.append(SECTION_SORTER[section]["out_kw"][0])
                ff.append(ff_string)
                topo_string = dfs[section][
                    SECTION_SORTER[section]["topo_header"]
                ].to_string(
                    header=False,
                    index_names=False
                )
                topo.append(SECTION_SORTER[section]["out_kw"][1])
                topo.append(topo_string)
                counts[section] = len(dfs[section].index)
        max_stats = len(str(max(list(counts.values()))))
        stats_template = "{:>%d}  {}" % max_stats
        count_lines = [stats_template.format(v, k) for k, v in counts.items()]
        type_lines = [
            stats_template.format(v, k[:-1] + " types")
            for k, v in counts.items()
        ]
        stats = "\n".join(count_lines + [""] + type_lines)
        header = [
            f"LAMMPS data file created by mdgo (by {__author__})\n"
            "# OPLS force field: harmonic, harmonic, opls, opls",
            stats,
            BOX.format(lo, hi)
        ]
        data_string = "\n\n".join(header + masses + ff + topo) + "\n"
        return data_string
