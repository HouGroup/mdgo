# coding: utf-8
# Copyright (c) Tingzheng Hou.
# Distributed under the terms of the MIT License.

"""
Computes the volume for each ligand or active site in a file.

In ligand mode, the volume of the entire structure is calculated. The -x, -y,
-z, -xsize, -ysize and -zsize options are ignored.

In active site mode, the unoccupied volume within a cube is calculated. The
center of the cube is defined by the -x, -y and -z options, and the size of
the cube is defined by the -xsize, -ysize and -zsize options.

"""

import sys
import os
import argparse
from typing import Union, Optional, Tuple, Dict
from pymatgen.core import Molecule, Element
import numpy as np


DEFAULT_VDW = 1.5  # See Ev:130902


def parse_command_line():
    """

    The command line parser helper function.

    Usage:
        python volume.py -xyz <input_xyz> [options]
    """
    usage = """
    python volume.py -xyz <input_xyz> [options]
    """
    parser = argparse.ArgumentParser(usage=usage, description=__doc__)

    parser.add_argument(
        "-i", "-xyz", type=str, dest="ixyz", default="", help="Input xyz file name", metavar="FILE", required=True
    )
    parser.add_argument(
        "-m",
        "-mode",
        type=str,
        dest="mode",
        choices=["lig", "act"],
        default="lig",
        help="Ligand or active site volume <lig|act> (default=lig)",
        metavar="MODE",
    )
    parser.add_argument(
        "-t",
        "-type",
        type=str,
        dest="radii_type",
        choices=["Bondi", "Lange", "pymatgen"],
        default="Bondi",
        help="Type of radii <Bondi|Lange|pymatgen> (default=Bondi)",
        metavar="TYPE",
    )
    parser.add_argument(
        "-n",
        "-name",
        type=str,
        dest="name",
        default="",
        help="Name of molecule",
        metavar="NAME",
    )
    parser.add_argument(
        "-v",
        "-volume",
        type=str,
        dest="molar_volume",
        choices=[
            "yes",
            "no",
            "y",
            "n",
            "Y",
            "N",
            "Yes",
            "No",
            "1",
            "0",
            "t",
            "f",
            "T",
            "F",
            "true",
            "True",
            "false",
            "False",
        ],
        default="yes",
        help="Print volume as molar volume <yes(y/Y/Yes)|no(n/N/No)> (default=yes)",
        metavar="YES OR NO",
    )
    parser.add_argument(
        "-H",
        "-exclude-h",
        type=str,
        dest="exclude_h",
        choices=[
            "yes",
            "no",
            "y",
            "n",
            "Y",
            "N",
            "Yes",
            "No",
            "1",
            "0",
            "t",
            "f",
            "T",
            "F",
            "true",
            "True",
            "false",
            "False",
        ],
        default="yes",
        help="Exclude volume of H <yes(y/Y/Yes)|no(n/N/No)> (default=yes)",
        metavar="YES OR NO",
    )
    parser.add_argument(
        "-r",
        "-resolution",
        type=float,
        dest="res",
        default="0.1",
        help="Resolution for volume grid (default=1.0)",
        metavar="N",
    )
    parser.add_argument(
        "-xsize",
        type=float,
        dest="xsize",
        default="10.0",
        help="X side length for volume grid (default=10.0)",
        metavar="N",
    )
    parser.add_argument(
        "-ysize",
        type=float,
        dest="ysize",
        default="10.0",
        help="Y side length for volume grid (default=10.0)",
        metavar="N",
    )
    parser.add_argument(
        "-zsize",
        type=float,
        dest="zsize",
        default="10.0",
        help="Z side length for volume grid (default=10.0)",
        metavar="N",
    )
    parser.add_argument(
        "-x",
        "-xcent",
        type=float,
        dest="xcent",
        default="0.0",
        help="X center for volume grid (default=0.0)",
        metavar="X",
    )
    parser.add_argument(
        "-y",
        "-ycent",
        type=float,
        dest="ycent",
        default="0.0",
        help="Y center for volume grid (default=0.0)",
        metavar="Y",
    )
    parser.add_argument(
        "-z",
        "-zcent",
        type=float,
        dest="zcent",
        default="0.0",
        help="Z center for volume grid (default=0.0)",
        metavar="Z",
    )

    args = parser.parse_args()

    if args.ixyz == "":
        parser.print_help()
        sys.exit(1)
    if not os.path.exists(args.ixyz):
        print(f"\nError: Input file '{args.ixyz}' not found.\n")
        sys.exit(1)

    return args


def get_max_dimensions(mol: Molecule) -> Tuple[float, float, float, float, float, float]:
    """
    Calculates the dimension of a Molecule

    Args:
        mol: A Molecule object.

    Returns:
        xmin, xmax, ymin, ymax, zmin, zmax
    """

    xmin = 9999
    ymin = 9999
    zmin = 9999
    xmax = -9999
    ymax = -9999
    zmax = -9999
    for a in mol.sites:
        if a.x < xmin:
            xmin = a.x
        if a.x > xmax:
            xmax = a.x
        if a.y < ymin:
            ymin = a.y
        if a.y > ymax:
            ymax = a.y
        if a.z < zmin:
            zmin = a.z
        if a.z > zmax:
            zmax = a.z
    return xmin, xmax, ymin, ymax, zmin, zmax


def set_max_dimensions(
    x: float = 0.0, y: float = 0.0, z: float = 0.0, x_size: float = 10.0, y_size: float = 10.0, z_size: float = 10.0
) -> Tuple[float, float, float, float, float, float]:
    """
    Set the max dimensions for calculating active site volume.

    Args:
        x: X center for volume grid. Default to 0.0.
        y: Y center for volume grid. Default to 0.0.
        z: Y center for volume grid. Default to 0.0.
        x_size: X side length for volume grid. Default to 10.0.
        y_size: Y side length for volume grid. Default to 10.0.
        z_size: Z side length for volume grid. Default to 10.0.

    Returns:
        x_min, x_max, y_min, y_max, z_min, z_max
    """
    x_min = x - (x_size / 2)
    x_max = x + (x_size / 2)
    y_min = y - (y_size / 2)
    y_max = y + (y_size / 2)
    z_min = z - (z_size / 2)
    z_max = z + (z_size / 2)
    return x_min, x_max, y_min, y_max, z_min, z_max


def round_dimensions(
    x_min: float, x_max: float, y_min: float, y_max: float, z_min: float, z_max: float, mode: str = "lig"
) -> Tuple[float, float, float, float, float, float]:
    """
    Round dimensions to a larger box size (+ buffer).

    Args:
        x_min: x min.
        x_max: x max.
        y_min: y min.
        y_max: y max.
        z_min: z min.
        z_max: z max.
        mode: "lig" or "act" mode. Default to "lig".

    Returns:
        x0, x1, y0, y1, z0, z1
    """
    buffer = 0.0  # addition to box for ligand calculations
    if mode == "lig":
        buffer = 1.5
    x0 = np.floor(x_min - buffer)
    x1 = np.ceil(x_max + buffer)
    y0 = np.floor(y_min - buffer)
    y1 = np.ceil(y_max + buffer)
    z0 = np.floor(z_min - buffer)
    z1 = np.ceil(z_max + buffer)
    return x0, x1, y0, y1, z0, z1


def dsq(a1: float, a2: float, a3: float, b1: float, b2: float, b3: float) -> float:
    """
    Squared distance between a and b

    Args:
        a1: x coordinate of a
        a2: y coordinate of a
        a3: z coordinate of a
        b1: x coordinate of b
        b2: y coordinate of b
        b3: z coordinate of b

    Returns:
        squared distance
    """
    d2 = (b1 - a1) ** 2 + (b2 - a2) ** 2 + (b3 - a3) ** 2
    return d2


def get_dimensions(
    x0: float, x1: float, y0: float, y1: float, z0: float, z1: float, res: float = 0.1
) -> Tuple[int, int, int]:
    """
    Mesh dimensions in unit of res.

    Args:
        x0: x min.
        x1: x max.
        y0: y min.
        y1: y max.
        z0: z min.
        z1: z max.
        res: Resolution of the mesh to use in Å

    Returns:
        xsteps, ysteps, zsteps
    """
    xrange = x1 - x0
    yrange = y1 - y0
    zrange = z1 - z0

    xsteps = int(xrange // res)
    ysteps = int(yrange // res)
    zsteps = int(zrange // res)

    return xsteps, ysteps, zsteps


def make_matrix(x_num: int, y_num: int, z_num: int) -> np.ndarray:
    """
    Make a matrix of None with specified dimensions.

    Args:
        x_num: x dimension.
        y_num: y dimension.
        z_num: z dimension.

    Returns:
        matrix
    """

    matrix = np.array([[[None for _ in range(z_num)] for _ in range(y_num)] for _ in range(x_num)])
    return matrix


def get_radii(radii_type: str = "Bondi") -> Dict[str, float]:
    """
    Get a radii dict by type.

    Args:
        radii_type: The radii type. Valid types are "Bondi", "Lange", and "pymatgen". Default to "Bondi".

    Return:
        A radii dict.

    """
    if radii_type == "Bondi":
        radii = {
            "H": 1.20,
            "B": 2.00,
            "C": 1.70,
            "N": 1.55,
            "O": 1.52,
            "F": 1.47,
            "Si": 2.10,
            "P": 1.80,
            "S": 1.80,
            "Cl": 1.75,
            "Br": 1.85,
            "I": 1.98,
        }
    elif radii_type == "Lange":  # from Lange's Handbook of Chemistry
        radii = {
            "H": 1.20,
            "B": 2.08,
            "C": 1.85,
            "N": 1.54,
            "O": 1.40,
            "F": 1.35,
            "Si": 2.00,
            "P": 1.90,
            "S": 1.85,
            "Cl": 1.81,
            "Br": 1.95,
            "I": 2.15,
        }
    elif radii_type == "pymatgen":
        radii = {Element(e).symbol: Element(e).van_der_waals_radius for e in Element.__members__}
    else:
        print("Wrong option for radii type: Choose Bondi, Lange, or pymatgen")
        sys.exit()
    return radii


def fill_volume_matrix(
    mol: Molecule,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    z0: float,
    z1: float,
    res: float,
    matrix: np.ndarray,
    radii_type: str,
    exclude_h: bool = True,
) -> np.ndarray:
    """
    This method perform the mesh point filling algorithm on a given matrix.

    Args:
        mol: The Molecule object to calculate volume.
        x0: x min.
        x1: x max.
        y0: y min.
        y1: y max.
        z0: z min.
        z1: z max.
        res: Resolution of the mesh to use when estimating molar volume, in Å
        matrix: The mesh matrix to perform mesh point filling.
        radii_type: The radii type. Valid types are "Bondi", "Lange", and "pymatgen". Default to "Bondi".
        exclude_h: Whether to exclude H when estimating molar volume. Default to True.

    Returns:
        The filled matrix.
    """
    sys.stdout.flush()

    radii = get_radii(radii_type)  # approximate heavy-atom radii

    xrange = x1 - x0
    yrange = y1 - y0
    zrange = z1 - z0

    xsteps = int(xrange // res)
    ysteps = int(yrange // res)
    zsteps = int(zrange // res)

    for a in mol.sites:
        element = str(a.species.elements[0])
        if exclude_h:
            if element == "H":
                continue
        radius = radii.get(element, DEFAULT_VDW)
        for i in range(0, xsteps):
            if abs(a.x - (x0 + 0.5 * res + i * res)) < radius:
                for j in range(0, ysteps):
                    if abs(a.y - (y0 + 0.5 * res + j * res)) < radius:
                        for k in range(0, zsteps):
                            if matrix[i][j][k] != 1:
                                if abs(a.z - (z0 + 0.5 * res + k * res)) < radius:
                                    if (
                                        dsq(
                                            a.x,
                                            a.y,
                                            a.z,
                                            x0 + 0.5 * res + i * res,
                                            y0 + 0.5 * res + j * res,
                                            z0 + 0.5 * res + k * res,
                                        )
                                        < (radius * radius)
                                    ):
                                        matrix[i][j][k] = 1
                                    else:
                                        matrix[i][j][k] = 0
                                else:
                                    matrix[i][j][k] = 0
    return matrix


def get_occupied_volume(matrix: np.ndarray, res: float, name: Optional[str] = None, molar_volume=True) -> float:
    """
    Get the occupied volume of the molecule in the box.

    Args:
        matrix: The filled mesh matrix.
        res: Resolution of the mesh in Å.
        name: The name of the molecule
        molar_volume: Whether to return molar_volume, otherwise molecular volume. Default to True.

    Returns:
        Volume
    """
    v = np.count_nonzero(matrix) * res * res * res
    if name is not None:
        print(name + f" molar volume = {(v * 0.6022):5.1f} cm^3/mol")
    if molar_volume:
        return v * 0.60221409  # cm^3/mol
    return v  # Å^3


def get_unoccupied_volume(matrix: np.ndarray, res: float, name: Optional[str] = None, molar_volume=True) -> float:
    """
    Get the unoccupied volume of the molecule in the box.

    Args:
        matrix: The filled mesh matrix.
        res: Resolution of the mesh in Å.
        name: The name of the molecule
        molar_volume: Whether to return molar_volume, otherwise molecular volume. Default to True.

    Returns:
        Volume
    """
    v = np.count_nonzero(matrix == 0) * res * res * res
    if name is not None:
        print(name + f" molar volume = {(v * 0.6022):5.1f} cm^3/mol")
    if molar_volume:
        return v * 0.60221409  # cm^3/mol
    return v  # Å^3


def molecular_volume(
    mol: Union[str, Molecule],
    name: Optional[str] = None,
    res: float = 0.1,
    radii_type: str = "Bondi",
    molar_volume: bool = True,
    exclude_h: bool = True,
    mode: str = "lig",
    x_cent: float = 0.0,
    y_cent: float = 0.0,
    z_cent: float = 0.0,
    x_size: float = 10.0,
    y_size: float = 10.0,
    z_size: float = 10.0,
) -> float:
    """
    Estimate the molar volume in cm^3/mol or volume in Å^3

    Args:
        mol: Molecule object or path to .xyz or other file that can be read
            by Molecule.from_file()
        name: String representing the name of the molecule, e.g. "NaCl"
        res: Resolution of the mesh to use when estimating molar volume, in Å
        radii_type: "Bondi", "Lange", or "pymatgen". Bondi and Lange vdW radii
            are compiled in this package for H, B, C, N, O, F, Si, P, S, Cl, Br,
            and I. Choose 'pymatgen' to use the vdW radii from pymatgen.Element,
            which are available for most elements and reflect the latest values in
            the CRC handbook.
        molar_volume: Whether to return molar volume. If false, then return volume.
            Default to True (molar volume).
        exclude_h: Whether to exclude H atoms during the calculation.
            Default to True.
        mode: In ligand mode ("lig"), the volume of the entire structure is calculated.
            In active site mode ("act"), the unoccupied volume within a cube is calculated.
            Default to "lig".
        x_cent: X center for volume grid. Default to 0.0.
        y_cent: Y center for volume grid. Default to 0.0.
        z_cent: Z center for volume grid. Default to 0.0.
        x_size: X side length for volume grid. Default to 10.0.
        y_size: Y side length for volume grid. Default to 10.0.
        z_size: Z side length for volume grid. Default to 10.0.

    Returns:
        The molar volume in cm^3/mol or volume in Å^3.
    """
    if isinstance(mol, str):
        molecule = Molecule.from_file(mol)
    else:
        molecule = mol
    if mode == "lig":
        print("Calculating occupied volume...")
        x_min, x_max, y_min, y_max, z_min, z_max = get_max_dimensions(molecule)
        x0, x1, y0, y1, z0, z1 = round_dimensions(x_min, x_max, y_min, y_max, z_min, z_max, mode)
    elif mode == "act":
        print("Calculating unoccupied volume...")
        x0, x1, y0, y1, z0, z1 = set_max_dimensions(x_cent, y_cent, z_cent, x_size, y_size, z_size)
    else:
        raise ValueError("Mode options are 'lig' and 'act'.")
    x_num, y_num, z_num = get_dimensions(x0, x1, y0, y1, z0, z1, res)
    volume_matrix = make_matrix(x_num, y_num, z_num)
    volume_matrix = fill_volume_matrix(
        molecule, x0, x1, y0, y1, z0, z1, res, volume_matrix, radii_type, exclude_h=exclude_h
    )
    if mode == "lig":
        molar_vol = get_occupied_volume(volume_matrix, res, name, molar_volume=molar_volume)
    else:
        molar_vol = get_unoccupied_volume(volume_matrix, res, name, molar_volume=molar_volume)
    return molar_vol


if __name__ == "__main__":
    """
    ec = Molecule.from_file(
        "/Users/th/Downloads/package/packmol-17.163/EC.xyz"
    )
    emc = Molecule.from_file(
        "/Users/th/Downloads/package/packmol-17.163/EMC.xyz"
    )
    dec = Molecule.from_file(
        "/Users/th/Downloads/package/packmol-17.163/DEC.xyz"
    )
    pf6 = Molecule.from_file(
        "/Users/th/Downloads/package/packmol-17.163/PF6.xyz"
    )
    tfsi = Molecule.from_file(
        "/Users/th/Downloads/package/packmol-17.163/TFSI.xyz"
    )
    lipf6 = Molecule.from_file(
        "/Users/th/Downloads/package/packmol-17.163/LiPF6.xyz"
    )
    """
    options = parse_command_line()

    print(
        molecular_volume(
            options.ixyz,
            name=options.name if options.name != "" else None,
            res=options.res,
            radii_type=options.radii_type,
            molar_volume=options.molar_volume in ["yes", "y", "Y", "Yes", "1", "t", "T", "true", "True"],
            exclude_h=options.exclude_h in ["yes", "y", "Y", "Yes", "1", "t", "T", "true", "True"],
            mode=options.mode,
        ),
        "cm^3/mol",
    )
