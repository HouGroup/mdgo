"""
Computes the volume for each ligand or active site in a file.

In ligand mode, the volume of the entire structure is calculated. The -x, -y,
  -z, -xsize, -ysize and -zsize options are ignored.

In active site mode, the unoccupied volume within a cube is calculated. The
center of the cube is defined by the -x, -y and -z options, and the size of
the cube is defined by the -xsize, -ysize and -zsize options.

"""

import math
import sys
import os
import argparse
from pymatgen.core import Molecule, Element
from typing import Union, Optional


DEFAULT_VDW = 1.5  # See Ev:130902


def parse_command_line():
    usage = """
    python volume.py -xyz <input_xyz> [options]
    """
    parser = argparse.ArgumentParser(usage=usage, description=__doc__)

    parser.add_argument("-i", "-ixyz", type=str, dest="ixzy", default="", help="Input xyz file name", metavar="FILE")
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
        print("\nError: Input file '%s' not found.\n" % args.ixyz)
        sys.exit(1)

    return args


def get_max_dimensions(mol):

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


def set_max_dimensions(x, y, z, xsize, ysize, zsize):
    xmin = x - (xsize / 2)
    xmax = x + (xsize / 2)
    ymin = y - (ysize / 2)
    ymax = y + (ysize / 2)
    zmin = z - (zsize / 2)
    zmax = z + (zsize / 2)
    return xmin, xmax, ymin, ymax, zmin, zmax


def round_dimensions(xmin, xmax, ymin, ymax, zmin, zmax):
    buffer = 1.5  # addition to box for ligand calculations
    x0 = math.floor(xmin - buffer)
    x1 = math.ceil(xmax + buffer)
    y0 = math.floor(ymin - buffer)
    y1 = math.ceil(ymax + buffer)
    z0 = math.floor(zmin - buffer)
    z1 = math.ceil(zmax + buffer)
    return x0, x1, y0, y1, z0, z1


def dsq(a1, a2, a3, b1, b2, b3):
    d2 = (b1 - a1) ** 2 + (b2 - a2) ** 2 + (b3 - a3) ** 2
    return d2


def get_dimensions(x0, x1, y0, y1, z0, z1, res):
    xrange = x1 - x0
    yrange = y1 - y0
    zrange = z1 - z0

    xsteps = int(xrange // res)
    ysteps = int(yrange // res)
    zsteps = int(zrange // res)

    return xsteps, ysteps, zsteps


##################################
def make_matrix(xnum, ynum, znum):

    matrix = [None] * xnum
    for i in range(xnum):
        matrix[i] = [None] * ynum
        for j in range(ynum):
            matrix[i][j] = [None] * znum
    return matrix


def get_radii(type):
    radii = {}
    if type == "Bondi":
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
    elif type == "Lange":  # from Lange's Handbook of Chemistry
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
    elif type == "pymatgen":
        radii = {Element(e).symbol: Element(e).van_der_waals_radius for e in Element.__members__.keys()}
    else:
        print("Wrong option for radii type: Choose Bondi, Lange, or pymatgen")
        sys.exit()
    return radii


def fill_volume_matrix(mol, x0, x1, y0, y1, z0, z1, res, matrix, radii_type):
    sys.stdout.flush()

    radii = get_radii(radii_type)  # approximate heavy-atom radii

    xrange = x1 - x0
    yrange = y1 - y0
    zrange = z1 - z0

    xsteps = int(xrange // res)
    ysteps = int(yrange // res)
    zsteps = int(zrange // res)

    for a in mol.sites:
        element = a.species.elements[0]
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


def print_occupied_volume(matrix, res, name):

    v = 0
    i = -1
    for x in matrix:
        i += 1
        j = -1
        for y in x:
            j += 1
            k = -1
            for z in y:
                k += 1
                if matrix[i][j][k] == 1:
                    v += 1

    v = v * res * res * res
    print(name + " volume = %5.1f Å^3" % v)
    print(name + " molar volume = %5.1f cm^3/mol" % (v * 0.6022))
    return v * 0.6022


def molecular_volume(path: Union[str, Molecule], name: Optional[str] = "", res=0.1, radii_type="Bondi") -> float:
    """
    Estimate the molar volume in cm^3/mol

    Args:
        path: Molecule object or path to .xyz or other file that can be read
            by Molecule.from_file()
        name: String representing the name of the molecule, e.g. "NaCl"
        res: Resolution of the mesh to use when estimating molar volume, in Å
        radii_type: "Bondi", "Lange", or "pymatgen". Bondi and Lange vdW radii
            are compiled in this package for H, B, C, N, O, F, Si, P, S, Cl, Br,
            and I. Choose 'pymatgen' to use the vdW radii from pymatgen.Element,
            which are available for most elements and reflect the latest values in
            the CRC handbook.
    Returns:
        float: The molar volume in cm^3/mol.
    """
    if isinstance(path, str):
        molecule = Molecule.from_file(path)
    else:
        molecule = path
    xmin, xmax, ymin, ymax, zmin, zmax = get_max_dimensions(molecule)
    x0, x1, y0, y1, z0, z1 = round_dimensions(xmin, xmax, ymin, ymax, zmin, zmax)
    xnum, ynum, znum = get_dimensions(x0, x1, y0, y1, z0, z1, res)
    volume_matrix = make_matrix(xnum, ynum, znum)
    volume_matrix = fill_volume_matrix(molecule, x0, x1, y0, y1, z0, z1, res, volume_matrix, radii_type)
    molar_vol = print_occupied_volume(volume_matrix, res, name)
    return molar_vol


if __name__ == "__main__":
    """
    options = parse_command_line()
    if options.mode == "lig":
        print("Calculating occupied volume...")
    elif options.mode == "act":
        print("Calculating unoccupied volume...")
    print("Title, Volume(A^3)")

    for st in structure.StructureReader(options.imae):
        if options.mode == "lig":
            (xmin, xmax, ymin, ymax, zmin, zmax) = get_max_dimensions(st)
            (x0, x1, y0, y1, z0, z1) = round_dimensions(
                xmin, xmax, ymin, ymax, zmin, zmax, options.mode)
        elif options.mode == "act":
            (x0, x1, y0, y1, z0, z1) = set_max_dimensions(
                options.xcent, options.ycent, options.zcent, options.xsize,
                options.ysize, options.zsize)
        (xnum, ynum, znum) = get_dimensions(x0, x1, y0, y1, z0, z1, options.res)
        volume_matrix = make_matrix(xnum, ynum, znum)
        volume_matrix = fill_volume_matrix(st, x0, x1, y0, y1, z0, z1,
                                           options.res, volume_matrix,
                                           options.radii_type)
        if options.mode == "lig":
            print_occupied_volume(volume_matrix, options.res)
        elif options.mode == "act":
            print_unoccupied_volume(volume_matrix, options.res)


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
    print(molecular_volume(lipf6, "lipf6"), "cm^3/mol")
    print(molecular_volume(pf6, "pf6"), "cm^3/mol")
    """
