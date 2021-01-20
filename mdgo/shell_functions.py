# atom selection functions
import warnings
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import distances

def get_atom_group(u, selection):
    """
    Casts an Atom, AtomGroup, Residue, or ResidueGroup to AtomGroup.

    Args:
        u: universe that contains selection
        selection: a selection of atoms

    Returns:
        AtomGroup

    """
    assert isinstance(selection, (mda.core.groups.Residue,
                                  mda.core.groups.ResidueGroup,
                                  mda.core.groups.Atom,
                                  mda.core.groups.AtomGroup)), \
        "central_species must be one of the preceding types"
    if isinstance(selection, (mda.core.groups.Residue, mda.core.groups.ResidueGroup)):
        selection = selection.atoms
    if isinstance(selection, mda.core.groups.Atom):
        selection = u.select_atoms(f'index {selection.index}')
    return selection


# test this
def get_n_shells(u, central_species, n_shell=2, radius=3, ignore_atoms=None):
    """
    A list containing the nth shell at the nth index. Note that the shells have 0 intersection.
    For example, calling get_n_shells with n_shell = 2 would return: [central_species, first_shell, second_shell].
    This scales factorially so probably don't go over n_shell = 3

    Args:
        u: universe that contains central species
        central_species: An Atom, AtomGroup, Residue, or ResidueGroup to AtomGroup.
        n_shell: number of shells to return
        radius: radius used to select atoms in next shell
        ignore_atoms: these atoms will be ignored

    Returns:
        List of n shells

    """
    if n_shell > 3:
        warnings.warn('get_n_shells scales factorially, very slow')
    central_species = get_atom_group(u, central_species)
    if not ignore_atoms:
        ignore_atoms = u.select_atoms('')

def get_cation_anion_shells(u, central_species, cation_group, anion_group, radius=4):
    """
    This is meant to help probe the solvation structure of an electrolyte solution. It will return a list of four
    shells: [central_species (cation), first_shell (anions), second_shell (cations), third_shell (anions)].
    Each outer shell is found by iterating through the atoms in the inner shell and searching within radius.

    Args:
        u: universe that contains central species
        central_species: An cation that is an Atom, AtomGroup, Residue, or ResidueGroup to AtomGroup.
        cation_group: The AtomGroup of all cations in the universe.
        anion_group: The AtomGroup of all anions in the universe.
        radius: the radius used to select nearby cations/anions

    Returns:
        List of four shells

    """
    central_species = get_atom_group(u, central_species)
    first_shell = get_radial_shell(u, central_species, radius) & anion_group
    second_shell = u.select_atoms('')
    for res in first_shell.residues:
        second_shell = second_shell | get_radial_shell(u, res, radius)
    second_shell = (second_shell - central_species) & cation_group
    third_shell = u.select_atoms('')
    for res in second_shell:
        third_shell = third_shell | get_radial_shell(u, res, radius)
    third_shell = (third_shell - first_shell) & anion_group
    fourth_shell = u.select_atoms('')
    for res in third_shell:
        fourth_shell = fourth_shell | get_radial_shell(u, res, radius)
    fourth_shell = (fourth_shell - second_shell - central_species) & cation_group
    return first_shell, second_shell, third_shell, fourth_shell


def get_closest_n_mol(u, central_species, n_mol=5, radius=3):
    """
    Returns the closest n molecules to the central species, an array of their resids,
    and an array of the distance of the closest atom in each molecule.

    Args:
        u: universe that contains central species
        central_species: An Atom, AtomGroup, Residue, or ResidueGroup to AtomGroup.
        n_mol: The number of molecules to return
        radius: an initial search radius to look for closest n mol

    Returns:
        (AtomGroup[molecules], Array[resids], Array[distances])

    """
    central_species = get_atom_group(u, central_species)
    coords = central_species.center_of_mass()
    str_coords = " ".join(str(coord) for coord in coords)
    partial_shell = u.select_atoms(f'point {str_coords} {radius}')
    shell_resids = partial_shell.resids
    if len(np.unique(shell_resids)) < n_mol + 1:
        return get_closest_n_mol(u, central_species, n_mol, radius + 1)
    radii = distances.distance_array(coords, partial_shell.positions, box=u.dimensions)[0]
    ordering = np.argsort(radii)
    ordered_resids = shell_resids[ordering]
    closest_n_resix = np.sort(np.unique(ordered_resids, return_index=True)[1])[0:n_mol + 1]
    str_resids = " ".join(str(resid) for resid in ordered_resids[closest_n_resix])
    full_shell = u.select_atoms(f'resid {str_resids}')
    return full_shell, ordered_resids[closest_n_resix], radii[ordering][closest_n_resix]


def get_radial_shell(u, central_species, radius):
    """
    Returns all molecules with atoms within the radius of the central species. (specifically, within the radius
    of the COM of central species).

    Args:
        u: universe that contains central species
        central_species: An Atom, AtomGroup, Residue, or ResidueGroup to AtomGroup.
        radius: radius used for atom selection

    Returns:
        An AtomGroup

    """
    central_species = get_atom_group(u, central_species)
    coords = central_species.center_of_mass()
    str_coords = " ".join(str(coord) for coord in coords)
    partial_shell = u.select_atoms(f'point {str_coords} {radius}')
    full_shell = partial_shell.residues.atoms
    return full_shell

# atom group counters

def get_counts(atom_group):
    unique_ids = np.unique(atom_group.resids, return_index=True)
    names, counts = np.unique(atom_group.resnames[unique_ids[1]], return_counts=True)
    return {i: j for i, j in zip(names, counts)}

def get_pair_type(u, central_species, cation_group, anion_group, radius=4):
    cation_name = cation_group.names[0]
    anion_name = anion_group.names[0]
    first, second, third, fourth = (get_counts(shell) for shell in
                                    get_cation_anion_shells(u, central_species, cation_group,
                                                           anion_group, radius))
    if len(first) == 0:
        return "SSIP"
    if (len(first) == 1) & (len(second) == 0):
        return "CIP"
    else:
        return "AGG"