# coding: utf-8
# Copyright (c) Tingzheng Hou.
# Distributed under the terms of the MIT License.

"""
This module implements functions for coordination analysis.
"""

import numpy as np
from tqdm.notebook import tqdm
from MDAnalysis import Universe, AtomGroup
from MDAnalysis.core.groups import Atom
from MDAnalysis.analysis.distances import distance_array
from scipy.signal import savgol_filter
from mdgo.util import atom_vec, angle

from typing import Dict, List, Tuple, Union, Callable, Optional

__author__ = "Tingzheng Hou"
__version__ = "1.0"
__maintainer__ = "Tingzheng Hou"
__email__ = "tingzheng_hou@berkeley.edu"
__date__ = "Feb 9, 2021"


def neighbor_distance(
    nvt_run: Universe,
    center_atom: Atom,
    run_start: int,
    run_end: int,
    species: str,
    select_dict: Dict[str, str],
    distance: float,
) -> Dict[str, np.ndarray]:
    """
    Calculates a distance dictionary of neighbor atoms to the center atoms.

    Args:
        nvt_run: An Universe object of wrapped trajectory.
        center_atom: the interested central atom object.
        run_start: Start time step.
        run_end: End time step.
        species: The interested neighbor species in the select_dict.
        select_dict: A dictionary of atom species, where each atom species name is a key
                and the corresponding values are the selection language.
        distance: The neighbor cutoff distance.

    Returns:
        A dictionary of distance of neighbor atoms to the center atom.
    """
    dist_dict = dict()
    time_count = 0
    trj_analysis = nvt_run.trajectory[run_start:run_end:]
    species_selection = select_dict.get(species)
    if species_selection is None:
        raise ValueError("Invalid species selection")
    for ts in trj_analysis:
        selection = (
            "(" + species_selection + ") and (around " + str(distance) + " index " + str(center_atom.id - 1) + ")"
        )
        shell = nvt_run.select_atoms(selection, periodic=True)
        for atom in shell.atoms:
            if str(atom.id) not in dist_dict:
                dist_dict[str(atom.id)] = np.full(run_end - run_start, 100.0)
        time_count += 1
    time_count = 0
    for ts in trj_analysis:
        for atomid in dist_dict.keys():
            dist = distance_array(ts[center_atom.id - 1], ts[(int(atomid) - 1)], ts.dimensions)
            dist_dict[atomid][time_count] = dist
        time_count += 1
    return dist_dict


def find_nearest(
    trj: Dict[str, np.ndarray],
    time_step: float,
    binding_cutoff: float,
    hopping_cutoff: float,
    smooth: int = 51,
) -> Tuple[List[int], Union[float, np.floating], List[int]]:
    """According to the dictionary of neighbor distance, finds the nearest neighbor that the central_atom binds to,
    and calculates the frequency of hopping between each neighbor, and steps when each binding site exhibits
    the closest distance to the central atom.

    Args:
        trj: A python dict of distances between central atom and selected atoms.
        time_step: The time step of the simulation.
        binding_cutoff: Binding cutoff distance.
        hopping_cutoff: Detaching cutoff distance.
        smooth: The length of the smooth filter window. Default to 51.

    Returns:
        Returns an array of nearest neighbors (unique on each timestep),
        the frequency of hopping between sites, and steps when each binding site
        exhibits the closest distance to the central atom.
    """
    time_span = len(list(trj.values())[0])
    for kw in list(trj):
        trj[kw] = savgol_filter(trj.get(kw), smooth, 2)
    site_distance = [100 for _ in range(time_span)]
    sites: List[Union[int, np.integer]] = [0 for _ in range(time_span)]
    start_site = min(trj, key=lambda k: trj[k][0])
    kw_start = trj.get(start_site)
    assert kw_start is not None
    if kw_start[0] < binding_cutoff:
        sites[0] = int(start_site)
        site_distance[0] = kw_start[0]
    else:
        pass
    for time in range(1, time_span):
        if sites[time - 1] == 0:
            old_site_distance = 100
        else:
            old_trj = trj.get(str(sites[time - 1]))
            assert old_trj is not None
            old_site_distance = old_trj[time]
        if old_site_distance > hopping_cutoff:
            new_site = min(trj, key=lambda k: trj[k][time])
            new_trj = trj.get(new_site)
            assert new_trj is not None
            new_site_distance = new_trj[time]
            if new_site_distance > binding_cutoff:
                site_distance[time] = 100
            else:
                sites[time] = int(new_site)
                site_distance[time] = new_site_distance
        else:
            sites[time] = sites[time - 1]
            site_distance[time] = old_site_distance
    sites = [int(i) for i in sites]
    sites_and_distance_array = np.array([[sites[i], site_distance[i]] for i in range(len(sites))])
    steps = []
    closest_step: Optional[int] = 0
    previous_site = sites_and_distance_array[0][0]
    if previous_site == 0:
        closest_step = None
    for i, step in enumerate(sites_and_distance_array):
        site = step[0]
        distance = step[1]
        if site == 0:
            pass
        else:
            if site == previous_site:
                if distance < sites_and_distance_array[closest_step][1]:
                    closest_step = i
                else:
                    pass
            else:
                if closest_step is not None:
                    steps.append(closest_step)
                closest_step = i
                previous_site = site
    if closest_step is not None:
        steps.append(closest_step)
    change = (np.diff([i for i in sites if i != 0]) != 0).sum()
    assert change == len(steps) - 1 or change == len(steps) == 0
    frequency = change / (time_span * time_step)
    return sites, frequency, steps


def find_nearest_free_only(
    trj: Dict[str, np.ndarray],
    time_step: float,
    binding_cutoff: float,
    hopping_cutoff: float,
    smooth: int = 51,
) -> Tuple[List[int], Union[float, np.floating], List[int]]:
    """According to the dictionary of neighbor distance, finds the nearest neighbor that the central_atom binds to, and calculates the frequency of hopping
    between each neighbor, and steps when each binding site exhibits the closest distance to the central atom.
    * Only hopping events with intermediate free state (no binded nearest neighbor) are counted.

    Args:
        trj: A python dict of distances between central atom and selected atoms.
        time_step: The time step of the simulation.
        binding_cutoff: Binding cutoff distance.
        hopping_cutoff: Detaching cutoff distance.
        smooth: The length of the smooth filter window. Default to 51.

    Returns:
        Returns an array of nearest neighbors (unique on each timestep),
        the frequency of hopping between sites, and steps when each binding site
        exhibits the closest distance to the central atom.
    """
    time_span = len(list(trj.values())[0])
    for kw in list(trj):
        trj[kw] = savgol_filter(trj.get(kw), smooth, 2)
    site_distance = [100 for _ in range(time_span)]
    sites: List[Union[int, np.integer]] = [0 for _ in range(time_span)]
    start_site = min(trj, key=lambda k: trj[k][0])
    kw_start = trj.get(start_site)
    assert kw_start is not None
    if kw_start[0] < binding_cutoff:
        sites[0] = int(start_site)
        site_distance[0] = kw_start[0]
    else:
        pass
    for time in range(1, time_span):
        if sites[time - 1] == 0:
            old_site_distance = 100
        else:
            old_trj = trj.get(str(sites[time - 1]))
            assert old_trj is not None
            old_site_distance = old_trj[time]
        if old_site_distance > hopping_cutoff:
            new_site = min(trj, key=lambda k: trj[k][time])
            new_trj = trj.get(new_site)
            assert new_trj is not None
            new_site_distance = new_trj[time]
            if new_site_distance > binding_cutoff:
                site_distance[time] = 100
            else:
                sites[time] = int(new_site)
                site_distance[time] = new_site_distance
        else:
            sites[time] = sites[time - 1]
            site_distance[time] = old_site_distance
    sites = [int(i) for i in sites]
    sites_and_distance_array = np.array([[sites[i], site_distance[i]] for i in range(len(sites))])
    steps = []
    closest_step: Optional[int] = 0
    previous_site = sites_and_distance_array[0][0]
    previous_zero = False
    if previous_site == 0:
        closest_step = None
        previous_zero = True
    for i, step in enumerate(sites_and_distance_array):
        site = step[0]
        distance = step[1]
        if site == 0:
            previous_zero = True
        else:
            if site == previous_site:
                if distance < sites_and_distance_array[closest_step][1]:
                    closest_step = i
                else:
                    pass
            elif not previous_zero:
                previous_site = site
                if distance < sites_and_distance_array[closest_step][1]:
                    closest_step = i
                else:
                    pass
            else:
                if closest_step is not None:
                    steps.append(closest_step)
                closest_step = i
                previous_site = site
    if closest_step is not None:
        steps.append(closest_step)
    frequency = (len(steps) - 1) / (time_span * time_step)
    return sites, frequency, steps


def find_in_n_out(
    trj: Dict[str, np.ndarray], binding_cutoff: float, hopping_cutoff: float, smooth: int = 51, cool: int = 20
) -> Tuple[List[int], List[int]]:
    """Gives the time steps when the central atom binds with the neighbor (binding) or hopping out (hopping)
    according to the dictionary of neighbor distance.

    Args:
        trj: A python dict of distances between central atom and selected atoms.
        binding_cutoff: Binding cutoff distance.
        hopping_cutoff: Detaching cutoff distance.
        smooth: The length of the smooth filter window. Default to 51.
        cool: The cool down timesteps between hopping in and hopping out.

    Returns:
        Two arrays of time steps of hopping in and hopping out event.
    """
    time_span = len(list(trj.values())[0])
    for kw in list(trj):
        trj[kw] = savgol_filter(trj.get(kw), smooth, 2)
    site_distance = [100 for _ in range(time_span)]
    sites = [0 for _ in range(time_span)]
    start_site = min(trj, key=lambda k: trj[k][0])
    kw_start = trj.get(start_site)
    assert kw_start is not None
    if kw_start[0] < binding_cutoff:
        sites[0] = int(start_site)
        site_distance[0] = kw_start[0]
    else:
        pass
    for time in range(1, time_span):
        if sites[time - 1] == 0:
            old_site_distance = 100
        else:
            old_trj = trj.get(str(sites[time - 1]))
            assert old_trj is not None
            old_site_distance = old_trj[time]
        if old_site_distance > hopping_cutoff:
            new_site = min(trj, key=lambda k: trj[k][time])
            new_trj = trj.get(new_site)
            assert new_trj is not None
            new_site_distance = new_trj[time]
            if new_site_distance > binding_cutoff:
                site_distance[time] = 100
            else:
                sites[time] = int(new_site)
                site_distance[time] = new_site_distance
        else:
            sites[time] = sites[time - 1]
            site_distance[time] = old_site_distance
    sites = [int(i) for i in sites]

    last = sites[0]
    steps_in: List[int] = list()
    steps_out: List[int] = list()
    in_cool = cool
    out_cool = cool
    for i, s in enumerate(sites):
        if last == s:
            pass
        elif last == 0:
            in_cool = 0
            steps_in.append(i)
            if out_cool < cool:
                steps_out.pop()
        elif s == 0:
            out_cool = 0
            steps_out.append(i)
            if in_cool < cool:
                steps_in.pop()
        else:
            if cool == 0:
                steps_out.append(i - 1)
                steps_in.append(i)
        last = s
        in_cool += 1
        out_cool += 1
    return steps_in, steps_out


def check_contiguous_steps(
    nvt_run: Universe,
    center_atom: Atom,
    distance_dict: Dict[str, float],
    select_dict: Dict[str, str],
    run_start: int,
    run_end: int,
    checkpoints: np.ndarray,
    lag: int = 20,
) -> Dict[str, np.ndarray]:
    """Returns an array of distance between the center atom and the interested atom
    in the checkpoint +/- lag time range.

    Args:
        nvt_run: An Universe object of wrapped trajectory.
        center_atom: the interested central atom object.
        distance_dict: Dict of Cutoff distance of neighbor for each species.
        select_dict: A dictionary of selection language of atom species.
        run_start: Start time step.
        run_end: End time step.
        checkpoints: The time step of interest to check for contiguous steps
        lag: The range (+/- lag) of the contiguous steps
    """
    coord_num: Dict[str, Union[List[List[int]], np.ndarray]] = {
        x: [[] for _ in range(lag * 2 + 1)] for x in distance_dict.keys()
    }
    trj_analysis = nvt_run.trajectory[run_start:run_end:]
    has = False
    for i, ts in enumerate(trj_analysis):
        log = False
        checkpoint = -1
        for j in checkpoints:
            if abs(i - j) <= lag:
                log = True
                has = True
                checkpoint = j
        if log:
            for kw in distance_dict.keys():
                selection = select_shell(select_dict, distance_dict, center_atom, kw)
                shell = nvt_run.select_atoms(selection, periodic=True)
                coord_num[kw][i - checkpoint + lag].append(len(shell))
    one_atom_ave = dict()
    if has:
        for kw in coord_num:
            np_arrays = np.array([np.array(time).mean() for time in coord_num[kw]])
            one_atom_ave[kw] = np_arrays
    return one_atom_ave


def heat_map(
    nvt_run: Universe,
    floating_atom: Atom,
    cluster_center_sites: List[int],
    cluster_terminal: str,
    cartesian_by_ref: np.ndarray,
    run_start: int,
    run_end: int,
) -> np.ndarray:
    """

    Args:
        nvt_run:
        floating_atom:
        cluster_center_sites:
        cluster_terminal:
        cartesian_by_ref:
        run_start:
        run_end:

    Returns:

    """
    trj_analysis = nvt_run.trajectory[run_start:run_end:]
    coordinates = []
    for i, ts in enumerate(trj_analysis):
        if cluster_center_sites[i] == 0:
            pass
        else:
            center_atom = nvt_run.select_atoms("index " + str(cluster_center_sites[i] - 1))[0]
            selection = "(" + cluster_terminal + ") and (same resid as index " + str(center_atom.id - 1) + ")"
            bind_atoms = nvt_run.select_atoms(selection, periodic=True)
            distances = distance_array(ts[floating_atom.id - 1], bind_atoms.positions, ts.dimensions)
            idx = np.argpartition(distances[0], 3)
            vertex_atoms = bind_atoms[idx[:3]]
            vector_atom = atom_vec(floating_atom, center_atom, ts.dimensions)
            vector_a = atom_vec(vertex_atoms[0], center_atom, ts.dimensions)
            vector_b = atom_vec(vertex_atoms[1], center_atom, ts.dimensions)
            vector_c = atom_vec(vertex_atoms[2], center_atom, ts.dimensions)
            basis_abc = np.transpose([vector_a, vector_b, vector_c])
            abc_atom = np.linalg.solve(basis_abc, vector_atom)
            unit_x = np.linalg.norm(
                cartesian_by_ref[0, 0] * vector_a
                + cartesian_by_ref[0, 1] * vector_b
                + cartesian_by_ref[0, 2] * vector_c
            )
            unit_y = np.linalg.norm(
                cartesian_by_ref[1, 0] * vector_a
                + cartesian_by_ref[1, 1] * vector_b
                + cartesian_by_ref[1, 2] * vector_c
            )
            unit_z = np.linalg.norm(
                cartesian_by_ref[2, 0] * vector_a
                + cartesian_by_ref[2, 1] * vector_b
                + cartesian_by_ref[2, 2] * vector_c
            )
            vector_x = cartesian_by_ref[0] / unit_x
            vector_y = cartesian_by_ref[1] / unit_y
            vector_z = cartesian_by_ref[2] / unit_z
            basis_xyz = np.transpose([vector_x, vector_y, vector_z])
            xyz_atom = np.linalg.solve(basis_xyz, abc_atom)
            coordinates.append(xyz_atom)
    return np.array(coordinates)


def process_evol(
    nvt_run: Universe,
    select_dict: Dict[str, str],
    in_list: Dict[str, List[np.ndarray]],
    out_list: Dict[str, List[np.ndarray]],
    distance_dict: Dict[str, float],
    run_start: int,
    run_end: int,
    lag_step: int,
    distance: float,
    hopping_cutoff: float,
    smooth: int,
    cool: int,
    binding_site: str,
    center_atom: str,
):
    """

    Args:
        nvt_run:
        select_dict:
        in_list:
        out_list:
        distance_dict:
        run_start:
        run_end:
        lag_step:
        distance:
        hopping_cutoff:
        smooth:
        cool:
        binding_site:
        center_atom:

    Returns:

    """
    nvt_run = nvt_run
    center_atoms = nvt_run.select_atoms(select_dict.get(center_atom))
    for atom in tqdm(center_atoms[::]):
        neighbor_trj = neighbor_distance(
            nvt_run, atom, run_start + lag_step, run_end - lag_step, binding_site, select_dict, distance
        )
        hopping_in, hopping_out = find_in_n_out(neighbor_trj, distance, hopping_cutoff, smooth=smooth, cool=cool)
        if len(hopping_in) > 0:
            in_one = check_contiguous_steps(
                nvt_run,
                atom,
                distance_dict,
                select_dict,
                run_start,
                run_end,
                np.array(hopping_in) + lag_step,
                lag=lag_step,
            )
            for kw, value in in_one.items():
                in_list[kw].append(value)
        if len(hopping_out) > 0:
            out_one = check_contiguous_steps(
                nvt_run,
                atom,
                distance_dict,
                select_dict,
                run_start,
                run_end,
                np.array(hopping_out) + lag_step,
                lag=lag_step,
            )
            for kw, value in out_one.items():
                out_list[kw].append(value)


def get_full_coords(
    coords: np.ndarray,
    reflection: Optional[List[np.ndarray]] = None,
    rotation: Optional[List[np.ndarray]] = None,
    inversion: Optional[List[np.ndarray]] = None,
    sample: Optional[int] = None,
) -> np.ndarray:
    """

    Args:
        coords:
        reflection:
        rotation:
        inversion:
        sample:

    Returns:

    """
    coords_full = coords
    if reflection:
        for vec in reflection:
            coords_full = np.concatenate((coords, coords * vec), axis=0)
    if rotation:
        coords_copy = coords_full
        for mat in rotation:
            coords_rot = np.dot(coords_copy, mat)
            coords_full = np.concatenate((coords_full, coords_rot), axis=0)
    if inversion:
        coords_copy = coords_full
        for mat in inversion:
            coords_inv = np.dot(coords_copy, mat)
            coords_full = np.concatenate((coords_full, coords_inv), axis=0)
    if sample:
        index = np.random.choice(coords_full.shape[0], sample, replace=False)
        coords_full = coords_full[index]
    return coords_full


def cluster_coordinates(
    nvt_run: Universe,
    select_dict: Dict[str, str],
    run_start: int,
    run_end: int,
    species: str,
    distance: float,
    basis_vectors: Optional[Union[List[np.ndarray], np.ndarray]] = None,
    cluster_center: str = "center",
) -> np.ndarray:
    """

    Args:
        nvt_run:
        select_dict:
        run_start:
        run_end:
        species:
        distance:
        basis_vectors:
        cluster_center:

    Returns:

    """
    trj_analysis = nvt_run.trajectory[run_start:run_end:]
    cluster_center_atom = nvt_run.select_atoms(select_dict.get(cluster_center), periodic=True)[0]
    selection = (
        "("
        + " or ".join([s for s in species])
        + ") and (around "
        + str(distance)
        + " index "
        + str(cluster_center_atom.id - 1)
        + ")"
    )
    print(selection)
    shell = nvt_run.select_atoms(selection, periodic=True)
    cluster = []
    for atom in shell:
        coord_list = []
        for ts in trj_analysis:
            coord_list.append(atom.position)
        cluster.append(np.mean(np.array(coord_list), axis=0))
    cluster_array = np.array(cluster)
    if basis_vectors:
        if len(basis_vectors) == 2:
            vec1 = basis_vectors[0]
            vec2 = basis_vectors[1]
            vec3 = np.cross(vec1, vec2)
            vec2 = np.cross(vec1, vec3)
        elif len(basis_vectors) == 3:
            vec1 = basis_vectors[0]
            vec2 = basis_vectors[1]
            vec3 = basis_vectors[2]
        else:
            raise ValueError("incorrect vector format")
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)
        vec3 = vec3 / np.linalg.norm(vec3)
        basis_xyz = np.transpose([vec1, vec2, vec3])
        cluster_norm = np.linalg.solve(basis_xyz, cluster_array.T).T
        cluster_norm = cluster_norm - np.mean(cluster_norm, axis=0)
        return cluster_norm
    else:
        return cluster_array


def num_of_neighbor(
    nvt_run: Universe,
    center_atom: Atom,
    distance_dict: Dict[str, float],
    select_dict: Dict[str, str],
    run_start,
    run_end,
    write=False,
    structure_code=None,
    write_freq=0,
    write_path=None,
    element_id_dict=None,
):
    """

    Args:
        nvt_run:
        center_atom:
        distance_dict:
        select_dict:
        run_start:
        run_end:
        write:
        structure_code:
        write_freq:
        write_path:
        element_id_dict:

    Returns:

    """
    time_count = 0
    trj_analysis = nvt_run.trajectory[run_start:run_end:]
    cn_values = dict()
    species = list(distance_dict.keys())
    for kw in species:
        if kw in select_dict.keys():
            cn_values[kw] = np.zeros(int(len(trj_analysis)))
        else:
            print("Invalid species selection")
            return None
    cn_values["total"] = np.zeros(int(len(trj_analysis)))
    for ts in trj_analysis:
        digit_of_species = len(species) - 1
        for kw in species:
            selection = select_shell(select_dict, distance_dict, center_atom, kw)
            shell = nvt_run.select_atoms(selection, periodic=True)
            # for each atom in shell, create/add to dictionary
            # (key = atom id, value = list of values for step function)
            for _ in shell.atoms:
                cn_values[kw][time_count] += 1
                cn_values["total"][time_count] += 10 ** digit_of_species
            digit_of_species = digit_of_species - 1
        if write and cn_values["total"][time_count] == structure_code:
            a = np.random.random()
            if a > 1 - write_freq:
                print("writing")
                selection_write = " or ".join(
                    "(same resid as " + select_shell(select_dict, distance_dict, center_atom, kw) + ")"
                    for kw in species
                )
                cation_selection = select_dict.get("cation")
                assert cation_selection is not None
                selection_write = "((" + selection_write + ")and not " + cation_selection + ")"
                structure = nvt_run.select_atoms(selection_write, periodic=True)
                li_pos = ts[(int(center_atom.id) - 1)]
                path = write_path + str(center_atom.id) + "_" + str(int(ts.time)) + "_" + str(structure_code) + ".xyz"
                write_out(li_pos, structure, element_id_dict, path)
        time_count += 1
    return cn_values


def num_of_neighbor_simple(
    nvt_run: Universe,
    center_atom: Atom,
    distance_dict: Dict[str, float],
    select_dict: Dict[str, str],
    run_start: int,
    run_end: int,
):
    """

    Args:
        nvt_run:
        center_atom:
        distance_dict:
        select_dict:
        run_start:
        run_end:

    Returns:

    """

    time_count = 0
    trj_analysis = nvt_run.trajectory[run_start:run_end:]
    center_selection = "same type as " + str(center_atom.id - 1)
    species = list(distance_dict.keys())[0]
    if species in select_dict.keys():
        cn_values = np.zeros(int(len(trj_analysis)))
    else:
        print("Invalid species selection")
        return None
    for ts in trj_analysis:
        selection = select_shell(select_dict, distance_dict, center_atom, species)
        shell = nvt_run.select_atoms(selection, periodic=True)
        shell_len = len(shell)
        if shell_len == 0:
            cn_values[time_count] = 1
        elif shell_len == 1:
            selection_species = select_shell(center_selection, distance_dict, shell.atoms[0], species)
            shell_species = nvt_run.select_atoms(selection_species, periodic=True)
            shell_species_len = len(shell_species) - 1
            if shell_species_len == 0:
                cn_values[time_count] = 2
            else:
                cn_values[time_count] = 3
        else:
            cn_values[time_count] = 3
        time_count += 1
    cn_values = {"total": cn_values}
    return cn_values


def num_of_neighbor_one_li_simple_extra(
    nvt_run: Universe,
    center_atom: Atom,
    species: str,
    select_dict: Dict[str, str],
    distance: float,
    run_start: int,
    run_end: int,
):
    """

    Args:
        nvt_run:
        center_atom:
        species:
        select_dict:
        distance:
        run_start:
        run_end:

    Returns:

    """

    time_count = 0
    emc_angle = list()
    ec_angle = list()
    trj_analysis = nvt_run.trajectory[run_start:run_end:]
    if species in select_dict.keys():
        cn_values = np.zeros(int(len(trj_analysis)))
    else:
        print("Invalid species selection")
        return None
    for ts in trj_analysis:
        selection = select_shell(select_dict, str(distance), center_atom, species)
        shell = nvt_run.select_atoms(selection, periodic=True)
        shell_len = len(shell)
        if shell_len == 0:
            cn_values[time_count] = 1
        elif shell_len == 1:
            selection_species = select_shell(select_dict, str(distance), shell.atoms[0], "cation")
            shell_species = nvt_run.select_atoms(selection_species, periodic=True)
            shell_species_len = len(shell_species) - 1
            if shell_species_len == 0:
                cn_values[time_count] = 2
                li_pos = center_atom.position
                p_pos = shell.atoms[0].position
                ec_select = select_shell(select_dict, str(3), center_atom, "EC")
                emc_select = (select_dict, str(3), center_atom, "EMC")
                ec_group = nvt_run.select_atoms(ec_select, periodic=True)
                emc_group = nvt_run.select_atoms(emc_select, periodic=True)
                for atom in ec_group.atoms:
                    theta = angle(p_pos, li_pos, atom.position)
                    ec_angle.append(theta)
                for atom in emc_group.atoms:
                    theta = angle(p_pos, li_pos, atom.position)
                    emc_angle.append(theta)
            else:
                cn_values[time_count] = 3
        else:
            cn_values[time_count] = 3
        time_count += 1
    return cn_values, np.array(ec_angle), np.array(emc_angle)


def num_of_neighbor_one_li_simple_extra_two(
    nvt_run: Universe,
    center_atom: Atom,
    species_list: List[str],
    select_dict: Dict[str, str],
    distance_dict: Dict[str, float],
    run_start: int,
    run_end: int,
):
    """

    Args:
        nvt_run:
        center_atom:
        species_list:
        select_dict:
        distance_dict:
        run_start:
        run_end:

    Returns:

    """
    time_count = 0
    trj_analysis = nvt_run.trajectory[run_start:run_end:]
    cip_step = list()
    ssip_step = list()
    agg_step = list()
    cn_values = dict()
    for kw in species_list:
        if kw in select_dict.keys():
            cn_values[kw] = np.zeros(int(len(trj_analysis)))
        else:
            print("Invalid species selection")
            return None
    cn_values["total"] = np.zeros(int(len(trj_analysis)))
    for ts in trj_analysis:
        digit_of_species = len(species_list) - 1
        for kw in species_list:
            selection = select_shell(select_dict, distance_dict, center_atom, kw)
            shell = nvt_run.select_atoms(selection, periodic=True)
            # for each atom in shell, create/add to dictionary
            # (key = atom id, value = list of values for step function)
            for _ in shell.atoms:
                cn_values[kw][time_count] += 1
                cn_values["total"][time_count] += 10 ** digit_of_species
            digit_of_species = digit_of_species - 1

        selection = select_shell(select_dict, distance_dict, center_atom, "anion")
        shell = nvt_run.select_atoms(selection, periodic=True)
        shell_len = len(shell)
        center_selection = "same type as " + str(center_atom.id - 1)
        if shell_len == 0:
            ssip_step.append(time_count)
        elif shell_len == 1:
            selection_species = select_shell(center_selection, distance_dict, shell.atoms[0], "anion")
            shell_species = nvt_run.select_atoms(selection_species, periodic=True)
            shell_species_len = len(shell_species) - 1
            if shell_species_len == 0:
                cip_step.append(time_count)
            else:
                agg_step.append(time_count)
        else:
            agg_step.append(time_count)
        time_count += 1
    cn_ssip = dict()
    cn_cip = dict()
    cn_agg = dict()
    for kw in species_list:
        cn_ssip[kw] = np.mean(cn_values[kw][ssip_step])
        cn_cip[kw] = np.mean(cn_values[kw][cip_step])
        cn_agg[kw] = np.mean(cn_values[kw][agg_step])
    return cn_ssip, cn_cip, cn_agg


# Depth-first traversal
def num_of_neighbor_one_li_complex(
    nvt_run: Universe,
    center_atom: Atom,
    species: str,
    select_dict: Dict[str, str],
    distance: float,
    run_start: int,
    run_end: int,
):
    """

    Args:
        nvt_run:
        center_atom:
        species:
        select_dict:
        distance:
        run_start:
        run_end:

    Returns:

    """
    time_count = 0
    trj_analysis = nvt_run.trajectory[run_start:run_end:]
    cn_values = np.zeros((int(len(trj_analysis)), 4))
    for ts in trj_analysis:
        cation_list = [center_atom.id]
        anion_list = []
        selection = select_shell(select_dict, str(distance), center_atom, species)
        shell = nvt_run.select_atoms(selection, periodic=True)
        for anion_1 in shell.atoms:
            if anion_1.resid not in anion_list:
                anion_list.append(anion_1.resid)
                cn_values[time_count][0] += 1
                shell_anion_1 = nvt_run.select_atoms(
                    "(type 17 and around 3 resid " + str(anion_1.resid) + ")",
                    periodic=True,
                )
                for cation_2 in shell_anion_1:
                    if cation_2.id not in cation_list:
                        cation_list.append(cation_2.id)
                        cn_values[time_count][1] += 1
                        shell_cation_2 = nvt_run.select_atoms(
                            "(type 15 and around 3 index " + str(cation_2.id - 1) + ")",
                            periodic=True,
                        )
                        for anion_3 in shell_cation_2.atoms:
                            if anion_3.resid not in anion_list:
                                anion_list.append(anion_3.resid)
                                cn_values[time_count][2] += 1
                                shell_anion_3 = nvt_run.select_atoms(
                                    "(type 17 and around 3 resid " + str(anion_3.resid) + ")",
                                    periodic=True,
                                )
                                for cation_4 in shell_anion_3:
                                    if cation_4.id not in cation_list:
                                        cation_list.append(cation_4.id)
                                        cn_values[time_count][3] += 1


def coord_shell_array(
    nvt_run: Universe,
    func: Callable,
    center_atoms: AtomGroup,
    distance_dict: Dict[str, float],
    select_dict: Dict[str, str],
    run_start: int,
    run_end: int,
):
    """
    Args:
        nvt_run: MDAnalysis Universe
        func: One of the neighbor statistical method (num_of_neighbor, num_of_neighbor_simple)
        center_atoms: Atom group of the center atoms.
        distance_dict (dict): A dict of coordination cutoff distance
            of the interested species.
        select_dict: A dictionary of species selection.
        run_start (int): Start time step.
        run_end (int): End time step.
    """
    num_array = func(nvt_run, center_atoms[0], distance_dict, select_dict, run_start, run_end)
    for atom in tqdm(center_atoms[1::]):
        this_atom = func(nvt_run, atom, distance_dict, select_dict, run_start, run_end)
        for kw in num_array.keys():
            num_array[kw] = np.concatenate((num_array.get(kw), this_atom.get(kw)), axis=0)
    return num_array


def write_out(li_pos: np.ndarray, selection: AtomGroup, element_id_dict: Dict[int, str], path: str):
    """

    Args:
        li_pos:
        selection:
        element_id_dict:
        path:

    Returns:

    """
    lines = list()
    lines.append(str(len(selection) + 1))
    lines.append("")
    lines.append("Li 0.0000000 0.0000000 0.0000000")
    box = selection.dimensions
    half_box = np.array([box[0], box[1], box[2]]) / 2
    for atom in selection:
        locs = list()
        for i in range(3):
            loc = atom.position[i] - li_pos[i]
            if loc > half_box[i]:
                loc = loc - box[i]
            elif loc < -half_box[i]:
                loc = loc + box[i]
            else:
                pass
            locs.append(loc)
        element_name = element_id_dict.get(int(atom.type))
        assert element_name is not None
        line = element_name + " " + " ".join(str(loc) for loc in locs)
        lines.append(line)
    with open(path, "w") as xyz_file:
        xyz_file.write("\n".join(lines))


def select_shell(
    select: Union[Dict[str, str], str], distance: Union[Dict[str, float], str], atom: Atom, kw: str
) -> str:
    """
    Select a group of atoms that is within a distance of an atom.

    Args:
        select:
        distance:
        atom:
        kw:

    Returns:

    """
    if isinstance(select, dict):
        species_selection = select[kw]
    else:
        species_selection = select
    assert species_selection is not None
    if isinstance(distance, dict):
        distance_str = str(distance[kw])
    else:
        distance_str = distance
    selection = "(" + species_selection + ") and (around " + distance_str + " index " + str(atom.id - 1) + ")"
    return selection
