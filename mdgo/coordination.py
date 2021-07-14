# coding: utf-8
# Copyright (c) Tingzheng Hou.
# Distributed under the terms of the MIT License.

"""
This module implements functions for coordination analysis.
"""

from typing import Dict, List, Tuple, Union, Callable, Optional

import numpy as np
from tqdm.notebook import tqdm
from MDAnalysis import Universe, AtomGroup
from MDAnalysis.core.groups import Atom
from MDAnalysis.analysis.distances import distance_array
from scipy.signal import savgol_filter
from mdgo.util import atom_vec, angle


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
    Calculates a dictionary of distances between the ``center_atom`` and neighbor atoms.

    Args:
        nvt_run: An MDAnalysis ``Universe`` containing wrapped trajectory.
        center_atom: The center atom object.
        run_start: Start frame of analysis.
        run_end: End frame of analysis.
        species: The neighbor species in the select_dict.
        select_dict: A dictionary of atom species selection, where each atom species name is a key
            and the corresponding values are the selection language.
        distance: The neighbor cutoff distance.

    Returns:
        A dictionary of distance of neighbor atoms to the ``center_atom``.
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
    """Using the dictionary of neighbor distance ``trj``, finds the nearest neighbor ``sites`` that the center atom
    binds to, and calculates the ``frequency`` of hopping between each neighbor, and ``steps`` when each binding site
    exhibits the closest distance to the center atom.

    Args:
        trj: A dictionary of distances between center atom and neighbor atoms.
        time_step: The time step of the simulation in ps.
        binding_cutoff: Binding cutoff distance.
        hopping_cutoff: Detaching cutoff distance.
        smooth: The length of the smooth filter window. Default to 51.

    Returns:
        Returns an array of nearest neighbor ``sites`` (unique on each frame),
        the ``frequency`` of hopping between sites, and ``steps`` when each binding site
        exhibits the closest distance to the center atom.
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
    """Using the dictionary of neighbor distance ``trj``, finds the nearest neighbor ``sites`` that the ``center_atom``
    binds to, and calculates the ``frequency`` of hopping between each neighbor, and ``steps`` when each binding site
    exhibits the closest distance to the center atom.
    * Only hopping events with intermediate free state (no binded nearest neighbor) are counted.

    Args:
        trj: A dictionary of distances between center atom and neighbor atoms.
        time_step: The time step of the simulation in ps.
        binding_cutoff: Binding cutoff distance.
        hopping_cutoff: Detaching cutoff distance.
        smooth: The length of the smooth filter window. Default to 51.

    Returns:
        Returns an array of nearest neighbor ``sites`` (unique on each frame),
        the ``frequency`` of hopping between sites, and ``steps`` when each binding site
        exhibits the closest distance to the center atom.
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
    """Finds the frames when the center atom binds with the neighbor (binding) or hopping out (hopping)
    according to the dictionary of neighbor distance.

    Args:
        trj: A dictionary of distances between center atom and neighbor atoms.
        binding_cutoff: Binding cutoff distance.
        hopping_cutoff: Hopping out cutoff distance.
        smooth: The length of the smooth filter window. Default to 51.
        cool: The cool down frames between hopping in and hopping out. Default to 20.

    Returns:
        Two arrays of numberings of frames with hopping in and hopping out event, respectively.
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
    """Calculates the distance between the center atom and the neighbor atom
    in the checkpoint +/- lag time range.

    Args:
        nvt_run: An MDAnalysis ``Universe`` containing wrapped trajectory.
        center_atom: The center atom object.
        distance_dict: A dictionary of Cutoff distance of neighbor for each species.
        select_dict: A dictionary of atom species selection, where each atom species name is a key
            and the corresponding values are the selection language.
        run_start: Start frame of analysis.
        run_end: End frame of analysis.
        checkpoints: The frame numberings of interest to check for contiguous steps.
        lag: The range (+/- lag) of the contiguous steps. Default to 20.

    Returns:
        An array of distance between the center atom and the neighbor atoms
        in the checkpoint +/- lag time range.
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
    Calculates the heat map of the floating atom around the cluster. The coordinates are normalized to
    a cartesian coordinate system where the cluster_center_sites atom is the origin.

    Args:
        nvt_run: An MDAnalysis ``Universe`` containing wrapped trajectory.
        floating_atom: Floating atom species.
        cluster_center_sites: A list of nearest cluster center sites (atom id).
        cluster_terminal: The terminal atom species of the cluster (typically the binding site for the floating ion).
        cartesian_by_ref: Transformation matrix between cartesian and reference coordinate systems.
        run_start: Start frame of analysis.
        run_end: End frame of analysis.

    Returns:
        The coordinates of the floating ion around clusters normalized to the desired cartesian coordinate system.
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
    lag: int,
    binding_cutoff: float,
    hopping_cutoff: float,
    smooth: int,
    cool: int,
    binding_site: str,
    center_atom: str,
):
    """Calculates the coordination number evolution of species around ``center_atom`` as a function of time,
    the coordination numbers are averaged over all frames around events when the center_atom
    hopping to and hopping out from the ``binding_site``.

    Args:
        nvt_run: An MDAnalysis ``Universe`` containing wrapped trajectory.
        select_dict: A dictionary of atom species selection, where each atom species name is a key
            and the corresponding values are the selection language.
        in_list: A list to store the distances for hopping in events.
        out_list: A list to store the distances for hopping out events.
        distance_dict: A dict of coordination cutoff distance of the neighbor species.
        run_start: Start frame of analysis.
        run_end: End frame of analysis.
        lag: The frame range (+/- lag) for check evolution.
        binding_cutoff: Binding cutoff distance.
        hopping_cutoff: Hopping out cutoff distance.
        smooth: The length of the smooth filter window. Default to 51.
        cool: The cool down frames between binding and hopping out.
        binding_site: The binding site of binding and hopping out events.
        center_atom: The solvation shell center atom.
    """
    nvt_run = nvt_run
    center_atoms = nvt_run.select_atoms(select_dict.get(center_atom))
    for atom in tqdm(center_atoms[::]):
        neighbor_trj = neighbor_distance(
            nvt_run, atom, run_start + lag, run_end - lag, binding_site, select_dict, binding_cutoff
        )
        hopping_in, hopping_out = find_in_n_out(neighbor_trj, binding_cutoff, hopping_cutoff, smooth=smooth, cool=cool)
        if len(hopping_in) > 0:
            in_one = check_contiguous_steps(
                nvt_run,
                atom,
                distance_dict,
                select_dict,
                run_start,
                run_end,
                np.array(hopping_in) + lag,
                lag=lag,
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
                np.array(hopping_out) + lag,
                lag=lag,
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
    A helper function for calculating the heatmap. It applies the ``reflection``, ``rotation`` and ``inversion``
    symmetry operations to ``coords`` and take ``sample`` number of samples.

    Args:
        coords: An array of coordinates.
        reflection: A list of reflection symmetry operation matrix.
        rotation: A list of rotation symmetry operation matrix.
        inversion: A list of inversion symmetry operation matrix.
        sample: Number of samples to take from ``coords``.

    Returns:
        An array with ``sample`` number of coordinates.
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
    """Calculates the average position of a cluster. TODO: rewrite the method.

    Args:
        nvt_run: An MDAnalysis ``Universe`` containing wrapped trajectory.
        select_dict: A dictionary of atom species selection, where each atom species name is a key
            and the corresponding values are the selection language.
        run_start: Start frame of analysis.
        run_end: End frame of analysis.
        species: The species in the cluster.
        distance: The coordination cutoff distance.
        basis_vectors: The basis vector for normalizing the coordinates of the cluster atoms.
        cluster_center: Cluster center atom species.

    Returns:
        A array of coordinates of the cluster atoms.
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
) -> Dict[str, np.ndarray]:
    """Calculates the coordination number of each specified neighbor species and the total coordination number
    in the specified frame range.

    Args:
        nvt_run: An MDAnalysis ``Universe`` containing wrapped trajectory.
        center_atom: The solvation shell center atom.
        distance_dict: A dict of coordination cutoff distance of the neighbor species.
        select_dict: A dictionary of atom species selection, where each atom species name is a key
            and the corresponding values are the selection language.
        run_start: Start frame of analysis.
        run_end: End frame of analysis.
        write: Whether to writes out a series of desired solvation structures as ``*.xyz`` files.
        structure_code: An integer code representing the solvation structure to write out.
            For example, 221 is two species A, two species B and one species C.
        write_freq: Probability to write out files.
        write_path: Path to write out files.
        element_id_dict: a dict for mapping atom type id to element name.

    Returns:
        A diction containing the coordination number sequence of each specified neighbor species
        and the total coordination number sequence in the specified frame range .
    """
    time_count = 0
    trj_analysis = nvt_run.trajectory[run_start:run_end:]
    cn_values = dict()
    species = list(distance_dict.keys())
    for kw in species:
        cn_values[kw] = np.zeros(int(len(trj_analysis)))
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
                center_pos = ts[(int(center_atom.id) - 1)]
                center_type = element_id_dict.get(int(center_atom.type))
                path = write_path + str(center_atom.id) + "_" + str(int(ts.time)) + "_" + str(structure_code) + ".xyz"
                write_out(center_pos, center_type, structure, element_id_dict, path)
        time_count += 1
    return cn_values


def num_of_neighbor_simple(
    nvt_run: Universe,
    center_atom: Atom,
    distance_dict: Dict[str, float],
    select_dict: Dict[str, str],
    run_start: int,
    run_end: int,
) -> Dict[str, np.ndarray]:
    """Calculates solvation structure type (1 for SSIP, 2 for CIP and 3 for AGG) with respect to the ``enter_atom``
    in the specified frame range.

    Args:
        nvt_run: An MDAnalysis ``Universe`` containing wrapped trajectory.
        center_atom: The solvation shell center atom.
        distance_dict: A dict of coordination cutoff distance of the neighbor species.
        select_dict: A dictionary of atom species selection, where each atom species name is a key
            and the corresponding values are the selection language.
        run_start: Start frame of analysis.
        run_end: End frame of analysis.

    Returns:
        A dict with "total" as the key and an array of the solvation structure type in the specified frame range
        as the value.
    """

    time_count = 0
    trj_analysis = nvt_run.trajectory[run_start:run_end:]
    center_selection = "same type as " + str(center_atom.id - 1)
    assert len(distance_dict) == 1, "Please only specify the counter-ion species in the distance_dict"
    species = list(distance_dict.keys())[0]
    cn_values = np.zeros(int(len(trj_analysis)))
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


def angular_dist_of_neighbor(
    nvt_run: Universe,
    center_atom: Atom,
    center_c: str,
    neighbor_a: str,
    neighbor_b: str,
    select_dict: Dict[str, str],
    distance_dict: Dict[str, float],
    run_start: int,
    run_end: int,
    cip: bool = True,
):
    """
    Calculates the angle of atoms a-c-b in the specified frames.

    Args:
        nvt_run: An MDAnalysis ``Universe`` containing wrapped trajectory.
        center_atom: The center atom object.
        center_c: The center species in the select_dict.
        neighbor_a: The neighbor species in the select_dict.
        neighbor_b: The neighbor species in the select_dict.
        select_dict: A dictionary of atom species selection, where each atom species name is a key
            and the corresponding values are the selection language.
        distance_dict: A dict of coordination cutoff distance of the neighbor species.
        run_start: Start frame of analysis.
        run_end: End frame of analysis.
        cip: Only includes contact ion pair structures with only one `a` and one `c` atoms.

    Returns:
        A array of angles of a-c-b occurrence in the specified frames.
    """
    acb_angle = list()
    trj_analysis = nvt_run.trajectory[run_start:run_end:]
    for ts in trj_analysis:
        a_selection = select_shell(select_dict, distance_dict, center_atom, neighbor_a)
        a_group = nvt_run.select_atoms(a_selection, periodic=True)
        a_num = len(a_group)
        if a_num == 0:
            continue
        elif a_num == 1:
            c_selection = select_shell(select_dict, distance_dict, a_group.atoms[0], center_c)
            c_atoms = nvt_run.select_atoms(c_selection, periodic=True)
            shell_species_len = len(c_atoms) - 1
            if shell_species_len == 0:
                shell_type = "cip"
            else:
                shell_type = "agg"
        else:
            shell_type = "agg"
        if shell_type == "agg" and cip:
            continue
        else:
            c_pos = center_atom.position
            for a_atom in a_group.atoms:
                a_pos = a_atom.position
                b_selection = select_shell(select_dict, distance_dict, center_atom, neighbor_b)
                b_group = nvt_run.select_atoms(b_selection, periodic=True)
                for b_atom in b_group.atoms:
                    b_pos = b_atom.position
                    theta = angle(a_pos, c_pos, b_pos)
                    acb_angle.append(theta)
    return np.array(acb_angle)


def num_of_neighbor_specific(
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
        nvt_run: An MDAnalysis ``Universe`` containing wrapped trajectory.
        center_atom: The center atom object.
        species_list:
        select_dict: A dictionary of atom species selection, where each atom species name is a key
            and the corresponding values are the selection language.
        distance_dict: A dict of coordination cutoff distance of the neighbor species.
        run_start: Start frame of analysis.
        run_end: End frame of analysis.

    Returns:

    """
    time_count = 0
    trj_analysis = nvt_run.trajectory[run_start:run_end:]
    cip_step = list()
    ssip_step = list()
    agg_step = list()
    cn_values = dict()
    for kw in species_list:
        cn_values[kw] = np.zeros(int(len(trj_analysis)))
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
def full_solvation_shell(
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
        nvt_run: An MDAnalysis ``Universe`` containing wrapped trajectory.
        center_atom: The center atom object.
        species:
        select_dict: A dictionary of atom species selection, where each atom species name is a key
            and the corresponding values are the selection language.
        distance: The coordination cutoff distance.
        run_start: Start frame of analysis.
        run_end: End frame of analysis.

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
    A helper function to analyze the coordination number/structure of every atoms in an ``AtomGroup`` using the
    specified function.

    Args:
        nvt_run: An MDAnalysis ``Universe`` containing wrapped trajectory.
        func: One of the neighbor statistical method (num_of_neighbor, num_of_neighbor_simple)
        center_atoms: Atom group of the center atoms.
        distance_dict: A dictionary of coordination cutoff distance of the neighbor species.
        select_dict: A dictionary of atom species selection, where each atom species name is a key
            and the corresponding values are the selection language.
        run_start: Start frame of analysis.
        run_end: End frame of analysis.

    Returns:
        A diction containing the coordination number sequence of each specified neighbor species
        and the total coordination number sequence in the specified frame range.
    """
    num_array = func(nvt_run, center_atoms[0], distance_dict, select_dict, run_start, run_end)
    for atom in tqdm(center_atoms[1::]):
        this_atom = func(nvt_run, atom, distance_dict, select_dict, run_start, run_end)
        for kw in num_array.keys():
            num_array[kw] = np.concatenate((num_array.get(kw), this_atom.get(kw)), axis=0)
    return num_array


def write_out(
    center_pos: np.ndarray, center_type: str, neighbors: AtomGroup, element_id_dict: Dict[int, str], path: str
):
    """
    Helper function for solvation structure coordinates write out.

    Args:
        center_pos: The coordinates of the center atom in the frame.
        center_type: The element type of the center atom in the frame.
        neighbors: The neighbor AtomGroup.
        element_id_dict: A dictionary for mapping atom type id to element from the mass information.
        path: The path to write out ``*.xyz`` file.
    """
    lines = list()
    lines.append(str(len(neighbors) + 1))
    lines.append("")
    lines.append("{} 0.0000000 0.0000000 0.0000000".format(center_type))
    box = neighbors.dimensions
    half_box = np.array([box[0], box[1], box[2]]) / 2
    for atom in neighbors:
        locs = list()
        for i in range(3):
            loc = atom.position[i] - center_pos[i]
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
    select: Union[Dict[str, str], str], distance: Union[Dict[str, float], str], center_atom: Atom, kw: str
) -> str:
    """
    Select a group of atoms that is within a distance of an ``center_atom``.

    Args:
        select: A selection string of neighbors or a dictionary of atom species selection, where each atom
            species name is a key and the corresponding values are the selection string.
        distance: A neighbor cutoff distance or a dict of cutoff distances of neighbor species.
        center_atom: The solvation shell center ``Atom`` object
        kw: The key for the select and/or distance dictionary if applicable.

    Returns:
        A selection string specifying the neighbor species within a distance of the ``center_atom``.
    """
    if isinstance(select, dict):
        species_selection = select[kw]
        if species_selection is None:
            raise ValueError("Species specified does not match entries in the select dict.")
    else:
        species_selection = select
    if isinstance(distance, dict):
        distance_value = distance[kw]
        if distance_value is None:
            raise ValueError("Species specified does not match entries in the distance dict.")
        distance_str = str(distance_value)
    else:
        distance_str = distance
    selection = "(" + species_selection + ") and (around " + distance_str + " index " + str(center_atom.id - 1) + ")"
    return selection
