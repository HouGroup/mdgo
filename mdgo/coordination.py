# coding: utf-8
# Copyright (c) Tingzheng Hou.
# Distributed under the terms of the MIT License.

import numpy as np
from tqdm import tqdm_notebook
from MDAnalysis.analysis.distances import distance_array
from scipy.signal import savgol_filter
from mdgo.util import atom_vec
# from itertools import groupby

__author__ = "Tingzheng Hou"
__version__ = "1.0"
__maintainer__ = "Tingzheng Hou"
__email__ = "tingzheng_hou@berkeley.edu"
__date__ = "Feb 9, 2021"


def trajectory(nvt_run, li_atom, run_start, run_end, species, selection_dict,
               distance):
    A_values = {}
    time_count = 0
    trj_analysis = nvt_run.trajectory[run_start:run_end:]
    if species not in list(selection_dict):
        print('Invalid species selection')
        return None
    for ts in trj_analysis:
        selection = "(" + selection_dict[species] + ") and (around " \
                    + str(distance) + " index " \
                    + str(li_atom.id - 1) + ")"
        shell = nvt_run.select_atoms(selection, periodic=True)
        for atom in shell.atoms:
            if str(atom.id) not in A_values:
                A_values[str(atom.id)] = np.full(run_end-run_start, 100.)
        time_count += 1
    #print(A_values.keys())
    time_count = 0
    for ts in trj_analysis:
        for atomid in A_values.keys():
            dist = distance_array(ts[li_atom.id-1], ts[(int(atomid)-1)], ts.dimensions)
            A_values[atomid][time_count] = dist
        time_count += 1
    return A_values


def find_nearest(trj, time_step, distance, hopping_cutoff, smooth=51):
    time_span = len(list(trj.values())[0])
    for kw in list(trj):
        trj[kw] = savgol_filter(trj.get(kw), smooth, 2)
    site_distance = [100 for _ in range(time_span)]
    sites = [0 for _ in range(time_span)]
    sites[0] = min(trj, key=lambda k: trj[k][0])
    site_distance[0] = trj.get(sites[0])[0]
    for time in range(1, time_span):
        if sites[time - 1] == 0:
            old_site_distance = 100
        else:
            old_site_distance = trj.get(sites[time - 1])[time]
        if old_site_distance > hopping_cutoff:
            new_site = min(trj, key=lambda k: trj[k][time])
            new_site_distance = trj.get(new_site)[time]
            if new_site_distance > distance:
                site_distance[time] = 100
            else:
                sites[time] = new_site
                site_distance[time] = new_site_distance
        else:
            sites[time] = sites[time-1]
            site_distance[time] = old_site_distance
    sites = [int(i) for i in sites]
    sites_and_distance_array = np.array([[sites[i], site_distance[i]]
                                         for i in range(len(sites))])
    steps = []
    closest_step = 0
    previous_site = sites_and_distance_array[0][0]
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
                steps.append(closest_step)
                closest_step = i
                previous_site = site
    if previous_site is not None:
        steps.append(closest_step)
    # change = len(steps) - 1
    change = (np.diff([i for i in sites if i != 0]) != 0).sum()
    frequency = change/(time_span * time_step)
    return sites, frequency, steps


def heat_map(nvt_run, li_atom, sites, dist_to_center, bind_atom_type,
             cartesian_by_ref, run_start, run_end):
    trj_analysis = nvt_run.trajectory[run_start:run_end:]
    coordinates = []
    for i, ts in enumerate(trj_analysis):
        if sites[i] == 0:
            pass
        else:
            center_atom = nvt_run.select_atoms("index " + str(sites[i] - 1))[0]
            selection = "(" + bind_atom_type + ") and " + "(around " \
                        + str(dist_to_center) + " index " \
                        + str(center_atom.id - 1) + ")"
            bind_atoms = nvt_run.select_atoms(selection, periodic=True)
            distances = distance_array(ts[li_atom.id - 1], bind_atoms.positions,
                                       ts.dimensions)
            idx = np.argpartition(distances[0], 3)
            vertex_atoms = bind_atoms[idx[:3]]
            vector_li = atom_vec(li_atom, center_atom, ts.dimensions)
            vector_a = atom_vec(vertex_atoms[0], center_atom, ts.dimensions)
            vector_b = atom_vec(vertex_atoms[1], center_atom, ts.dimensions)
            vector_c = atom_vec(vertex_atoms[2], center_atom, ts.dimensions)
            basis_abc = np.transpose([vector_a, vector_b, vector_c])
            abc_li = np.linalg.solve(basis_abc, vector_li)
            unit_x = np.linalg.norm(cartesian_by_ref[0, 0] * vector_a
                                    + cartesian_by_ref[0, 1] * vector_b
                                    + cartesian_by_ref[0, 2] * vector_c)
            unit_y = np.linalg.norm(cartesian_by_ref[1, 0] * vector_a
                                    + cartesian_by_ref[1, 1] * vector_b
                                    + cartesian_by_ref[1, 2] * vector_c)
            unit_z = np.linalg.norm(cartesian_by_ref[2, 0] * vector_a
                                    + cartesian_by_ref[2, 1] * vector_b
                                    + cartesian_by_ref[2, 2] * vector_c)
            vector_x = cartesian_by_ref[0] / unit_x
            vector_y = cartesian_by_ref[1] / unit_y
            vector_z = cartesian_by_ref[2] / unit_z
            basis_xyz = np.transpose([vector_x, vector_y, vector_z])
            xyz_li = np.linalg.solve(basis_xyz, abc_li)
            coordinates.append(xyz_li)
    return np.array(coordinates)


def get_full_coords(coords, reflection=None, rotation=None, inversion=None,
                    sample=None):
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


def cluster_coordinates(nvt_run, select_dict, run_start, run_end, species,
                        distance, basis_vectors=None, cluster_center="center"):
    trj_analysis = nvt_run.trajectory[run_start:run_end:]
    cluster_center = nvt_run.select_atoms(select_dict[cluster_center],
                                          periodic=True)[0]
    selection = "(" + " or ".join([s for s in species]) \
                + ") and (around " + str(distance) + " index "\
                    + str(cluster_center.id - 1) + ")"
    print(selection)
    shell = nvt_run.select_atoms(selection, periodic=True)
    cluster = []
    for atom in shell:
        coord_list = []
        for ts in trj_analysis:
            coord_list.append(atom.position)
        cluster.append(np.mean(np.array(coord_list), axis=0))
    cluster = np.array(cluster)
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
        cluster_norm = np.linalg.solve(basis_xyz, cluster.T).T
        cluster_norm = cluster_norm - np.mean(cluster_norm, axis=0)
        return cluster_norm
    else:
        return cluster


def num_of_neighbor_one_li(nvt_run, li_atom, species, select_dict,
                           distance, run_start, run_end):

    time_count = 0
    trj_analysis = nvt_run.trajectory[run_start:run_end:]
    if species in select_dict.keys():
        cn_values = np.zeros(int(len(trj_analysis)))
    else:
        print('Invalid species selection')
        return None
    for ts in trj_analysis:
        selection = "(" + select_dict[species] + ") and (around "\
                    + str(distance) + " index "\
                    + str(li_atom.id - 1) + ")"
        shell = nvt_run.select_atoms(selection, periodic=True)
        for _ in shell.atoms:
            cn_values[time_count] += 1
        time_count += 1
    return cn_values


def num_of_neighbor_one_li_multi(nvt_run, li_atom, species_list, select_dict,
                                 distances, run_start, run_end, write=False,
                                 structure_code=None, write_freq=0,
                                 write_path=None, element_id_dict=None):

    time_count = 0
    trj_analysis = nvt_run.trajectory[run_start:run_end:]
    cn_values = dict()
    for kw in species_list:
        if kw in select_dict.keys():
            cn_values[kw] = np.zeros(int(len(trj_analysis)))
        else:
            print('Invalid species selection')
            return None
    cn_values["total"] = np.zeros(int(len(trj_analysis)))
    for ts in trj_analysis:
        digit_of_species = len(species_list) - 1
        for kw in species_list:
            selection = "(" + select_dict[kw] + ") and (around "\
                        + str(distances[kw]) + " index "\
                        + str(li_atom.id - 1) + ")"
            shell = nvt_run.select_atoms(selection, periodic=True)
            # for each atom in shell, create/add to dictionary
            # (key = atom id, value = list of values for step function)
            for _ in shell.atoms:
                cn_values[kw][time_count] += 1
                cn_values["total"][time_count] += 10**digit_of_species
            digit_of_species = digit_of_species - 1
        if write and cn_values["total"][time_count] == structure_code:
            a = np.random.random()
            if a > 1 - write_freq:
                print("writing")
                species = ' or '.join("(same resid as (" + select_dict[kw]
                                      + " and around " + str(distances[kw])
                                      + " index " + str(li_atom.id - 1)
                                      + "))" for kw in species_list)
                species = "((" + species + ")and not " \
                          + select_dict["cation"] + ")"
                structure = nvt_run.select_atoms(species,
                                                 periodic=True)
                li_pos = ts[(int(li_atom.id)-1)]
                path = write_path + str(li_atom.id) + "_" + str(int(ts.time)) \
                    + "_" + str(structure_code) + ".xyz"
                write_out(li_pos, structure, element_id_dict, path)
        time_count += 1
    return cn_values


def num_of_neighbor_one_li_simple(nvt_run, li_atom, species, select_dict,
                                  distance, run_start, run_end):

    time_count = 0
    trj_analysis = nvt_run.trajectory[run_start:run_end:]
    if species in select_dict.keys():
        cn_values = np.zeros(int(len(trj_analysis)))
    else:
        print('Invalid species selection')
        return None
    for ts in trj_analysis:
        selection = "(" + select_dict[species] + ") and (around "\
                    + str(distance) + " index "\
                    + str(li_atom.id - 1) + ")"
        shell = nvt_run.select_atoms(selection, periodic=True)
        shell_len = len(shell)
        if shell_len == 0:
            cn_values[time_count] = 1
        elif shell_len == 1:
            selection_species = "(" + select_dict["cation"] + " and around " + \
                                str(distance) + " index " + \
                                str(shell.atoms[0].id - 1) + ")"
            shell_species = nvt_run.select_atoms(selection_species,
                                                 periodic=True)
            shell_species_len = len(shell_species) - 1
            if shell_species_len == 0:
                cn_values[time_count] = 2
            else:
                cn_values[time_count] = 3
        else:
            cn_values[time_count] = 3
        time_count += 1
    return cn_values


def num_of_neighbor_one_li_simple_extra(nvt_run, li_atom, species, select_dict,
                                        distance, run_start, run_end):

    time_count = 0
    emc_angle = list()
    ec_angle = list()
    trj_analysis = nvt_run.trajectory[run_start:run_end:]
    if species in select_dict.keys():
        cn_values = np.zeros(int(len(trj_analysis)))
    else:
        print('Invalid species selection')
        return None
    for ts in trj_analysis:
        selection = "(" + select_dict[species] + ") and (around "\
                    + str(distance) + " index "\
                    + str(li_atom.id - 1) + ")"
        shell = nvt_run.select_atoms(selection, periodic=True)
        shell_len = len(shell)
        if shell_len == 0:
            cn_values[time_count] = 1
        elif shell_len == 1:
            selection_species = "(" + select_dict["cation"] + " and around " + \
                                str(distance) + " index " + \
                                str(shell.atoms[0].id - 1) + ")"
            shell_species = nvt_run.select_atoms(selection_species,
                                                 periodic=True)
            shell_species_len = len(shell_species) - 1
            if shell_species_len == 0:
                cn_values[time_count] = 2
                li_pos = li_atom.position
                p_pos = shell.atoms[0].position
                ec_select = "(" + select_dict["EC"] + ") and (around "\
                    + str(3) + " index "\
                    + str(li_atom.id - 1) + ")"
                emc_select = "(" + select_dict["EMC"] + ") and (around "\
                    + str(3) + " index "\
                    + str(li_atom.id - 1) + ")"
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


def num_of_neighbor_one_li_simple_extra_two(nvt_run, li_atom, species_list,
                                            select_dict, distances, run_start,
                                            run_end):
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
            print('Invalid species selection')
            return None
    cn_values["total"] = np.zeros(int(len(trj_analysis)))
    for ts in trj_analysis:
        digit_of_species = len(species_list) - 1
        for kw in species_list:
            selection = "(" + select_dict[kw] + ") and (around "\
                        + str(distances[kw]) + " index "\
                        + str(li_atom.id - 1) + ")"
            shell = nvt_run.select_atoms(selection, periodic=True)
            # for each atom in shell, create/add to dictionary
            # (key = atom id, value = list of values for step function)
            for _ in shell.atoms:
                cn_values[kw][time_count] += 1
                cn_values["total"][time_count] += 10**digit_of_species
            digit_of_species = digit_of_species - 1

        selection = "(" + select_dict["anion"] + ") and (around "\
                    + str(distances["anion"]) + " index "\
                    + str(li_atom.id - 1) + ")"
        shell = nvt_run.select_atoms(selection, periodic=True)
        shell_len = len(shell)
        if shell_len == 0:
            ssip_step.append(time_count)
        elif shell_len == 1:
            selection_species = "(" + select_dict["cation"] + " and around " + \
                                str(distances["anion"]) + " index " + \
                                str(shell.atoms[0].id - 1) + ")"
            shell_species = nvt_run.select_atoms(selection_species,
                                                 periodic=True)
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


def angle(a, b, c):
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)


# Depth-first traversal
def num_of_neighbor_one_li_complex(nvt_run, li_atom, species, selection_dict,
                                   distance, run_start, run_end):
    time_count = 0
    trj_analysis = nvt_run.trajectory[run_start:run_end:]
    A_values = np.zeros((int(len(trj_analysis)), 4))
    for ts in trj_analysis:
        li_list = [li_atom.id]
        p_list = []
        shell = nvt_run.select_atoms("(" + selection_dict[species] +
                                     " and around " + str(distance)
                                     + " index " + str(li_atom.id - 1) + ")",
                                     periodic=True)
        for atom in shell.atoms:
            if atom.resid not in p_list:
                p_list.append(atom.resid)
                A_values[time_count][0] += 1
                shell_pf6 = nvt_run.select_atoms("(type 17 and around 3 resid "
                                                 + str(atom.resid) + ")",
                                                 periodic=True)
                for li in shell_pf6:
                    if li.id not in li_list:
                        li_list.append(li.id)
                        A_values[time_count][1] += 1
                        shell_li = nvt_run.select_atoms("(type 15 and around 3 index " + str(li.id - 1) + ")",
                                                        periodic=True)
                        for p in shell_li.atoms:
                            if p.resid not in p_list:
                                p_list.append(p.resid)
                                A_values[time_count][2] += 1
                                shell_p = nvt_run.select_atoms("(type 17 and around 3 resid " + str(p.resid) + ")",
                                                               periodic=True)
                                for moreli in shell_pf6:
                                    if moreli.id not in li_list:
                                        li_list.append(moreli.id)
                                        A_values[time_count][3] += 1


def coord_shell_array(nvt_run, func, li_atoms, species, select_dict,
                      distance, run_start, run_end):
    """
    Args:
        nvt_run: MDAnalysis Universe
        func: One of the neighbor statistical method
        li_atoms: atom group of the Li atoms.
    """
    num_array = func(nvt_run, li_atoms[0], species, select_dict,
                     distance, run_start, run_end)
    if isinstance(num_array, np.ndarray):
        for li in tqdm_notebook(li_atoms[1::]):
            this_li = func(nvt_run, li, species, select_dict,
                           distance, run_start, run_end)
            num_array = np.concatenate((num_array, this_li), axis=0)
    else:
        for li in tqdm_notebook(li_atoms[1::]):
            this_li = func(nvt_run, li, species, select_dict,
                           distance, run_start, run_end)
            for kw in num_array.keys():
                num_array[kw] = np.concatenate((num_array.get(kw),
                                                this_li.get(kw)), axis=0)
    return num_array


def write_out(li_pos, selection, element_id_dict, path):
    lines = list()
    lines.append(str(len(selection) + 1))
    lines.append("")
    lines.append("Li 0.0000000 0.0000000 0.0000000")
    box = selection.dimensions
    half_box = np.array([box[0], box[1], box[2]])/2
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
        line = element_id_dict.get(int(atom.type)) \
            + ' ' + ' '.join(str(loc) for loc in locs)
        lines.append(line)
    with open(path, "w") as xyz_file:
        xyz_file.write("\n".join(lines))
