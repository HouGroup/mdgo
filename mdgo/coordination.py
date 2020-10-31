import numpy as np
from tqdm import tqdm_notebook
from MDAnalysis.analysis.distances import distance_array
from scipy.signal import savgol_filter
from itertools import groupby


def trajectory(nvt_run, li_atom, run_start, run_end, species, selection_dict, distance):
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
            previous_site = None
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

    grouped_sites = [list(v) for k, v in groupby(sites, key=lambda x: x != 0)
                     if k != 0]
    change = (np.diff(sites) != 0).sum() - len(grouped_sites) + 1
    frequency = change/(time_span * time_step)
    return sites, frequency, steps


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
                                 distance, run_start, run_end):

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
                        + str(distance) + " index "\
                        + str(li_atom.id - 1) + ")"
            shell = nvt_run.select_atoms(selection, periodic=True)
            # for each atom in shell, create/add to dictionary
            # (key = atom id, value = list of values for step function)
            for _ in shell.atoms:
                cn_values[kw][time_count] += 1
                cn_values["total"][time_count] += 10**digit_of_species
            digit_of_species = digit_of_species - 1
        time_count += 1
    return cn_values


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
    for li in tqdm_notebook(li_atoms[1::]):
        this_li = func(nvt_run, li, species, select_dict,
                       distance, run_start, run_end)
        num_array = np.concatenate((num_array, this_li), axis=0)
    return num_array

