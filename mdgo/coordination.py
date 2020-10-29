import numpy as np
from tqdm import tqdm_notebook


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

