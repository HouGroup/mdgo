# atom group counters
from mdgo.shell_functions import get_cation_anion_shells
import numpy as np

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


def count_dicts(dict_list):
    unique_dicts = []
    dict_counts = []
    for dic in dict_list:
        new = True
        for i, unique_dict in enumerate(unique_dicts):
            if dic == unique_dict:
                dict_counts[i] += 1
                new = False
                break
        if new:
            unique_dicts.append(dic)
            dict_counts.append(1)
    return zip(dict_counts, unique_dicts)