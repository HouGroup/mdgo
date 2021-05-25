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

# def get_detailed_cation_speciation(self, timestep):
#     """
#     This function should be move to a utility folder and wrapped, right now
#     its not very useful. Currently broken
#
#     Returns:
#
#     """
#     self.u_wrapped.trajectory[timestep]
#     solvation_shell_speciation = [get_counts(get_radial_shell(self.u, cation, 3))
#                                   for cation in self.cations]
#     counts_by_speciation = count_dicts(solvation_shell_speciation)
#     print('before: ', len(solvation_shell_speciation))
#     print('after: ', len(counts_by_speciation))
#     print('after: ', sum(counts_by_speciation.keys()))
#     return counts_by_speciation
