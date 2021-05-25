# atom group counters
from mdgo.shell_functions import get_cation_anion_shells
import numpy as np

"""
This file exists as a placeholder, it should eventually hold the counting utilities
that are currently implemented in shell_functions.

However for clarity of pull requests, I chose to temporarily implement those functions
in shell functions.

This will implement get_counts, get_pair_type, count_dicts
"""

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
#     return counts_by_speciation
