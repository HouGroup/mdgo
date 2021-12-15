# coding: utf-8
# Copyright (c) Tingzheng Hou.
# Distributed under the terms of the MIT License.

"""
This module calculates species correlation lifetime (residence time).
"""
from typing import List, Dict, Union, Tuple

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acovf
from scipy.optimize import curve_fit
from tqdm.notebook import tqdm

from MDAnalysis import Universe
from MDAnalysis.core.groups import Atom

__author__ = "Kara Fong, Tingzheng Hou"
__version__ = "1.0"
__maintainer__ = "Tingzheng Hou"
__email__ = "tingzheng_hou@berkeley.edu"
__date__ = "Feb 9, 2021"


def neighbors_one_atom(
    nvt_run: Universe,
    center_atom: Atom,
    species: str,
    select_dict: Dict[str, str],
    distance: float,
    run_start: int,
    run_end: int,
) -> Dict[str, np.ndarray]:
    """
    Create adjacency matrix for one center atom.

    Args:
        nvt_run: An MDAnalysis ``Universe``.
        center_atom: The center atom object.
        species: The neighbor species in the select_dict.
        select_dict: A dictionary of atom species selection, where each atom species name is a key
            and the corresponding values are the selection language.
        distance: The neighbor cutoff distance.
        run_start: Start frame of analysis.
        run_end: End frame of analysis.

    Returns:
        A neighbor dict with neighbor atom id as keys and arrays of adjacent boolean (0/1) as values.
    """
    bool_values = {}
    time_count = 0
    for ts in nvt_run.trajectory[run_start:run_end:]:
        if species in select_dict.keys():
            selection = (
                "("
                + select_dict[species]
                + ") and (around "
                + str(distance)
                + " index "
                + str(center_atom.id - 1)
                + ")"
            )
            shell = nvt_run.select_atoms(selection)
        else:
            raise ValueError("Invalid species selection")
        for atom in shell.atoms:
            if str(atom.id) not in bool_values:
                bool_values[str(atom.id)] = np.zeros(int((run_end - run_start) / 1))
            bool_values[str(atom.id)][time_count] = 1
        time_count += 1
    return bool_values


def calc_acf(a_values: Dict[str, np.ndarray]) -> List[np.ndarray]:
    """
    Calculate auto-correlation function (ACF)

    Args:
        a_values: A dict of adjacency matrix with neighbor atom id as keys and arrays
        of adjacent boolean (0/1) as values.

    Returns:
        A list of auto-correlation functions for each neighbor species.
    """
    acfs = []
    for atom_id, neighbors in a_values.items():
        #  atom_id_numeric = int(re.search(r"\d+", atom_id).group())
        acfs.append(acovf(neighbors, demean=False, unbiased=True, fft=True))
    return acfs


def exponential_func(
    x: Union[float, np.floating, np.ndarray],
    a: Union[float, np.floating, np.ndarray],
    b: Union[float, np.floating, np.ndarray],
    c: Union[float, np.floating, np.ndarray],
) -> Union[np.floating, np.ndarray]:
    """
    An exponential decay function

    Args:
        x: Independent variable.
        a: Initial quantity.
        b: Exponential decay constant.
        c: Constant.

    Returns:
        The acf
    """
    return a * np.exp(-b * x) + c


def calc_neigh_corr(
    nvt_run: Universe,
    distance_dict: Dict[str, float],
    select_dict: Dict[str, str],
    time_step: float,
    run_start: int,
    run_end: int,
    center_atom: str = "cation",
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Calculates the neighbor auto-correlation function (ACF)
    of selected species around center atom.

    Args:
        nvt_run: An MDAnalysis ``Universe``.
        distance_dict:
        select_dict:
        time_step:
        run_start: Start frame of analysis.
        run_end: End frame of analysis.
        center_atom: The center atom to calculate the ACF for. Default to "cation".

    Returns:
        A tuple containing the time series, and a dict of acf of neighbor species.
    """
    # Set up times array
    times = []
    step = 0
    center_atoms = nvt_run.select_atoms(select_dict[center_atom])
    for ts in nvt_run.trajectory[run_start:run_end]:
        times.append(step * time_step)
        step += 1
    times = np.array(times)

    acf_avg = {}
    for kw in distance_dict.keys():
        acf_all = []
        for atom in tqdm(center_atoms[::]):
            distance = distance_dict.get(kw)
            assert distance is not None
            adjacency_matrix = neighbors_one_atom(
                nvt_run,
                atom,
                kw,
                select_dict,
                distance,
                run_start,
                run_end,
            )
            acfs = calc_acf(adjacency_matrix)
            for acf in acfs:
                acf_all.append(acf)
        acf_avg[kw] = np.mean(acf_all, axis=0)
    return times, acf_avg


def fit_residence_time(
    times: np.ndarray,
    acf_avg_dict: Dict[str, np.ndarray],
    cutoff_time: int,
    time_step: float,
) -> Dict[str, np.floating]:
    """
    Use the ACF to fit the residence time (Exponential decay constant).
    TODO: allow defining the residence time according to a threshold value of the decay

    Args:
        times: A time series.
        acf_avg_dict: A dict containing the ACFs of the species.
        cutoff_time: Fitting cutoff time.
        time_step: The time step between each frame, in ps.

    Returns:
        A dict containing residence time of each species
    """
    acf_avg_norm = {}
    popt = {}
    pcov = {}
    tau = {}
    species_list = list(acf_avg_dict.keys())

    # Exponential fit of solvent-Li ACF
    for kw in species_list:
        acf_avg_norm[kw] = acf_avg_dict[kw] / acf_avg_dict[kw][0]
        popt[kw], pcov[kw] = curve_fit(
            exponential_func,
            times[:cutoff_time],
            acf_avg_norm[kw][:cutoff_time],
            p0=(1, 1e-4, 0),
        )
        tau[kw] = 1 / popt[kw][1]  # ps

    # Plot ACFs
    colors = ["b", "g", "r", "c", "m", "y"]
    line_styles = ["-", "--", "-.", ":"]
    for i, kw in enumerate(species_list):
        plt.plot(times, acf_avg_norm[kw], label=kw, color=colors[i])
        plt.plot(
            np.linspace(0, cutoff_time * time_step, cutoff_time),
            exponential_func(np.linspace(0, cutoff_time * time_step, cutoff_time), *popt[kw]),
            line_styles[i],
            color="k",
            label=kw + " Fit",
        )

    plt.xlabel("Time (ps)")
    plt.legend()
    plt.ylabel("Neighbor Auto-correlation Function")
    plt.ylim(0, 1)
    plt.xlim(0, cutoff_time * time_step)
    plt.show()

    return tau
