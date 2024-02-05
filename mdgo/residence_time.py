# Copyright (c) Tingzheng Hou.
# Distributed under the terms of the MIT License.

"""This module calculates species correlation lifetime (residence time)."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from statsmodels.tsa.stattools import acovf
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from MDAnalysis import Universe
    from MDAnalysis.core.groups import Atom

__author__ = "Kara Fong, Tingzheng Hou"
__version__ = "0.3.0"
__maintainer__ = "Tingzheng Hou"
__email__ = "tingzheng_hou@berkeley.edu"
__date__ = "Jul 19, 2021"


def neighbors_one_atom(
    nvt_run: Universe,
    center_atom: Atom,
    species: str,
    select_dict: dict[str, str],
    distance: float,
    run_start: int,
    run_end: int,
) -> dict[str, np.ndarray]:
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
    for time_count, _ts in enumerate(nvt_run.trajectory[run_start:run_end:]):
        if species in select_dict:
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
    return bool_values


def calc_acf(a_values: dict[str, np.ndarray]) -> list[np.ndarray]:
    """
    Calculate auto-correlation function (ACF).

    Args:
        a_values: A dict of adjacency matrix with neighbor atom id as keys and arrays
        of adjacent boolean (0/1) as values.

    Returns:
        A list of auto-correlation functions for each neighbor species.
    """
    acfs = []
    for neighbors in a_values.values():  # for _atom_id, neighbors in a_values.items():
        #  atom_id_numeric = int(re.search(r"\d+", _atom_id).group())
        acfs.append(acovf(neighbors, demean=False, unbiased=True, fft=True))
    return acfs


def exponential_func(
    x: float | np.floating | np.ndarray,
    a: float | np.floating | np.ndarray,
    b: float | np.floating | np.ndarray,
    c: float | np.floating | np.ndarray,
) -> np.floating | np.ndarray:
    """
    An exponential decay function.

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
    distance_dict: dict[str, float],
    select_dict: dict[str, str],
    time_step: float,
    run_start: int,
    run_end: int,
    center_atom: str = "cation",
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Calculates the neighbor auto-correlation function (ACF)
    of selected species around center atom.

    Args:
        nvt_run: An MDAnalysis ``Universe``.
        distance_dict: A dict of coordination cutoff distance of the neighbor species.
        select_dict: A dictionary of atom species selection.
        time_step: Timestep between each frame, in ps.
        run_start: Start frame of analysis.
        run_end: End frame of analysis.
        center_atom: The center atom to calculate the ACF for. Default to "cation".

    Returns:
        A tuple containing the time series, and a dict of acf of neighbor species.
    """
    # Set up times array
    times = []
    center_atoms = nvt_run.select_atoms(select_dict[center_atom])
    for step, _ts in enumerate(nvt_run.trajectory[run_start:run_end]):
        times.append(step * time_step)
    times = np.array(times)

    acf_avg = {}
    for kw in distance_dict:
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
            acf_all.extend(list(acfs))
        acf_avg[kw] = np.mean(acf_all, axis=0)
    return times, acf_avg


def fit_residence_time(
    times: np.ndarray,
    acf_avg_dict: dict[str, np.ndarray],
    cutoff_time: int,
    time_step: float,
    save_curve: str | bool = False,
) -> dict[str, np.floating]:
    """
    Use the ACF to fit the residence time (Exponential decay constant).
    TODO: allow defining the residence time according to a threshold value of the decay.

    Args:
        times: A time series.
        acf_avg_dict: A dict containing the ACFs of the species.
        cutoff_time: Fitting cutoff time.
        time_step: The time step between each frame, in ps.
        save_curve: Whether to save the curve as a csv file for post-processing.
                Default to False.

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
        fitted_x = np.linspace(0, cutoff_time * time_step, cutoff_time)
        fitted_y = exponential_func(np.linspace(0, cutoff_time * time_step, cutoff_time), *popt[kw])
        save_decay = np.vstack(
            (
                times[:cutoff_time],
                acf_avg_norm[kw][:cutoff_time],
                fitted_x,
                fitted_y,
            )
        )
        if save_curve:
            if save_curve is True:
                np.savetxt(f"decay{i}.csv", save_decay.T, delimiter=",")
            elif os.path.exists(str(save_curve)):
                np.savetxt(str(save_curve) + f"decay{i}.csv", save_decay.T, delimiter=",")
            else:
                raise ValueError("Please specify a bool or a path in string.")
        plt.plot(
            fitted_x,
            fitted_y,
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
