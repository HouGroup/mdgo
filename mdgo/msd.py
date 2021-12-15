# coding: utf-8
# Copyright (c) Tingzheng Hou.
# Distributed under the terms of the MIT License.

"""
This module implements functions for calculating meen square displacement (MSD).
"""
from typing import List, Dict, Tuple, Union, Optional

try:
    import MDAnalysis.analysis.msd as mda_msd
except ImportError:
    mda_msd = None

import numpy as np
from tqdm.notebook import trange

from MDAnalysis import Universe, AtomGroup
from MDAnalysis.core.groups import Atom

__author__ = "Tingzheng Hou"
__version__ = "1.0"
__maintainer__ = "Tingzheng Hou"
__email__ = "tingzheng_hou@berkeley.edu"
__date__ = "Feb 9, 2021"


def total_msd(
    nvt_run: Universe, start: int, stop: int, select: str = "all", msd_type: str = "xyz", fft: bool = True
) -> np.ndarray:
    """
    From a MD Universe, calculates the MSD array of a group of atoms defined by select.

    Args:
        nvt_run: An MDAnalysis ``Universe`` containing unwrapped trajectory.
        start: Start frame of analysis.
        stop: End frame of analysis.
        select: A selection string. Defaults to “all” in which case all atoms are selected.
        msd_type: Desired dimensions to be included in the MSD. Defaults to ‘xyz’.
        fft: Whether to use FFT to accelerate the calculation. Default to True.

    Warning:
        To correctly compute the MSD using this analysis module, you must supply coordinates in the
        unwrapped convention. That is, when atoms pass the periodic boundary, they must not be
        wrapped back into the primary simulation cell.

    Note:
         The built in FFT method is under construction.

    Returns:
        An array of calculated MSD.
    """
    if mda_msd is not None:
        msd_calculator = mda_msd.EinsteinMSD(nvt_run, select=select, msd_type=msd_type, fft=fft)
        msd_calculator.run(start=start, stop=stop)
        try:
            total_array = msd_calculator.timeseries
        except AttributeError:
            total_array = msd_calculator.results.timeseries
        return total_array

    if fft:
        raise ValueError("Warning! MDAnalysis version too low, fft not supported. PleaseUse fft=False instead")
    return _total_msd(nvt_run, start, stop, select=select)


def _total_msd(nvt_run: Universe, run_start: int, run_end: int, select: str = "all") -> np.ndarray:
    """
    A native MSD calculator. Uses the conventional algorithm. TODO: add xyz dimension selection; use fft method

    Args:
        nvt_run: An MDAnalysis ``Universe`` containing unwrapped trajectory.
        select: A selection string. Defaults to “all” in which case all atoms are selected.
        run_start: Start frame of analysis.
        run_end: End frame of analysis.

    Returns:
        An array of calculated MSD.
    """
    trj_analysis = nvt_run.trajectory[run_start:run_end:]
    li_atoms = nvt_run.select_atoms(select)
    all_list = []
    for li_atom in li_atoms:
        coords = []
        for ts in trj_analysis:
            current_coord = ts[li_atom.id - 1]
            coords.append(current_coord)
        all_list.append(np.array(coords))
    total_array = msd_from_frags(all_list, run_end - run_start - 1)
    return total_array


def msd_from_frags(coord_list: List[np.ndarray], largest: int) -> np.ndarray:
    """
    Calculates the MSD using a list of fragments of trajectory with the conventional algorithm.

    Args:
        coord_list: A list of trajectory.
        largest: The largest interval of time frame for calculating MSD.

    Returns:
        The MSD series.
    """
    msd_dict: Dict[Union[int, np.integer], np.ndarray] = {}
    for state in coord_list:
        n_frames = state.shape[0]
        lag_times = np.arange(1, min(n_frames, largest))
        for lag in lag_times:
            disp = state[:-lag, :] - state[lag:, :]
            sqdist = np.square(disp).sum(axis=-1)
            if lag in msd_dict.keys():
                msd_dict[lag] = np.concatenate((msd_dict[lag], sqdist), axis=0)
            else:
                msd_dict[lag] = sqdist
    timeseries = []
    time_range = len(msd_dict) + 1
    msds_by_state = np.zeros(time_range)
    for kw in range(1, time_range):
        msds = msd_dict.get(kw)
        assert msds is not None
        msds_by_state[kw] = msds.mean()
        timeseries.append(msds_by_state[kw])
    timeseries = np.array(timeseries)
    return timeseries


def states_coord_array(
    nvt_run: Universe,
    atom: Atom,
    select_dict: Dict[str, str],
    distance: float,
    run_start: int,
    run_end: int,
    binding_site: str = "anion",
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Cuts the trajectory of an atom into fragments. Each fragment contains consecutive timesteps of coordinates
    of the atom in either attached or free state. The Attached state is when the atom coordinates with the
    ``binding_site`` species (distance < ``distance``), and vice versa for the free state.
    TODO: check if need wrapped trj

    Args:
        nvt_run: An MDAnalysis ``Universe`` containing unwrapped trajectory.
        atom: The Atom object to analyze.
        select_dict: A dictionary of atom species selection, where each atom species name is a key
            and the corresponding values are the selection language.
        distance: The coordination cutoff distance.
        run_start: Start frame of analysis.
        run_end: End frame of analysis.
        binding_site: The species the ``atom`` coordinates to.

    Returns:
        Two list of coordinates arrays containing where each coordinates array is a consecutive trajectory fragment
        of atom in a certain state. One for the attached state, the other for the free state.
    """
    trj_analysis = nvt_run.trajectory[run_start:run_end:]
    attach_list = []
    free_list = []
    coords = []
    prev_state = None
    prev_coord = None
    for ts in trj_analysis:
        selection = (
            "(" + select_dict[binding_site] + ") and (around " + str(distance) + " index " + str(atom.id - 1) + ")"
        )
        shell = nvt_run.select_atoms(selection, periodic=True)
        current_state = 0
        if len(shell) > 0:
            current_state = 1
        current_coord = ts[atom.id - 1]

        if prev_state:
            if current_state == prev_state:
                coords.append(current_coord)
                prev_coord = current_coord
            else:
                if len(coords) > 1:
                    if prev_state:
                        attach_list.append(np.array(coords))
                    else:
                        free_list.append(np.array(coords))
                prev_state = current_state
                coords = []
                coords.append(prev_coord)
                coords.append(current_coord)
                prev_coord = current_coord
        else:
            coords.append(current_coord)
            prev_state = current_state
            prev_coord = current_coord

    if len(coords) > 1:
        if prev_state:
            attach_list.append(np.array(coords))
        else:
            free_list.append(np.array(coords))

    return attach_list, free_list


def partial_msd(
    nvt_run: Universe,
    atoms: AtomGroup,
    largest: int,
    select_dict: Dict[str, str],
    distance: float,
    run_start: int,
    run_end: int,
    binding_site: str = "anion",
) -> Tuple[Optional[List[np.ndarray]], Optional[List[np.ndarray]]]:
    """Calculates the mean square displacement (MSD) of the ``atoms`` according to coordination states.
    The returned ``free_data`` include the MSD when ``atoms`` are not coordinated to ``binding_site``.
    The ``attach_data`` includes the MSD of ``atoms`` are not coordinated to ``binding_site``.

    Args:
        nvt_run: An MDAnalysis ``Universe`` containing unwrapped trajectory.
        atoms: The AtomGroup for
        largest: The largest interval of time frame for calculating MSD.
        select_dict: A dictionary of atom species selection, where each atom species name is a key
            and the corresponding values are the selection language.
        distance: The coordination cutoff distance between
        run_start: Start frame of analysis.
        run_end: End frame of analysis.
        binding_site: The species the ``atoms`` coordinates to.

    Returns:
        Two arrays of MSD in the trajectory
    """
    free_coords = []
    attach_coords = []
    for i in trange(len(atoms)):
        attach_coord, free_coord = states_coord_array(
            nvt_run, atoms[i], select_dict, distance, run_start, run_end, binding_site=binding_site
        )
        attach_coords.extend(attach_coord)
        free_coords.extend(free_coord)
    attach_data = None
    free_data = None
    if len(attach_coords) > 0:
        attach_data = msd_from_frags(attach_coords, largest)
    if len(free_coords) > 0:
        free_data = msd_from_frags(free_coords, largest)
    return free_data, attach_data
