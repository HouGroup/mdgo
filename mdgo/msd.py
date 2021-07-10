# coding: utf-8
# Copyright (c) Tingzheng Hou.
# Distributed under the terms of the MIT License.

"""
This module implements functions for calculating meen square displacement (MSD).
"""

try:
    import MDAnalysis.analysis.msd as mda_msd
except ImportError:
    mda_msd = None


import numpy as np
from tqdm.notebook import trange

from MDAnalysis import Universe, AtomGroup
from MDAnalysis.core.groups import Atom

from typing import List, Dict, Tuple, Union, Optional

__author__ = "Tingzheng Hou"
__version__ = "1.0"
__maintainer__ = "Tingzheng Hou"
__email__ = "tingzheng_hou@berkeley.edu"
__date__ = "Feb 9, 2021"


def total_msd(
    nvt_run: Universe, start: int, stop: int, select: str = "all", msd_type: str = "xyz", fft: bool = True
) -> np.ndarray:
    """

    Args:
        nvt_run:
        start:
        stop:
        select:
        msd_type:
        fft:

    Returns:

    """
    if mda_msd is not None:
        msd_calculator = mda_msd.EinsteinMSD(nvt_run, select=select, msd_type=msd_type, fft=fft)
        msd_calculator.run(start=start, stop=stop)
        total_array = msd_calculator.timeseries
        return total_array
    else:
        if fft:
            print("Warning! MDAnalysis version too low, fft not supported. Use conventional instead")
        return _total_msd(nvt_run, select, start, stop)


def _total_msd(nvt_run: Universe, select: str, run_start: int, run_end: int) -> np.ndarray:
    trj_analysis = nvt_run.trajectory[run_start:run_end:]
    li_atoms = nvt_run.select_atoms(select)
    all_list = list()
    for li_atom in li_atoms:
        coords = list()
        for ts in trj_analysis:
            current_coord = ts[li_atom.id - 1]
            coords.append(current_coord)
        all_list.append(np.array(coords))
    total_array = msd_states(all_list, run_end - run_start - 1)
    return total_array


def msd_states(coord_list: List[np.ndarray], largest: int) -> np.ndarray:
    """

    Args:
        coord_list:
        largest:

    Returns:

    """
    msd_dict: Dict[Union[int, np.integer], np.ndarray] = dict()
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
    timeseries = list()
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
    nvt_run: Universe, li_atom: Atom, select_dict: Dict[str, str], distance: float, run_start: int, run_end: int
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """

    Args:
        nvt_run:
        li_atom:
        select_dict:
        distance:
        run_start:
        run_end:

    Returns:

    """
    trj_analysis = nvt_run.trajectory[run_start:run_end:]
    attach_list = list()
    free_list = list()
    coords = list()
    prev_state = None
    prev_coord = None
    for ts in trj_analysis:
        selection = (
            "(" + select_dict["anion"] + ") and (around " + str(distance) + " index " + str(li_atom.id - 1) + ")"
        )
        shell_anion = nvt_run.select_atoms(selection, periodic=True)
        current_state = 0
        if len(shell_anion) > 0:
            current_state = 1
        current_coord = ts[li_atom.id - 1]

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
                coords = list()
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
) -> Tuple[Optional[List[np.ndarray]], Optional[List[np.ndarray]]]:
    """

    Args:
        nvt_run:
        atoms:
        largest:
        select_dict:
        distance:
        run_start:
        run_end:

    Returns:

    """
    free_coords = list()
    attach_coords = list()
    for i in trange(len(atoms)):
        attach_coord, free_coord = states_coord_array(nvt_run, atoms[i], select_dict, distance, run_start, run_end)
        attach_coords.extend(attach_coord)
        free_coords.extend(free_coord)
    attach_data = None
    free_data = None
    if len(attach_coords) > 0:
        attach_data = msd_states(attach_coords, largest)
    if len(free_coords) > 0:
        free_data = msd_states(free_coords, largest)
    return free_data, attach_data
