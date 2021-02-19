# coding: utf-8
# Copyright (c) Tingzheng Hou.
# Distributed under the terms of the MIT License.

try:
    import MDAnalysis.analysis.msd as msd
except ImportError:
    msd = None


import numpy as np
from tqdm import tqdm_notebook

__author__ = "Tingzheng Hou"
__version__ = "1.0"
__maintainer__ = "Tingzheng Hou"
__email__ = "tingzheng_hou@berkeley.edu"
__date__ = "Feb 9, 2021"


def total_msd(nvt_run, start, stop, select='all', msd_type='xyz', fft=True):
    if msd is not None:
        msd_calculator = msd.EinsteinMSD(nvt_run, select=select,
                                         msd_type=msd_type, fft=fft)
        msd_calculator.run(start=start, stop=stop)
        total_array = msd_calculator.timeseries
        return total_array
    else:
        if fft:
            print(
                "Warning! MDAnalysis version too low, fft not supported. "
                "Use conventional instead"
            )
        return _total_msd(nvt_run, select, start, stop)


def _total_msd(nvt_run, select, run_start, run_end):
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


def msd_states(coord_list, largest):
    msd_dict = dict()
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
        msds_by_state[kw] = msd_dict.get(kw).mean()
        timeseries.append(msds_by_state[kw])
    timeseries = np.array(timeseries)
    return timeseries


def states_coord_array(nvt_run, li_atom, select_dict, distance,
                       run_start, run_end):
    trj_analysis = nvt_run.trajectory[run_start:run_end:]
    attach_list = list()
    free_list = list()
    coords = list()
    prev_state = None
    prev_coord = None
    for ts in trj_analysis:
        selection = "(" + select_dict["anion"] + ") and (around " \
                    + str(distance) + " index " \
                    + str(li_atom.id - 1) + ")"
        shell_anion = nvt_run.select_atoms(selection, periodic=True)
        current_state = 0
        if len(shell_anion) > 0:
            current_state = 1
        current_coord = ts[li_atom.id-1]

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


def partial_msd(nvt_run, li_atoms, largest, select_dict, distance,
                run_start, run_end):
    free_coords = list()
    attach_coords = list()
    for i in tqdm_notebook(list(range(len(li_atoms)))):
        attach_coord, free_coord = states_coord_array(nvt_run, li_atoms[i],
                                                      select_dict, distance,
                                                      run_start, run_end)
        attach_coords.extend(attach_coord)
        free_coords.extend(free_coord)
    attach_data = None
    free_data = None
    if len(attach_coords) > 0:
        attach_data = msd_states(attach_coords, largest)
    if len(free_coords) > 0:
        free_data = msd_states(free_coords, largest)
    return free_data, attach_data


def msd_by_length(coord_list):
    msd_dict = dict()
    for state in coord_list:
        n_frames = state.shape[0]
        disp = state[0, :] - state[-1, :]
        sqdist = np.square(disp).sum()
        if n_frames in list(msd_dict):
            msd = msd_dict.get(n_frames)
            msd_dict[n_frames].append(sqdist)
        else:
            msd_dict[n_frames] = [sqdist]
    timespan = list(msd_dict)
    timespan.sort()
    timeseries = list()
    for time in timespan:
        timeseries.append(np.array([time, np.mean(msd_dict.get(time))]))
    timeseries = np.array(timeseries)
    return timeseries


def special_msd(nvt_run, li_atoms, select_dict, distance, run_start, run_end):
    free_coords = list()
    attach_coords = list()
    for i in tqdm_notebook(list(range(len(li_atoms)))):
        attach_coord, free_coord = states_coord_array(nvt_run, li_atoms[i],
                                                      select_dict, distance,
                                                      run_start, run_end)
        attach_coords.extend(attach_coord)
        free_coords.extend(free_coord)
    attach_data = None
    free_data = None
    if len(attach_coords) > 0:
        attach_data = msd_by_length(attach_coords)
    if len(free_coords) > 0:
        free_data = msd_by_length(free_coords)
    return free_data, attach_data
