import MDAnalysis.analysis.msd as msd
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook


def total_msd(nvt_run, start, stop, select='all', msd_type='xyz', fft=True):
    msd_calculator = msd.EinsteinMSD(nvt_run, select=select,
                                     msd_type=msd_type, fft=fft)
    msd_calculator.run(start=start, stop=stop)
    total_array = msd_calculator.timeseries
    return total_array


def msd_states(coord_list, largest):
    lis = []
    for state in coord_list:
        n_frame = state.shape[0]
        for i, start in enumerate(state[:-1, :]):
            for j, end in enumerate(state[i+1:min((i+largest), n_frame), :]):
                square_distance = np.square(end-start).sum()
                time = (j + 1) * 10
                lis.append(np.array([time, square_distance]))
    lis = np.array(lis)
    data = pd.DataFrame(data=lis)
    data.columns = ["time", "msd"]
    return data


def msd_states_new(coord_list, largest):
    msd_dict = dict()
    for state in coord_list:
        n_frames = state.shape[0]
        lagtimes = np.arange(1, min(n_frames, largest))
        for lag in lagtimes:
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
    free_datas = list()
    attach_datas = list()
    for i in tqdm_notebook(list(range(len(li_atoms)))):
        attach_coord, free_coord = states_coord_array(nvt_run, li_atoms[i],
                                                      select_dict, distance,
                                                      run_start, run_end)
        if len(attach_coord) > 0:
            attach_data = msd_states(attach_coord, largest)
            attach_datas.append(attach_data)
        if len(free_coord) > 0:
            free_data = msd_states(free_coord, largest)
            free_datas.append(free_data)
    free_datas = pd.concat(free_datas, sort=False)
    free_datas = free_datas.groupby("time").mean()
    attach_datas = pd.concat(attach_datas, sort=False)
    attach_datas = attach_datas.groupby("time").mean()
    return free_datas, attach_datas


def partial_msd_new(nvt_run, li_atoms, largest, select_dict, distance,
                    run_start, run_end):
    free_coords = list()
    attach_coords = list()
    for i in tqdm_notebook(list(range(len(li_atoms)))):
        attach_coord, free_coord = states_coord_array(nvt_run, li_atoms[i],
                                                      select_dict, distance,
                                                      run_start, run_end)
        attach_coords.extend(attach_coord)
        free_coords.extend(free_coord)
    if len(attach_coords) > 0:
        attach_data = msd_states_new(attach_coords, largest)
    else:
        attach_data = None
    if len(free_coords) > 0:
        free_data = msd_states_new(free_coords, largest)
    else:
        free_data = None
    return free_data, attach_data
