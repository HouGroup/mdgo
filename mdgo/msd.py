import MDAnalysis.analysis.msd as msd
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook


def total_msd(nvt_run, start, stop, select='all', msd_type='xyz', fft=True):
    msd_calculator = msd.EinsteinMSD(nvt_run, select=select, msd_type=msd_type, fft=fft)
    msd_calculator.run(start=start, stop=stop)
    total_array = msd_calculator.timeseries
    return total_array


def msd_states(coord_list, largest):
    lis = []
    for state in coord_list:
        for i, start in enumerate(state[:-1, :]):
            for j, end in enumerate(state[i+1:min((i+largest), state.shape[0]),
                                    :]):
                distance = np.square(np.linalg.norm(end-start))
                time = (j + 1) * 10
                lis.append(np.array([time, distance]))
    lis = np.array(lis)
    data = pd.DataFrame(data=lis[:, 1:], index=lis[:, 0])
    return data


def coords_oneLi(nvt_run, li_atom, select_dict, distance, run_start, run_end):
    trj_analysis = nvt_run.trajectory[run_start:run_end:]
    A_values = np.zeros(int(len(trj_analysis)))
    attach_list = []
    free_list = []
    current_state = -1
    current_coord = None
    next_coord = None
    coords = []
    for ts in trj_analysis:
        selection = "(" + select_dict["anion"] + ") and (around " \
                    + str(distance) + " index " \
                    + str(li_atom.id - 1) + ")"
        shell_anion = nvt_run.select_atoms(selection, periodic=True)
        next_state = 0
        if len(shell_anion) > 0:
            next_state = 1
        next_coord = ts[li_atom.id-1]

        if current_state != -1:
            coords.append(next_coord)
            current_coord = next_coord
            if next_state == current_state:
                pass
            else:
                if current_state:
                    attach_list.append(np.array(coords))
                else:
                    free_list.append(np.array(coords))
                current_state = next_state
                current_coord = next_coord
                coords = []
                coords.append(next_coord)
        else:
            current_state = next_state
            current_coord = next_coord
            coords.append(next_coord)
    if len(attach_list) == 0:
        attach_list.append(np.array(coords))
    return attach_list, free_list


def partial_msd(nvt_run, li_atoms, largest, select_dict, distance,
                run_start, run_end):
    free_datas = []
    attach_datas = []
    for i in tqdm_notebook(list(range(len(li_atoms)))):
        attach_coord, free_coord = coords_oneLi(nvt_run, li_atoms[i],
                                                select_dict, distance,
                                                run_start, run_end)
        if len(attach_coord) > 0:
            attach_data = msd_states(attach_coord, largest)
            attach_datas.append(attach_data)
        if len(free_coord) > 0:
            free_data = msd_states(free_coord, largest)
            free_datas.append(free_data)
    free_datas = pd.concat(free_datas, sort=False)
    free_datas = free_datas.groupby(free_datas.index).mean()
    attach_datas = pd.concat(attach_datas, sort=False)
    attach_datas = attach_datas.groupby(attach_datas.index).mean()
    return free_datas, attach_datas
