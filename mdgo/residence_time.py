import re
import numpy as np
import matplotlib.pyplot as plt
from mdgo.coordination import num_of_neighbor_one_li
from statsmodels.tsa.stattools import acovf
from scipy.optimize import curve_fit
from tqdm import tqdm_notebook


# Calculate ACF
def calc_acf(a_values):
    acfs = []
    for atom_id, neighbors in a_values.items():
        atom_id = int(re.search(r'\d+', atom_id).group())
        acfs.append(acovf(neighbors, demean=False, unbiased=True, fft=True))
    return (acfs)


def exponential_func(x, a, b, c):
    return a * np.exp(-b * x) + c


def calc_neigh_corr(nvt_run, species_list, select_dict, distance, timestep,
                    run_start, run_end):
    # Set up times array
    times = []
    step = 0
    li_atoms = nvt_run.select_atoms(species_list["cation"])
    for ts in nvt_run.trajectory[run_start:run_end]:
        times.append(step * timestep)
        step += 1
    times = np.array(times)

    acf_avg = dict()
    for kw in species_list:
        acf_all = []
        for li in tqdm_notebook(li_atoms[::]):
            adjacency_matrix = num_of_neighbor_one_li(nvt_run, li, kw,
                                                      select_dict, distance,
                                                      run_start, run_end)
            acfs = calc_acf(adjacency_matrix)
            [acf_all.append(acf) for acf in acfs]
        acf_avg[kw] = np.mean(acf_all, axis=0)
    return times, acf_avg


def fit_residence_time(times, species_list, acf_avg_dict, cutoff_time):
    acf_avg_norm = dict()
    popt = dict()
    pcov = dict()
    tau = dict()

    # Exponential fit of solvent-Li ACF
    for kw in species_list:
        acf_avg_norm[kw] = acf_avg_dict[kw] / acf_avg_dict[kw][0]
        popt[kw], pcov[kw] = curve_fit(exponential_func, times[:cutoff_time],
                                       acf_avg_norm[kw][:cutoff_time],
                                       p0=(1, 1e-4, 0))
        tau[kw] = 1 / popt[kw][1]  # ps

    # Plot ACFs
    colors = ["b", "g", "r", "c", "m", "y"]
    line_styles = ['-', '--', '-.', ':']
    for i, kw in enumerate(species_list):
        plt.plot(times, acf_avg_norm[kw], label=kw, color=colors[i])
        plt.plot(np.linspace(0, 10000, 1000),
                 exponential_func(np.linspace(0, 10000, 1000), *popt[kw]),
                 line_styles[i], color='k', label=kw + ' Fit')

    plt.xlabel('Time (ps)')
    plt.legend()
    plt.ylabel('Neighbor Autocorrelation Function')
    plt.ylim(0, 1)
    plt.xlim(0, 10000)
    plt.show()

    return tau
