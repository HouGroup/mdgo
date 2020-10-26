import MDAnalysis
from MDAnalysis.analysis import contacts
from MDAnalysis.analysis.rdf import InterRDF
#from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from statsmodels.tsa.stattools import acovf
from scipy.optimize import curve_fit
from tqdm import tqdm_notebook
from mdgo.conductivity import calc_cond, conductivity_calculator
from mdgo.coordination import coord_shell_array, num_of_neighbor_one_li
from mdgo.msd import total_msd, partial_msd


class MdRun:

    def __init__(self, data_dir, wrapped_dir, unwrapped_dir, nvt_start,
                 time_step, name, select_dict, c_to_a_ratio=1):
        self.wrapped_run = MDAnalysis.Universe(data_dir,
                                               wrapped_dir,
                                               format="LAMMPS")
        self.unwrapped_run = MDAnalysis.Universe(data_dir,
                                                 unwrapped_dir,
                                                 format="LAMMPS")
        self.nvt_start = nvt_start
        self.time_step = time_step
        self.name = name
        self.select_dict = select_dict
        self.nvt_steps = self.wrapped_run.trajectory.n_frames
        self.time_array = [i * 10 for i in range(self.nvt_steps)]
        self.c_to_a_ratio = c_to_a_ratio
        self.num_li = \
            len(self.wrapped_run.select_atoms(self.select_dict["cation"]))
        self.cond_array = self.get_cond_array()
        self.init_x = self.get_init_dimension()[0]
        self.init_y = self.get_init_dimension()[1]
        self.init_z = self.get_init_dimension()[2]
        self.init_v = self.init_x * self.init_y * self.init_z
        self.nvt_x = self.get_nvt_dimension()[0]
        self.nvt_y = self.get_nvt_dimension()[1]
        self.nvt_z = self.get_nvt_dimension()[2]
        self.nvt_v = self.nvt_x * self.nvt_y * self.nvt_z
        self.c_to_a_ratio = c_to_a_ratio

    def get_init_dimension(self):
        return self.wrapped_run.dimensions

    def get_nvt_dimension(self):
        return self.wrapped_run.trajectory[-1].dimensions

    def get_msd(self):
        return

    def get_cond_array(self):
        nvt_run = self.unwrapped_run
        cations = nvt_run.select_atoms(self.select_dict["cation"])
        anions = nvt_run.select_atoms(self.select_dict["anion"])
        cond_array = calc_cond(nvt_run, anions, cations, self.nvt_start,
                               self.c_to_a_ratio)
        return cond_array

    def plot_cond_array(self, start, end, *runs):
        colors = ["g", "r", "c", "m", "y", "k"]
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        line0 = ax.loglog(self.time_array[start:end],
                          self.cond_array[start:end], color="b", lw=2,
                          label=self.name)
        for i, run in enumerate(runs):
            line = ax.loglog(run.time_array[start:end],
                             run.cond_array[start:end], color=colors[i], lw=2,
                             label=run.name)
        ax.set_ylabel('MSD (A^2)')
        ax.set_xlabel('Time (ps)')
        ax.set_ylim([10, 1000000])
        ax.set_xlim([100, 500000000])
        ax.legend()
        fig.show()

    def get_conductivity(self, start, end):
        conductivity_calculator(self.time_array, self.cond_array,
                                self.nvt_v, self.name, start, end)
        return None

    def coord_num_array_one_species(self, species, distance,
                                    run_start, run_end):
        nvt_run = self.wrapped_run
        li_atoms = nvt_run.select_atoms(self.select_dict["cation"])
        num_array = coord_shell_array(nvt_run, num_of_neighbor_one_li,
                                      li_atoms, species, self.select_dict,
                                      distance, run_start, run_end)
        return num_array

    def coordination_one_species(self, species, distance, run_start, run_end):
        num_array = self.coord_num_array_one_species(species, distance,
                                                     run_start, run_end)
        shell_component, shell_count = np.unique(num_array.flatten(),
                                                 return_counts=True)
        combined = np.vstack((shell_component, shell_count)).T

        item_name = "Num of " + species + " within " + str(distance) + " " \
                    + "\u212B"
        item_list = []
        percent_list = []
        for i in range(len(combined)):
            item_list.append(str(int(combined[i, 0])))
            percent_list.append(str("%.4f" % (combined[i, 1] /
                                              combined[:, 1].sum() * 100))
                                + '%')
        df_dict = {item_name: item_list, 'Percentage': percent_list}
        df = pd.DataFrame(df_dict)
        return df

    def get_msd_all(self, start=None, stop=None):
        msd_array = total_msd(self.unwrapped_run, start=start, stop=stop,
                              select=self.select_dict["cation"])
        return msd_array

    def get_msd_partial(self, distance, run_start, run_end, largest=1000):
        nvt_run = self.unwrapped_run
        li_atoms = nvt_run.select_atoms(self.select_dict["cation"])
        free_array, attach_array = partial_msd(nvt_run, li_atoms, largest,
                                               self.select_dict, distance,
                                               run_start, run_end)
        free_array.columns = ['msd']
        attach_array.columns = ['msd']
        return free_array, attach_array

    def get_d(self, msd_array, start, stop):
        A2 = 1e-20
        ps = 1e-12
        R = 8.314
        T = 298.15
        F2 = 96485 * 96485
        c = (self.num_li / (self.nvt_v * (1e-30))) / (6.022*1e23)
        SmTomScm = 10
        DtoSigma = c*F2/(R * T)

        d = (msd_array[start] - msd_array[stop]) \
            / (start-stop) / self.time_step / 6 * A2 / ps
        sigma = d * DtoSigma * SmTomScm

        print("Diffusivity of all Li: ", d, "m^2/s")
        print("Conductivity of all Li: ", sigma, "mS/cm")

    def get_partial_d(self, msd_array, start, stop, percentage):
        if isinstance(msd_array, pd.DataFrame):
            msd_array = msd_array["msd"].to_numpy()
        A2 = 1e-20
        ps = 1e-12
        R = 8.314
        T = 298.15
        F2 = 96485 * 96485
        c = (self.num_li / (self.nvt_v * (1e-30))) / (6.022*1e23)
        SmTomScm = 10
        DtoSigma = c*F2/(R * T)

        d = percentage * (msd_array[start] - msd_array[stop]) \
            / (start-stop) / self.time_step / 6 * A2 / ps
        sigma = d * DtoSigma * SmTomScm

        print("Diffusivity of partial Li: ", d, "m^2/s")
        print("Conductivity of partial Li: ", sigma, "mS/cm")

