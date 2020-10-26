import MDAnalysis
from MDAnalysis.analysis import contacts
from MDAnalysis.analysis.rdf import InterRDF
#from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import re
from statsmodels.tsa.stattools import acovf
from scipy.optimize import curve_fit
from tqdm import tqdm_notebook
from mdgo.conductivity import calc_cond, conductivity_calculator


class MdRun:

    def __init__(self, data_dir, wrapped_dir, unwrapped_dir,
                 cation_select, anion_select, nvt_start, time_step, name):
        self.wrapped_run = MDAnalysis.Universe(data_dir,
                                               wrapped_dir,
                                               format="LAMMPS")
        self.unwrapped_run = MDAnalysis.Universe(data_dir,
                                                 unwrapped_dir,
                                                 format="LAMMPS")
        self.cation_select = cation_select
        self.anion_select = anion_select
        self.nvt_start = nvt_start
        self.time_step = time_step
        self.name = name
        self.nvt_steps = self.wrapped_run.trajectory.n_frames
        self.time_array = [i * 10 for i in range(self.nvt_steps)]
        self.cond_array = self.get_cond_array()
        self.init_x = self.get_init_dimension()[0]
        self.init_y = self.get_init_dimension()[1]
        self.init_z = self.get_init_dimension()[2]
        self.init_v = self.init_x * self.init_y * self.init_z
        self.nvt_x = self.get_nvt_dimension()[0]
        self.nvt_y = self.get_nvt_dimension()[1]
        self.nvt_z = self.get_nvt_dimension()[2]
        self.nvt_v = self.nvt_x * self.nvt_y * self.nvt_z

    def get_init_dimension(self):
        return self.wrapped_run.dimensions

    def get_nvt_dimension(self):
        return self.wrapped_run.trajectory[-1].dimensions

    def get_msd(self):
        return

    def get_cond_array(self):
        nvt_run = self.unwrapped_run
        cations = nvt_run.select_atoms(self.cation_select)
        anions = nvt_run.select_atoms(self.anion_select)
        cond_array = calc_cond(nvt_run, anions, cations, self.nvt_start)
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
        return






