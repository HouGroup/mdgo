# coding: utf-8
# Copyright (c) Tingzheng Hou.
# Distributed under the terms of the MIT License.

"""
This module implements two core class MdRun and MdJob
for molecular dynamics simulation analysis and job setup.
"""

import MDAnalysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymatgen.io.lammps.data import LammpsData
from MDAnalysis.analysis.distances import distance_array
from MDAnalysis.lib.distances import capped_distance
from tqdm.notebook import tqdm
from mdgo.util import (
    mass_to_name,
    assign_name,
    assign_resname,
    res_dict_from_lammpsdata,
    res_dict_from_select_dict,
    res_dict_from_datafile,
    select_dict_from_resname,
)
from mdgo.conductivity import calc_cond_msd, conductivity_calculator
from mdgo.coordination import (
    coord_shell_array,
    num_of_neighbor_one_li,
    num_of_neighbor_one_li_simple,
    trajectory,
    find_nearest,
    find_nearest_free_only,
    process_evol,
    heat_map,
    get_full_coords,
)
from mdgo.msd import total_msd, partial_msd, special_msd
from mdgo.residence_time import calc_neigh_corr, fit_residence_time

__author__ = "Tingzheng Hou"
__version__ = "1.0"
__maintainer__ = "Tingzheng Hou"
__email__ = "tingzheng_hou@berkeley.edu"
__date__ = "Feb 9, 2021"


class MdRun:
    """
    A core class for MD results analysis.
    """

    def __init__(
        self,
        lammps_data,
        wrapped_run,
        unwrapped_run,
        nvt_start,
        time_step,
        name,
        select_dict=None,
        res_dict=None,
        cation_name="cation",
        anion_name="anion",
        cation_charge=1,
        anion_charge=-1,
        temperature=298.5,
        cond=True,
    ):
        """
        Base constructor. This is a low level constructor designed to work with
         parsed data (mda.universe) or other bridging objects (CombinedData). Not
        recommended to use directly.

        Args:
            lammps_data (LammpsData or CombinedData): LammpsData object.
            wrapped_run (MDAnalysis.Universe): The Universe object of wrapped trajectory.
            unwrapped_run (MDAnalysis.Universe): The Universe object of unwrapped trajectory.
            nvt_start (int): NVT start time step.
            time_step (int or float): LAMMPS timestep.
            name (str): Name of the MD run.
            select_dict (dict): A dictionary of atom species, where each atom species name is a key
                    and the corresponding values are the selection language. This dict is intended for
                    analyzing interested atoms.
            res_dict (dict): A dictionary of resnames, where each resname is a key
                    and the corresponding values are the selection language. This dict is intended for
                    analyzing interested residues (ions/molecules).
            cation_name: Name of cation. Default to "cation".
            anion_name: Name of anion. Default to "anion".
            cation_charge: Charge of cation. Default to 1.
            anion_charge: Charge of anion. Default to 1.
            temperature: Temperature of the MD run. Default to 298.15.
            cond (bool): Whether to calculate conductivity MSD. Default to True.
        """

        self.wrapped_run = wrapped_run
        self.unwrapped_run = unwrapped_run
        self.nvt_start = nvt_start
        self.time_step = time_step
        self.temp = temperature
        self.name = name
        self.data = lammps_data
        self.element_id_dict = mass_to_name(self.data.masses)
        assign_name(self.wrapped_run, self.element_id_dict)
        assign_name(self.unwrapped_run, self.element_id_dict)
        self.select_dict = select_dict
        self.res_dict = res_dict
        if self.select_dict is None and self.res_dict is None:
            self.res_dict = res_dict_from_lammpsdata(self.data)
        if self.res_dict is None:
            self.res_dict = res_dict_from_select_dict(self.wrapped_run, self.select_dict)
        assign_resname(self.wrapped_run, self.res_dict)
        assign_resname(self.unwrapped_run, self.res_dict)
        if self.select_dict is None:
            self.select_dict = select_dict_from_resname(self.wrapped_run)
        self.nvt_steps = self.wrapped_run.trajectory.n_frames
        self.time_array = [i * self.time_step for i in range(self.nvt_steps)]
        self.cation_name = cation_name
        self.anion_name = anion_name
        self.cation_charge = cation_charge
        self.anion_charge = anion_charge
        self.cations = self.wrapped_run.select_atoms(self.select_dict.get("cation"))
        self.anion_center = self.wrapped_run.select_atoms(self.select_dict.get("anion"))
        self.anions = self.anion_center.residues.atoms
        self.num_cation = len(self.cations)
        if cond:
            self.cond_array = self.get_cond_array()
        else:
            self.cond_array = None
        self.init_x = self.get_init_dimension()[0]
        self.init_y = self.get_init_dimension()[1]
        self.init_z = self.get_init_dimension()[2]
        self.init_v = self.init_x * self.init_y * self.init_z
        self.nvt_x = self.get_nvt_dimension()[0]
        self.nvt_y = self.get_nvt_dimension()[1]
        self.nvt_z = self.get_nvt_dimension()[2]
        self.nvt_v = self.nvt_x * self.nvt_y * self.nvt_z
        gas_constant = 8.314
        temp = 298.15
        faraday_constant_2 = 96485 * 96485
        self.c = (self.num_cation / (self.nvt_v * 1e-30)) / (6.022 * 1e23)
        self.d_to_sigma = self.c * faraday_constant_2 / (gas_constant * temp)

    @classmethod
    def from_output_full(
        cls,
        data_dir,
        wrapped_dir,
        unwrapped_dir,
        nvt_start,
        time_step,
        name,
        select_dict=None,
        res_dict=None,
        cation_name="cation",
        anion_name="anion",
        cation_charge=1,
        anion_charge=-1,
        temperature=298.5,
        cond=True,
    ):
        """
        Constructor from lammps data file and wrapped and unwrapped trajectory dcd file.

        Args:
            data_dir (str): Path to the data file.
            wrapped_dir (str): Path to the wrapped dcd file.
            unwrapped_dir (str): Path to the unwrapped dcd file.
            nvt_start (int): NVT start time step.
            time_step (int or float): LAMMPS timestep.
            name (str): Name of the MD run.
            select_dict: A dictionary of species selection.
            res_dict (dict): A dictionary of resnames.
            cation_name: Name of cation. Default to "cation".
            anion_name: Name of anion. Default to "anion".
            cation_charge: Charge of cation. Default to 1.
            anion_charge: Charge of anion. Default to 1.
            temperature: Temperature of the MD run. Default to 298.15.
            cond (bool): Whether to calculate conductivity MSD. Default to True.
        """
        lammps_data = LammpsData.from_file(data_dir)
        if res_dict is None:
            res_dict = res_dict_from_datafile(data_dir)
        wrapped_run = MDAnalysis.Universe(data_dir, wrapped_dir, format="LAMMPS")
        unwrapped_run = MDAnalysis.Universe(data_dir, unwrapped_dir, format="LAMMPS")

        return cls(
            lammps_data,
            wrapped_run,
            unwrapped_run,
            nvt_start,
            time_step,
            name,
            select_dict=select_dict,
            res_dict=res_dict,
            cation_name=cation_name,
            anion_name=anion_name,
            cation_charge=cation_charge,
            anion_charge=anion_charge,
            temperature=temperature,
            cond=cond,
        )

    def get_init_dimension(self):
        """
        Returns the initial box dimension.
        """
        return self.wrapped_run.trajectory[0].dimensions

    def get_equilibrium_dimension(self, npt_range, period=200):
        """
        Returns the equilibrium box dimension.
        """
        ave_dx = [np.inf, np.inf - 1]
        count = 0
        ave_dxi = 0
        convergence = -1
        for i in range(npt_range):
            ave_dxi += self.wrapped_run.trajectory[i].dimensions[0]
            count += 1
            if count * self.time_step == period:
                print(ave_dxi / count)
                ave_dx.append(ave_dxi / count)
                count = 0
            if ave_dx[-1] >= ave_dx[-2]:
                convergence = i
                break
        d = list()
        for j in range(convergence, npt_range):
            d.append(self.wrapped_run.trajectory[j].dimensions)
        return np.mean(np.array(d), axis=0)

    def get_nvt_dimension(self):
        """
        Returns the equilibrium box dimension.
        """
        return self.wrapped_run.trajectory[-1].dimensions

    def get_cond_array(self):
        """Calculates the conductivity "mean square displacement".

        Returns an array of MSD values for each time in the trajectory.
        """
        nvt_run = self.unwrapped_run
        cations = nvt_run.select_atoms(self.select_dict.get("cation"))
        anions = nvt_run.select_atoms(self.select_dict.get("anion"))
        cond_array = calc_cond_msd(
            nvt_run,
            anions,
            cations,
            self.nvt_start,
            self.cation_charge,
            self.anion_charge,
        )
        return cond_array

    def plot_cond_array(self, start, end, *runs, reference=True):
        """Plots the conductivity MSD as a function of time

        Args:
            start (int): Start time step.
            end (int): End time step.
            runs (MdRun): Other runs to be compared in the same plot.
            reference (bool): Whether to plot reference line.
                Default to True.
        """
        if self.cond_array is None:
            self.cond_array = self.get_cond_array()
        colors = ["g", "r", "c", "m", "y", "k"]
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.loglog(
            self.time_array[start:end],
            self.cond_array[start:end],
            color="b",
            lw=2,
            label=self.name,
        )
        for i, run in enumerate(runs):
            ax.loglog(
                run.time_array[start:end],
                run.cond_array[start:end],
                color=colors[i],
                lw=2,
                label=run.name,
            )
        if reference:
            ax.loglog((100000, 1000000), (1000, 10000))
        ax.set_ylabel("MSD (A^2)")
        ax.set_xlabel("Time (ps)")
        ax.set_ylim([10, 1000000])
        ax.set_xlim([100, 500000000])
        ax.legend()
        fig.show()

    def get_conductivity(self, start, end):
        """Calculates the Green-Kubo (GK) conductivity

        Args:
            start (int): Start time step.
            end (int): End time step.

        Print conductivity in mS/cm.
        """
        conductivity_calculator(self.time_array, self.cond_array, self.nvt_v, self.name, start, end, T=self.temp)
        return None

    def coord_num_array_one_species(self, species, distance, run_start, run_end):
        """Calculates the coordination number array of one species around
        the cation.

        Args:
            species (str): The interested species.
            distance (int or float): The coordination cutoff distance.
            run_start (int): Start time step.
            run_end (int): End time step.

        Returns an array of the species coordination number for each time
        in the trajectory.
        """
        nvt_run = self.wrapped_run
        species_dict = {species: distance}
        li_atoms = nvt_run.select_atoms(self.select_dict.get("cation"))
        num_array = coord_shell_array(
            nvt_run,
            num_of_neighbor_one_li,
            li_atoms,
            species_dict,
            self.select_dict,
            run_start,
            run_end,
        )["total"]
        return num_array

    def coord_num_array_multi_species(self, species_dict, run_start, run_end):
        """Calculates the coordination number array of multiple species around
        the cation

        Args:
            species_dict (dict): A dict of coordination cutoff distance
                of the interested species.
            run_start (int): Start time step.
            run_end (int): End time step.

        Returns a python dict of arrays of coordination numbers of each species
         for each time in the trajectory.
        """
        nvt_run = self.wrapped_run
        li_atoms = nvt_run.select_atoms(self.select_dict.get("cation"))
        num_array = coord_shell_array(
            nvt_run,
            num_of_neighbor_one_li,
            li_atoms,
            species_dict,
            self.select_dict,
            run_start,
            run_end,
        )

        return num_array

    def get_solvation_structure(self, species_dict, run_start, run_end, structure_code, write_freq, write_path):
        """Writes out the desired solvation structure

        Args:
            species_dict (dict): A dict of coordination cutoff distance
                of interested species.
            run_start (int): Start time step.
            run_end (int): End time step.
            structure_code: An integer code representing the solvation
                structure, for example, 221 is two species A, two species B
                and one species C.
            write_freq: Probability to write out files.
            write_path: Path to write out files.

        Write out a series of solvation structures as .xyz files .
        """
        nvt_run = self.wrapped_run
        li_atoms = nvt_run.select_atoms(self.select_dict.get("cation"))
        for li in tqdm(li_atoms):
            num_of_neighbor_one_li(
                nvt_run,
                li,
                species_dict,
                self.select_dict,
                run_start,
                run_end,
                write=True,
                structure_code=structure_code,
                write_freq=write_freq,
                write_path=write_path,
                element_id_dict=self.element_id_dict,
            )

    def coord_num_array_simple(self, species, distance, run_start, run_end):
        """Calculates the solvation structure type (1 for SSIP, 2 for CIP,
        3 for AGG) array of the cation

        Args:
            species (str): The interested species.
            distance (int or float): The coordination cutoff distance.
            run_start (int): Start time step.
            run_end (int): End time step.

        Returns an array of the solvation structure type
        for each time in the trajectory.
        """
        nvt_run = self.wrapped_run
        species_dict = {species: distance}
        li_atoms = nvt_run.select_atoms(self.select_dict.get("cation"))
        num_array = coord_shell_array(
            nvt_run,
            num_of_neighbor_one_li_simple,
            li_atoms,
            species_dict,
            self.select_dict,
            run_start,
            run_end,
        )["total"]
        return num_array

    def coordination_one_species(self, species, distance, run_start, run_end):
        """Tabulates the coordination number distribution of one species
        around the cation

        Args:
            species (str): The interested species.
            distance (int or float): The coordination cutoff distance.
            run_start (int): Start time step.
            run_end (int): End time step.

        Returns pandas.dataframe of the species coordination number
        and corresponding percentage.
        """
        num_array = self.coord_num_array_one_species(species, distance, run_start, run_end)
        shell_component, shell_count = np.unique(num_array.flatten(), return_counts=True)
        combined = np.vstack((shell_component, shell_count)).T

        item_name = "Num of " + species + " within " + str(distance) + " " + "\u212B"
        item_list = []
        percent_list = []
        for i in range(len(combined)):
            item_list.append(str(int(combined[i, 0])))
            percent_list.append(str("%.4f" % (combined[i, 1] / combined[:, 1].sum() * 100)) + "%")
        df_dict = {item_name: item_list, "Percentage": percent_list}
        df = pd.DataFrame(df_dict)
        return df

    def rdf_integral(self, species_dict, run_start, run_end):
        """Calculate the integral of the radial distribution function of
        selected species

        Args:
            species_dict (dict): A dict of coordination cutoff distance
                of interested species.
            run_start (int): Start time step.
            run_end (int): End time step.

        Returns pandas.dataframe of the species and the coordination number.
        """
        cn_values = self.coord_num_array_multi_species(species_dict, run_start, run_end)
        item_name = "species in first solvation shell"
        item_list = []
        cn_list = []
        for kw in cn_values.keys():
            if kw != "total":
                shell_component, shell_count = np.unique(cn_values[kw].flatten(), return_counts=True)
                cn = (shell_component * shell_count / shell_count.sum()).sum()
                item_list.append(kw)
                cn_list.append(cn)
        df_dict = {item_name: item_list, self.name: cn_list}
        df = pd.DataFrame(df_dict)
        return df

    def shell_simple(self, species, distance, run_start, run_end):
        """Tabulates the percentage of each solvation structures (CIP/SSIP/AGG)

        Args:
            species (str): The interested species.
            distance (int or float): The coordination cutoff distance.
            run_start (int): Start time step.
            run_end (int): End time step.

        Returns pandas.dataframe of the solvation structure and percentage.
        """
        num_array = self.coord_num_array_simple(species, distance, run_start, run_end)

        shell_component, shell_count = np.unique(num_array.flatten(), return_counts=True)
        combined = np.vstack((shell_component, shell_count)).T

        item_name = "solvation structure"
        item_dict = {"1": "ssip", "2": "cip", "3": "agg"}
        item_list = []
        percent_list = []
        for i in range(len(combined)):
            item = str(int(combined[i, 0]))
            item_list.append(item_dict.get(item))
            percent_list.append(str("%.4f" % (combined[i, 1] / combined[:, 1].sum() * 100)) + "%")
        df_dict = {item_name: item_list, "Percentage": percent_list}
        df = pd.DataFrame(df_dict)
        return df

    def get_msd_all(self, start=None, stop=None, fft=True, species="cation"):
        """Calculates the mean square displacement (MSD) of the cation

        Args:
            start (int): Start time step.
            stop (int): End time step.
            fft (bool): Whether to use fft to calculate msd. Default to True.
            species (str): The select_dict key of the atom group to calculate. Default to "cation".

        Returns an array of MSD values for each time in the trajectory
        """
        msd_array = total_msd(
            self.unwrapped_run,
            start=start,
            stop=stop,
            select=self.select_dict.get(species),
            fft=fft,
        )
        return msd_array

    def get_msd_by_length(self, distance, run_start, run_end):
        """

        Args:
            distance (int or float): The coordination cutoff distance.
            run_start (int): Start time step.
            run_end (int): End time step.

        Returns:

        """
        nvt_run = self.unwrapped_run
        li_atoms = nvt_run.select_atoms(self.select_dict.get("cation"))
        free_array, attach_array = special_msd(nvt_run, li_atoms, self.select_dict, distance, run_start, run_end)
        return free_array, attach_array

    def get_msd_partial(self, distance, run_start, run_end, largest=1000):
        """

        Args:
            distance (int or float): The coordination cutoff distance.
            run_start (int): Start time step.
            run_end (int): End time step.
            largest (int): The largest time sequence to trace.

        Returns:

        """
        nvt_run = self.unwrapped_run
        li_atoms = nvt_run.select_atoms(self.select_dict.get("cation"))
        free_array, attach_array = partial_msd(
            nvt_run, li_atoms, largest, self.select_dict, distance, run_start, run_end
        )
        return free_array, attach_array

    def get_d(self, msd_array, start, stop, percentage=1, species="cation"):
        """Calculates the self-diffusion coefficient of the cation and
        the Nernst-Einstein conductivity

        Args:
            msd_array (numpy.array): msd array.
            start (int): Start time step.
            stop (int): End time step.
            percentage (int or float): The percentage of the cation.
                Default to 1.
            species (str): The select_dict key of the atom group to calculate. Default to "cation".

        Print self-diffusion coefficient in m^2/s and NE conductivity in mS/cm.
        """
        a2 = 1e-20
        ps = 1e-12
        s_m_to_ms_cm = 10
        if percentage != 1:
            d = (msd_array[start] - msd_array[stop]) / (start - stop) / self.time_step / 6 * a2 / ps
            sigma = percentage * d * self.d_to_sigma * s_m_to_ms_cm
            print(
                "Diffusivity of",
                "%.2f" % (percentage * 100) + "% " + species + ": ",
                d,
                "m^2/s",
            )
            if species.lower() == "cation" or species.lower() == "li":
                print(
                    "NE Conductivity of",
                    "%.2f" % (percentage * 100) + "% " + species + ": ",
                    sigma,
                    "mS/cm",
                )
        else:
            d = (msd_array[start] - msd_array[stop]) / (start - stop) / self.time_step / 6 * a2 / ps
            sigma = d * self.d_to_sigma * s_m_to_ms_cm
            print("Diffusivity of all " + species + ":", d, "m^2/s")
            if species.lower() == "cation" or species.lower() == "li":
                print("NE Conductivity of all " + species + ":", sigma, "mS/cm")

    def get_neighbor_corr(self, species_dict, run_start, run_end):
        """Calculates the neighbor auto-correlation function (ACF)
        of selected species around cation

        Args:
            species_dict (dict): Dict of Cutoff distance of neighbor
                for each species.
            run_start: Start time step.
            run_end (int): End time step.

        Returns an array of the time series and a dict of ACFs of each species.
        """
        return calc_neigh_corr(
            self.wrapped_run,
            species_dict,
            self.select_dict,
            self.time_step,
            run_start,
            run_end,
        )

    def get_residence_time(self, species_list, times, acf_avg_dict, cutoff_time):
        """Calculates the residence time of selected species around cation

        Args:
            species_list (list): List of species name.
            times (np.ndarray): The time series.
            acf_avg_dict: A dict of ACFs of each species.
            cutoff_time (int): Cutoff time for fitting the exponential decay.

        Returns the residence time of each species.
        """
        return fit_residence_time(times, species_list, acf_avg_dict, cutoff_time, self.time_step)

    def get_neighbor_trj(self, run_start, run_end, li_atom, species, distance):
        """Calculates the distance of cation-neighbor as a function of time

        Args:
            run_start (int): start time step.
            run_end (int): end time step.
            li_atom (MDAnalysis.core.groups.Atom): the interested cation
                atom object.
            species (str): The interested neighbor species.
            distance (int or float): The coordination cutoff distance.

        Returns a dict of distance arrays of cation-neighbor
        as a function of time with neighbor id as keys.
        """
        return trajectory(
            self.wrapped_run,
            li_atom,
            run_start,
            run_end,
            species,
            self.select_dict,
            distance,
        )

    def get_hopping_freq_dist(self, run_start, run_end, species, distance, hopping_cutoff, smooth=51, mode="full"):
        """Calculates the cation hopping rate and hopping distance

        Args:
            run_start (int): Start time step.
            run_end (int): End time step.
            species (str): cation Binding site species.
            distance (int or float): Binding cutoff distance.
            hopping_cutoff: (int or float): Detaching cutoff distance.
            smooth (int): The length of the smooth filter window. Default to 51.
            mode (str): The mode of treating hopping event. Default to "full".

        Returns the cation average hopping rate and average hopping distance.
        """
        nvt_run = self.wrapped_run
        li_atoms = nvt_run.select_atoms(self.select_dict.get("cation"))
        freqs = []
        hopping_distance = []
        for li in tqdm(li_atoms[:]):
            neighbor_trj = trajectory(nvt_run, li, run_start, run_end, species, self.select_dict, distance)
            if mode == "full":
                sites, freq, steps = find_nearest(neighbor_trj, self.time_step, distance, hopping_cutoff, smooth=smooth)
            elif mode == "free":
                sites, freq, steps = find_nearest_free_only(
                    neighbor_trj, self.time_step, distance, hopping_cutoff, smooth=smooth
                )
            else:
                raise ValueError("invalid mode")
            coords = []
            for step in steps:
                coord_li = nvt_run.trajectory[step + run_start][li.id - 1]
                coords.append(coord_li)
            if len(coords) > 1:
                dists = []
                for i in range(len(coords) - 1):
                    dist = distance_array(coords[i + 1], coords[i], box=self.get_nvt_dimension())[0][0]
                    dists.append(dist)
                li_mean_dists = np.mean(dists)
                hopping_distance.append(li_mean_dists)
            freqs.append(freq)
        return np.mean(freqs), np.mean(hopping_distance)

    def shell_evolution(
        self,
        species_dict,
        run_start,
        run_end,
        lag_step,
        distance,
        hopping_cutoff,
        smooth=51,
        cool=0,
        center="center",
        duplicate_run=None,
    ):
        """Calculates the coordination number of species in the species_dict
        as a function of time before and after hopping events.

        Args:
            species_dict (dict): A dict of coordination cutoff distance
                of interested species.
            run_start (int): Start time step.
            run_end (int): End time step.
            lag_step (int): time steps to track before and after the hopping event
            distance (int or float): Binding cutoff distance.
            hopping_cutoff: (int or float): Detaching cutoff distance.
            smooth (int): The length of the smooth filter window. Default to 51.
            cool (int): The cool down timesteps between hopping in and hopping out.
            center (str): The select_dict key of the binding site. Default to "center".
            duplicate_run (list): Default to None.
        """
        in_list = dict()
        out_list = dict()
        for k in list(species_dict):
            in_list[k] = []
            out_list[k] = []
        process_evol(
            self,
            in_list,
            out_list,
            species_dict,
            run_start,
            run_end,
            lag_step,
            distance,
            hopping_cutoff,
            smooth,
            cool,
            center,
        )
        if duplicate_run is not None:
            for run in duplicate_run:
                process_evol(
                    run,
                    in_list,
                    out_list,
                    species_dict,
                    run_start,
                    run_end,
                    lag_step,
                    distance,
                    hopping_cutoff,
                    smooth,
                    cool,
                    center,
                )
        cn_dict = dict()
        cn_dict["time"] = np.array([i * self.time_step - lag_step * self.time_step for i in range(lag_step * 2 + 1)])
        for k in list(species_dict):
            if "in_count" not in cn_dict:
                cn_dict["in_count"] = np.array(in_list[k]).shape[0]
                cn_dict["out_count"] = np.array(out_list[k]).shape[0]
            k_dict = dict()
            k_dict["in_ave"] = np.nanmean(np.array(in_list[k]), axis=0)
            k_dict["in_err"] = np.nanstd(np.array(in_list[k]), axis=0)
            k_dict["out_ave"] = np.nanmean(np.array(out_list[k]), axis=0)
            k_dict["out_err"] = np.nanstd(np.array(out_list[k]), axis=0)
            cn_dict[k] = k_dict
        return cn_dict

    def get_heat_map(
        self,
        run_start,
        run_end,
        species,
        distance,
        hopping_cutoff,
        cartesian_by_ref=None,
        bind_atom_type=None,
        sym_dict=None,
        sample=None,
        smooth=51,
    ):
        """Calculates the heatmap matrix of cation around a cluster/molecule

        Args:
            run_start (int): Start time step.
            run_end (int): End time step.
            species (str): cation Binding site species.
            distance (int or float): Binding cutoff distance.
            hopping_cutoff: (int or float): Detaching cutoff distance.
            cartesian_by_ref (np.array): Transformation matrix between cartesian
                and reference coordinate systems. Default to None.
            bind_atom_type (str): Selection for binding site atom.
                Default to None.
            sym_dict (dict): Dictionary of symmetry operation dictionary.
                Default to None.
            sample (int): Number of samples desired. Default to None,
                which is no sampling.
            smooth (int): The length of the smooth filter window. Default to 51.
        """
        nvt_run = self.wrapped_run
        li_atoms = nvt_run.select_atoms(self.select_dict.get("cation"))
        coord_list = np.array([[0, 0, 0]])
        for li in tqdm(li_atoms[:]):
            neighbor_trj = trajectory(nvt_run, li, run_start, run_end, species, self.select_dict, distance)
            sites, freq, steps = find_nearest(neighbor_trj, self.time_step, distance, hopping_cutoff, smooth=smooth)
            if bind_atom_type is None:
                bind_atom_type = self.select_dict.get("anion")
            if cartesian_by_ref is None:
                cartesian_by_ref = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            coords = heat_map(
                nvt_run,
                li,
                sites,
                bind_atom_type,
                cartesian_by_ref,
                run_start,
                run_end,
            )
            coord_list = np.concatenate((coord_list, coords), axis=0)
        coord_list = coord_list[1:]
        if sym_dict:
            return get_full_coords(coord_list, **sym_dict, sample=sample)
        else:
            return get_full_coords(coord_list, sample=sample)

    def get_cluster_distance(self, run_start, run_end, neighbor_cutoff, cluster_center="center"):
        """Calculates the average distance of the center of clusters/molecules

        Args:
            run_start (int): Start time step.
            run_end (int): End time step.
            neighbor_cutoff (int of float): Upper limit of
                first nearest neighbor.
            cluster_center (str): species name of cluster center.
                Default to "center".
        """
        center_atoms = self.wrapped_run.select_atoms(self.select_dict.get(cluster_center))
        trj = self.wrapped_run.trajectory[run_start:run_end:]
        means = []
        for ts in trj:
            distance_matrix = capped_distance(
                center_atoms.positions,
                center_atoms.positions,
                max_cutoff=neighbor_cutoff,
                box=ts.dimensions,
                return_distances=True,
            )[1]
            distance_matrix[distance_matrix == 0] = np.nan
            means.append(np.nanmean(distance_matrix))
        return np.mean(means)


class MdJob:
    """
    A core class for MD results analysis.
    """

    def __init__(self, name):
        """
        Base constructor
        """
        self.name = name

    @classmethod
    def from_dict(cls):
        """
        Constructor.

        Returns:

        """
        return cls("name")

    @classmethod
    def from_recipe(cls):
        """
        Constructor.

        Returns:

        """
        return cls("name")
