# coding: utf-8
# Copyright (c) Tingzheng Hou.
# Distributed under the terms of the MIT License.

"""
This module implements two core class MdRun and MdJob
for molecular dynamics simulation analysis and job setup.
"""
from __future__ import annotations
import MDAnalysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union, Dict, Tuple, List, Optional
from pymatgen.io.lammps.data import LammpsData, CombinedData
from MDAnalysis import Universe
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
from mdgo.conductivity import calc_cond, conductivity_calculator
from mdgo.coordination import (
    coord_shell_array,
    num_of_neighbor,
    num_of_neighbor_simple,
    neighbor_distance,
    find_nearest,
    find_nearest_free_only,
    process_evol,
    heat_map,
    get_full_coords,
)
from mdgo.msd import total_msd, partial_msd
from mdgo.residence_time import calc_neigh_corr, fit_residence_time

__author__ = "Tingzheng Hou"
__version__ = "1.0"
__maintainer__ = "Tingzheng Hou"
__email__ = "tingzheng_hou@berkeley.edu"
__date__ = "Feb 9, 2021"


class MdRun:
    """
    A core class for MD results analysis.

    Args:
        lammps_data: The LammpsData object that has the force field and topology information.
        wrapped_run: The Universe object of wrapped trajectory.
        unwrapped_run: The Universe object of unwrapped trajectory.
        nvt_start: NVT start time step.
        time_step: LAMMPS timestep.
        name: Name of the MD run.
        select_dict: A dictionary of atom species selection, where each atom species name is a key
                and the corresponding values are the selection language. This dict is intended for
                analyzing interested atoms.
        res_dict: A dictionary of resnames, where each resname is a key
                and the corresponding values are the selection language. This dict is intended for
                analyzing interested residues (ions/molecules).
        cation_name: Name of cation. Default to "cation".
        anion_name: Name of anion. Default to "anion".
        cation_charge: Charge of cation. Default to 1.
        anion_charge: Charge of anion. Default to 1.
        temperature: Temperature of the MD run. Default to 298.15.
        cond: Whether to calculate conductivity MSD. Default to True.
    """

    def __init__(
        self,
        lammps_data: Union[LammpsData, CombinedData],
        wrapped_run: Universe,
        unwrapped_run: Universe,
        nvt_start: int,
        time_step: float,
        name: str,
        select_dict: Optional[Dict[str, str]] = None,
        res_dict: Optional[Dict[str, str]] = None,
        cation_name: str = "cation",
        anion_name: str = "anion",
        cation_charge: float = 1,
        anion_charge: float = -1,
        temperature: float = 298.15,
        cond: bool = True,
    ):
        """
        Base constructor. This is a low level constructor designed to work with
         parsed data ({Universe}) or other bridging objects ({CombinedData}). Not
        recommended to use directly.
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
        if select_dict is None and res_dict is None:
            self.res_dict = res_dict_from_lammpsdata(self.data)
        elif res_dict is None:
            assert isinstance(select_dict, dict)
            self.res_dict = res_dict_from_select_dict(self.wrapped_run, select_dict)
        else:
            self.res_dict = res_dict
        assign_resname(self.wrapped_run, self.res_dict)
        assign_resname(self.unwrapped_run, self.res_dict)
        if select_dict is None:
            self.select_dict = select_dict_from_resname(self.wrapped_run)
        else:
            self.select_dict = select_dict
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
        data_dir: str,
        wrapped_dir: str,
        unwrapped_dir: str,
        nvt_start: int,
        time_step: float,
        name: str,
        select_dict: Optional[Dict[str, str]] = None,
        res_dict: Optional[Dict[str, str]] = None,
        cation_name: str = "cation",
        anion_name: str = "anion",
        cation_charge: float = 1,
        anion_charge: float = -1,
        temperature: float = 298.15,
        cond: bool = True,
    ):
        """
        Constructor from lammps data file and wrapped and unwrapped trajectory dcd file.

        Args:
            data_dir: Path to the data file.
            wrapped_dir: Path to the wrapped dcd file.
            unwrapped_dir: Path to the unwrapped dcd file.
            nvt_start: NVT start time step.
            time_step: LAMMPS timestep.
            name: Name of the MD run.
            select_dict: A dictionary of species selection.
            res_dict: A dictionary of resnames.
            cation_name: Name of cation. Default to "cation".
            anion_name: Name of anion. Default to "anion".
            cation_charge: Charge of cation. Default to 1.
            anion_charge: Charge of anion. Default to 1.
            temperature: Temperature of the MD run. Default to 298.15.
            cond: Whether to calculate conductivity MSD. Default to True.
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

    def get_init_dimension(self) -> np.ndarray:
        """
        Returns the initial box dimension.
        """
        return self.wrapped_run.trajectory[0].dimensions

    def get_equilibrium_dimension(self, npt_range: int, period: int = 200) -> np.ndarray:
        """
        Returns the equilibrium box dimension.

        Args:
            npt_range: The maximum time step of the npt run.
            period: The interval of checking points for volume convergence.
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

    def get_nvt_dimension(self) -> np.ndarray:
        """
        Returns the box dimension at the last frame.
        """
        return self.wrapped_run.trajectory[-1].dimensions

    def get_cond_array(self) -> np.ndarray:
        """Calculates the conductivity "mean square displacement".

        Return:
             An array of MSD values for each time in the trajectory.
        """
        nvt_run = self.unwrapped_run
        cations = nvt_run.select_atoms(self.select_dict.get("cation"))
        anions = nvt_run.select_atoms(self.select_dict.get("anion"))
        cond_array = calc_cond(
            nvt_run,
            anions,
            cations,
            self.nvt_start,
            self.cation_charge,
            self.anion_charge,
        )
        return cond_array

    def plot_cond_array(self, start: int, end: int, *runs: MdRun, reference: bool = True):
        """Plots the conductivity MSD as a function of time.

        Args:
            start: Start time step.
            end: End time step.
            runs: Other runs to be compared in the same plot.
            reference: Whether to plot reference line. Default to True.
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

    def get_conductivity(self, start: int, end: int) -> float:
        """Calculates the Green-Kubo (GK) conductivity in mS/cm.

        Args:
            start: Start time step.
            end: End time step.
        """
        cond = conductivity_calculator(self.time_array, self.cond_array, self.nvt_v, self.name, start, end)
        return cond

    def coord_num_array_one_species(
        self,
        species: str,
        distance: float,
        run_start: int,
        run_end: int,
        center_atom: str = "cation",
    ) -> np.ndarray:
        """Calculates the coordination number array of one {species} around the interested {center_atom}.

        Args:
            species: The interested species.
            distance: The coordination cutoff distance.
            run_start: Start frame of analysis.
            run_end: End frame of analysis.
            center_atom: The interested atom. Default to "cation".

        Return:
             An array of coordination number for each time in the trajectory.
        """
        nvt_run = self.wrapped_run
        distance_dict = {species: distance}
        center_atoms = nvt_run.select_atoms(self.select_dict.get(center_atom))
        num_array = coord_shell_array(
            nvt_run,
            num_of_neighbor,
            center_atoms,
            distance_dict,
            self.select_dict,
            run_start,
            run_end,
        )["total"]
        return num_array

    def coord_num_array_multi_species(
        self,
        distance_dict: Dict[str, float],
        run_start: int,
        run_end: int,
        center_atom: str = "cation",
    ) -> Dict[str, np.ndarray]:
        """Calculates the coordination number array of multiple species around the interested {center_atom}.

        Args:
            distance_dict: A dict of coordination cutoff distance
                of the interested species.
            run_start: Start frame of analysis.
            run_end: End frame of analysis.
            center_atom: The center atom. Default to "cation".

        Return:
             The coordination numbers of each species as a python dict of arrays
             for each timestep in the trajectory.
        """
        nvt_run = self.wrapped_run
        center_atoms = nvt_run.select_atoms(self.select_dict.get(center_atom))
        num_array = coord_shell_array(
            nvt_run,
            num_of_neighbor,
            center_atoms,
            distance_dict,
            self.select_dict,
            run_start,
            run_end,
        )
        return num_array

    def get_solvation_structure(
        self,
        distance_dict: Dict[str, float],
        run_start: int,
        run_end: int,
        structure_code: int,
        write_freq: float,
        write_path: str,
        center_atom: str = "cation",
    ):
        """Writes out a series of desired solvation structures as {.xyz} files

        Args:
            distance_dict: A dict of coordination cutoff distance of interested species.
            run_start: Start frame of analysis.
            run_end: End frame of analysis.
            structure_code: An integer code representing the solvation
                structure, for example, 221 is two species A, two species B
                and one species C.
            write_freq: Probability to write out files.
            write_path: Path to write out files.
            center_atom: The interested atom. Default to "cation".
        """
        nvt_run = self.wrapped_run
        center_atoms = nvt_run.select_atoms(self.select_dict.get(center_atom))
        for atom in tqdm(center_atoms):
            num_of_neighbor(
                nvt_run,
                atom,
                distance_dict,
                self.select_dict,
                run_start,
                run_end,
                write=True,
                structure_code=structure_code,
                write_freq=write_freq,
                write_path=write_path,
                element_id_dict=self.element_id_dict,
            )

    def coord_num_array_simple(
        self,
        species: str,
        distance: float,
        run_start: int,
        run_end: int,
        center_atom: str = "cation",
    ) -> np.ndarray:
        """Calculates the solvation structure type (1 for SSIP, 2 for CIP,
        3 for AGG) array of the solvation structure {center_atom} (typically the cation).

        Args:
            species: The interested species.
            distance: The coordination cutoff distance.
            run_start: Start frame of analysis.
            run_end: End frame of analysis.
            center_atom: The solvation structure center atom. Default to "cation".

        Return:
            An array of the solvation structure type for each timestep in the trajectory.
        """
        nvt_run = self.wrapped_run
        distance_dict = {species: distance}
        center_atoms = nvt_run.select_atoms(self.select_dict.get(center_atom))
        num_array = coord_shell_array(
            nvt_run,
            num_of_neighbor_simple,
            center_atoms,
            distance_dict,
            self.select_dict,
            run_start,
            run_end,
        )["total"]
        return num_array

    def coordination_one_species(
        self,
        species: str,
        distance: float,
        run_start: int,
        run_end: int,
        center_atom: str = "cation",
    ) -> pd.DataFrame:
        """Tabulates the coordination number distribution of one species
        around the solvation structure {center_atom}.

        Args:
            species: The interested species.
            distance: The coordination cutoff distance.
            run_start: Start frame of analysis.
            run_end: End frame of analysis.
            center_atom: The solvation structure center atom. Default to "cation".

        Return:
             A dataframe of the species coordination number and corresponding percentage.
        """
        num_array = self.coord_num_array_one_species(species, distance, run_start, run_end, center_atom=center_atom)
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

    def rdf_integral(
        self,
        distance_dict: Dict[str, float],
        run_start: int,
        run_end: int,
        center_atom: str = "cation",
    ) -> pd.DataFrame:
        """Calculate the integral of the radial distribution function of selected species around the {center_atom}

        Args:
            distance_dict: A dict of coordination cutoff distance
                of interested species.
            run_start: Start frame of analysis.
            run_end: End frame of analysis.
            center_atom: The center atom to calculate the radial distribution for. Default to "cation".

        Return:
             A dataframe of the species and the coordination number.
        """
        cn_values = self.coord_num_array_multi_species(distance_dict, run_start, run_end, center_atom=center_atom)
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

    def shell_simple(self, species: str, distance: float, run_start: int, run_end: int) -> pd.DataFrame:
        """Tabulates the percentage of each solvation structures (CIP/SSIP/AGG)

        Args:
            species: The interested species.
            distance: The coordination cutoff distance.
            run_start: Start frame of analysis.
            run_end: End frame of analysis.

        Return:
             A dataframe of the solvation structure and percentage.
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

    def get_msd_all(
        self,
        start: int = 0,
        stop: int = -1,
        fft: bool = True,
        species: str = "cation",
    ) -> np.ndarray:
        """Calculates the mean square displacement (MSD) of the interested atom species.

        Args:
            start: Start time step.
            stop: End time step.
            fft: Whether to use fft to calculate msd. Default to True.
            species: The select_dict key of the atom group to calculate. Default to "cation".

        Return:
             An array of MSD values in the trajectory
        """
        selection = self.select_dict.get(species)
        assert selection is not None
        msd_array = total_msd(
            self.unwrapped_run,
            start=start,
            stop=stop,
            select=selection,
            fft=fft,
        )
        return msd_array

    def get_msd_partial(
        self,
        distance: float,
        run_start: int,
        run_end: int,
        largest: int = 1000,
        center_atom: str = "cation",
        binding_site: str = "anion",
    ) -> Tuple[Optional[List[np.ndarray]], Optional[List[np.ndarray]]]:
        """
        Calculates the mean square displacement (MSD) of the {center_atom} according to coordination states.
        The returned {free_array} include the MSD when {center_atom} is not coordinated to {binding_site}.
        The {attach_array} includes the MSD of {center_atom} is not coordinated to {binding_site}.

        Args:
            distance: The coordination cutoff distance.
            run_start: Start frame of analysis.
            run_end: End frame of analysis.
            largest: The largest time sequence to trace.
            center_atom: The interested atom. Default to "cation".
            binding_site: The species the {center_atom} coordinates to. Default to "anion".

        Returns:
            Two arrays of MSD in the trajectory
        """
        nvt_run = self.unwrapped_run
        center_atoms = nvt_run.select_atoms(self.select_dict.get(center_atom))
        free_array, attach_array = partial_msd(
            nvt_run, center_atoms, largest, self.select_dict, distance, run_start, run_end, binding_site=binding_site
        )
        return free_array, attach_array

    def get_d(self, msd_array: np.ndarray, start: int, stop: int, percentage: float = 1, species: str = "cation"):
        """Prints the self-diffusion coefficient (in m^2/s) of the species.
        Prints the Nernst-Einstein conductivity (in mS/cm) if it's the cation.

        Args:
            msd_array: msd array.
            start: Start time step.
            stop: End time step.
            percentage: The percentage of the cation. Default to 1.
            species: The select_dict key of the atom group to calculate. Default to "cation".
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

    def get_neighbor_corr(
        self,
        distance_dict: Dict[str, float],
        run_start: int,
        run_end: int,
        center_atom: str = "cation",
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Calculates the neighbor auto-correlation function (ACF)
        of selected species around center_atom.

        Args:
            distance_dict: Dict of Cutoff distance of neighbor
                for each species.
            run_start: Start frame of analysis.
            run_end: End frame of analysis.
            center_atom: The center atom to calculate the ACF for. Default to "cation".

        Return:
             An array of the time series and a dict of ACFs of each species.
        """
        return calc_neigh_corr(
            self.wrapped_run,
            distance_dict,
            self.select_dict,
            self.time_step,
            run_start,
            run_end,
            center_atom=center_atom,
        )

    def get_residence_time(
        self, species_list: List[str], times: np.ndarray, acf_avg_dict: Dict[str, np.ndarray], cutoff_time: int
    ) -> Dict[str, np.floating]:
        """Calculates the residence time of selected species around cation

        Args:
            species_list: List of species name.
            times: The time series.
            acf_avg_dict: A dict of ACFs of each species.
            cutoff_time: Cutoff time for fitting the exponential decay.

        Return:
             The residence time of each species in a dict.
        """
        return fit_residence_time(times, species_list, acf_avg_dict, cutoff_time, self.time_step)

    def get_neighbor_trj(
        self,
        run_start: int,
        run_end: int,
        species: str,
        neighbor_cutoff: float,
        center_atom: str = "cation",
        index: int = 0,
    ) -> Dict[str, np.ndarray]:
        """Returns the distance between one center atom and neighbors as a function of time

        Args:
            run_start: Start frame of analysis.
            run_end: End frame of analysis.
            center_atom: The interested atom. Default to "cation".
            species: The interested neighbor species.
            neighbor_cutoff: The neighbor cutoff distance.
            index: The index of the atom in the interested atom group.

        Return:
             A dict of distance arrays of the center atom-neighbor as a function of time with neighbor id as keys.
        """
        center_atoms = self.wrapped_run.select_atoms(self.select_dict.get(center_atom))
        return neighbor_distance(
            self.wrapped_run,
            center_atoms[index],
            run_start,
            run_end,
            species,
            self.select_dict,
            neighbor_cutoff,
        )

    def get_hopping_freq_dist(
        self,
        run_start: int,
        run_end: int,
        binding_site: str,
        distance: float,
        hopping_cutoff: float,
        floating_atom: str = "cation",
        smooth: int = 51,
        mode: str = "full",
    ) -> Tuple[np.floating, np.floating]:
        """Calculates the cation hopping rate and hopping distance.

        Args:
            run_start: Start frame of analysis.
            run_end: End frame of analysis.
            binding_site: Floating ion binding site species.
            distance: Binding cutoff distance.
            hopping_cutoff: Detaching cutoff distance.
            floating_atom: Floating ion species.
            smooth: The length of the smooth filter window. Default to 51.
            mode: The mode of treating hopping event. Default to "full".

        Return:
             The floating_atom average hopping rate and average hopping distance.
        """
        nvt_run = self.wrapped_run
        floating_atoms = nvt_run.select_atoms(self.select_dict.get(floating_atom))
        freqs = []
        hopping_distance = []
        for ion in tqdm(floating_atoms[:]):
            neighbor_trj = neighbor_distance(nvt_run, ion, run_start, run_end, binding_site, self.select_dict, distance)
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
                coord_ion = nvt_run.trajectory[step + run_start][ion.id - 1]
                coords.append(coord_ion)
            if len(coords) > 1:
                dists = []
                for i in range(len(coords) - 1):
                    dist = distance_array(coords[i + 1], coords[i], box=self.get_nvt_dimension())[0][0]
                    dists.append(dist)
                ion_mean_dists = np.mean(dists)
                hopping_distance.append(ion_mean_dists)
            freqs.append(freq)
        return np.mean(freqs), np.mean(hopping_distance)

    def shell_evolution(
        self,
        distance_dict: Dict[str, float],
        run_start: int,
        run_end: int,
        lag_step: int,
        binding_cutoff: float,
        hopping_cutoff: float,
        smooth: int = 51,
        cool: int = 0,
        binding_site: str = "anion",
        center_atom: str = "cation",
        duplicate_run: Optional[List[MdRun]] = None,
    ) -> Dict[str, Dict[str, Union[int, np.ndarray]]]:
        """Calculates the coordination number evolution of species around {center_atom} as a function of time,
        the coordination numbers are averaged over all time steps around events when the center_atom
        hopping to and hopping out from the {binding_site}. If {duplicate_run} is given, it is also averaged over
        all duplicate runs.

        Args:
            distance_dict: A dict of coordination cutoff distance of interested species.
            run_start: Start frame of analysis.
            run_end: End frame of analysis.
            lag_step: time steps to track before and after the hopping event
            binding_cutoff: Binding cutoff distance.
            hopping_cutoff: Detaching cutoff distance.
            smooth: The length of the smooth filter window. Default to 51.
            cool: The cool down timesteps between hopping in and hopping out.
            center_atom:
            binding_site: The select_dict key of the binding site. Default to "anion".
            duplicate_run: Default to None.

        Return:
            A dictionary containing the number of trj logged, the averaged coordination number and standard deviation
            for each species, and the corresponding time sequence.
        """
        in_list: Dict[str, List[np.ndarray]] = dict()
        out_list: Dict[str, List[np.ndarray]] = dict()
        for k in list(distance_dict):
            in_list[k] = []
            out_list[k] = []
        process_evol(
            self.wrapped_run,
            self.select_dict,
            in_list,
            out_list,
            distance_dict,
            run_start,
            run_end,
            lag_step,
            binding_cutoff,
            hopping_cutoff,
            smooth,
            cool,
            binding_site,
            center_atom,
        )
        if duplicate_run is not None:
            for run in duplicate_run:
                process_evol(
                    run.wrapped_run,
                    run.select_dict,
                    in_list,
                    out_list,
                    distance_dict,
                    run_start,
                    run_end,
                    lag_step,
                    binding_cutoff,
                    hopping_cutoff,
                    smooth,
                    cool,
                    binding_site,
                    center_atom,
                )
        cn_dict = dict()
        cn_dict["time"] = np.array([i * self.time_step - lag_step * self.time_step for i in range(lag_step * 2 + 1)])
        for k in list(distance_dict):
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
        run_start: int,
        run_end: int,
        cluster_center: str,
        cluster_terminal: str,
        binding_cutoff: float,
        hopping_cutoff: float,
        floating_atom: str = "cation",
        cartesian_by_ref: np.ndarray = None,
        sym_dict: Dict[str, List[np.ndarray]] = None,
        sample: Optional[int] = None,
        smooth: int = 51,
    ) -> np.ndarray:
        """Calculates the heatmap matrix of floating ion around a cluster

        Args:
            run_start: Start frame of analysis.
            run_end: End frame of analysis.
            cluster_center: The center atom species of the cluster.
            cluster_terminal: The terminal atom species of the cluster
                (typically the binding site for the floating ion).
            binding_cutoff: Binding cutoff distance.
            hopping_cutoff: Detaching cutoff distance.
            floating_atom: The species of the floating ion.
            cartesian_by_ref: Transformation matrix between cartesian
                and reference coordinate systems. Default to None.
            sym_dict: Dictionary of symmetry operation dictionary. Default to None.
            sample: Number of samples desired. Default to None, which is no sampling.
            smooth: The length of the smooth filter window. Default to 51.

        Return:
            An array of coordinates.
        """
        nvt_run = self.wrapped_run
        floating_atoms = nvt_run.select_atoms(self.select_dict.get(floating_atom))
        coord_list = np.array([[0, 0, 0]])
        for atom in tqdm(floating_atoms[:]):
            neighbor_trj = neighbor_distance(
                nvt_run, atom, run_start, run_end, cluster_center, self.select_dict, binding_cutoff
            )
            sites, freq, steps = find_nearest(
                neighbor_trj, self.time_step, binding_cutoff, hopping_cutoff, smooth=smooth
            )
            if cartesian_by_ref is None:
                cartesian_by_ref = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            coords = heat_map(
                nvt_run,
                atom,
                sites,
                cluster_terminal,
                cartesian_by_ref,
                run_start,
                run_end,
            )
            coord_list = np.concatenate((coord_list, coords), axis=0)
        coord_list = coord_list[1:]
        if sym_dict:
            return get_full_coords(coord_list, sample=sample, **sym_dict)
        else:
            return get_full_coords(coord_list, sample=sample)

    def get_cluster_distance(
        self, run_start: int, run_end: int, neighbor_cutoff: float, cluster_center: str = "center"
    ) -> np.floating:
        """Calculates the average distance of the center of clusters/molecules

        Args:
            run_start: Start frame of analysis.
            run_end: End frame of analysis.
            neighbor_cutoff: Upper limit of first nearest neighbor.
            cluster_center: species name of cluster center. Default to "center".

        Return:
            The averaged distance.
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
