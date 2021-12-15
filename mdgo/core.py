# coding: utf-8
# Copyright (c) Tingzheng Hou.
# Distributed under the terms of the MIT License.

"""
This module implements two core class MdRun and MdJob
for molecular dynamics simulation analysis and job setup.
"""
from __future__ import annotations
from typing import Union, Dict, Tuple, List, Optional
import MDAnalysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from MDAnalysis import Universe
from MDAnalysis.analysis.distances import distance_array
from MDAnalysis.lib.distances import capped_distance
from tqdm.notebook import tqdm
from mdgo.util import (
    mass_to_name,
    assign_name,
    assign_resname,
    res_dict_from_select_dict,
    res_dict_from_datafile,
    select_dict_from_resname,
)
from mdgo.conductivity import calc_cond_msd, conductivity_calculator, choose_msd_fitting_region, get_beta
from mdgo.coordination import (
    concat_coord_array,
    num_of_neighbor,
    num_of_neighbor_simple,
    num_of_neighbor_specific,
    angular_dist_of_neighbor,
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
    A core class for MD results analysis. TODO: add support for 2d and dimension selection.

    Args:
        wrapped_run: The Universe object of wrapped trajectory.
        unwrapped_run: The Universe object of unwrapped trajectory.
        nvt_start: NVT start time step.
        time_step: Timestep between each frame, in ps.
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
        units: unit system (currently 'real' and 'lj' are supported)
    """

    def __init__(
        self,
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
        units="real",
    ):
        """
        Base constructor. This is a low level constructor designed to work with
         parsed data (``Universe``) or other bridging objects (``CombinedData``). Not
        recommended to use directly.
        """

        self.wrapped_run = wrapped_run
        self.unwrapped_run = unwrapped_run
        self.nvt_start = nvt_start
        self.time_step = time_step
        self.temp = temperature
        self.name = name
        self.atom_names = mass_to_name(self.wrapped_run.atoms.masses)
        if not hasattr(self.wrapped_run.atoms, "names") or not hasattr(self.unwrapped_run.atoms, "names"):
            assign_name(self.wrapped_run, self.atom_names)
            assign_name(self.unwrapped_run, self.atom_names)
        if not hasattr(self.wrapped_run.atoms, "resnames") or not hasattr(self.unwrapped_run.atoms, "resnames"):
            if res_dict is None:
                assert select_dict is not None, "Either one of select_dict or res_dict should be given."
                res_dict = res_dict_from_select_dict(self.wrapped_run, select_dict)
            assign_resname(self.wrapped_run, res_dict)
            assign_resname(self.unwrapped_run, res_dict)
        if select_dict is None:
            self.select_dict = select_dict_from_resname(self.wrapped_run)
        else:
            self.select_dict = select_dict
        self.nvt_steps = self.wrapped_run.trajectory.n_frames
        self.time_array = np.array([i * self.time_step for i in range(self.nvt_steps - self.nvt_start)])
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
        self.units = units

    @classmethod
    def from_lammps(
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
        units: str = "real",
    ):
        """
        Constructor from lammps data file and wrapped and unwrapped trajectory dcd file.

        Args:
            data_dir: Path to the data file.
            wrapped_dir: Path to the wrapped dcd file.
            unwrapped_dir: Path to the unwrapped dcd file.
            nvt_start: NVT start time step.
            time_step: LAMMPS timestep in ps.
            name: Name of the MD run.
            select_dict: A dictionary of species selection.
            res_dict: A dictionary of resnames.
            cation_name: Name of cation. Default to "cation".
            anion_name: Name of anion. Default to "anion".
            cation_charge: Charge of cation. Default to 1.
            anion_charge: Charge of anion. Default to 1.
            temperature: Temperature of the MD run. Default to 298.15.
            cond: Whether to calculate conductivity MSD. Default to True.
            units: unit system (currently 'real' and 'lj' are supported)
        """
        if res_dict is None:
            res_dict = res_dict_from_datafile(data_dir)
        wrapped_run = MDAnalysis.Universe(data_dir, wrapped_dir, format="LAMMPS")
        unwrapped_run = MDAnalysis.Universe(data_dir, unwrapped_dir, format="LAMMPS")

        return cls(
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
            units=units,
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
        d = []
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
        cond_array = calc_cond_msd(
            nvt_run,
            anions,
            cations,
            self.nvt_start,
            self.cation_charge,
            self.anion_charge,
        )
        return cond_array

    def choose_cond_fit_region(self) -> tuple:
        """Computes the optimal fitting region (linear regime) of conductivity MSD.

        Args:
            msd (numpy.array): mean squared displacement

        Returns at tuple with the start of the fitting regime (int), end of the
        fitting regime (int), and the beta value of the fitting regime (float).
        """
        if self.cond_array is None:
            self.cond_array = self.get_cond_array()
        start, end, beta = choose_msd_fitting_region(self.cond_array, self.time_array)
        return start, end, beta

    def plot_cond_array(
        self,
        start: int = -1,
        end: int = -1,
        *runs: MdRun,
        reference: bool = True,
    ):
        """Plots the conductivity MSD as a function of time.
        If no fitting region (start, end) is provided, computes the optimal
        fitting region based on the portion of the MSD with greatest
        linearity.

        Args:
            start (int): Start time step for fitting.
            end (int): End time step for fitting.
            runs (MdRun): Other runs to be compared in the same plot.
            reference (bool): Whether to plot reference line.
                Default to True.
            units (str): unit system (currently 'real' and 'lj' are supported)
        """
        if self.cond_array is None:
            self.cond_array = self.get_cond_array()
        if start == -1 and end == -1:
            start, end, _ = choose_msd_fitting_region(self.cond_array, self.time_array)
        colors = ["g", "r", "c", "m", "y", "k"]
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.loglog(
            self.time_array,
            self.cond_array,
            color="b",
            lw=2,
            label=self.name,
        )
        for i, run in enumerate(runs):
            ax.loglog(
                run.time_array,
                run.cond_array,
                color=colors[i],
                lw=2,
                label=run.name,
            )
        if reference:
            slope_guess = (self.cond_array[int(np.log(len(self.time_array)) / 2)] - self.cond_array[5]) / (
                self.time_array[int(np.log(len(self.time_array)) / 2)] - self.time_array[5]
            )
            ax.loglog(self.time_array[start:end], np.array(self.time_array[start:end]) * slope_guess * 2, "k--")
        if self.units == "real":
            ax.set_ylabel("MSD (A$^2$)")
            ax.set_xlabel("Time (ps)")
        elif self.units == "lj":
            ax.set_ylabel("MSD ($\\sigma^2$)")
            ax.set_xlabel("Time ($\\tau$)")
        else:
            raise ValueError("units selection not supported")
        ax.set_ylim(min(np.abs(self.cond_array[1:])) * 0.9, max(np.abs(self.cond_array)) * 1.2)
        ax.legend()
        fig.show()

    def get_conductivity(self, start: int = -1, end: int = -1) -> float:
        """Calculates the Green-Kubo (GK) conductivity given fitting region.
        If no fitting region (start, end) is provided, computes the optimal
        fitting region based on the portion of the MSD with greatest
        linearity.

        Args:
            start (int): Start time step for fitting MSD.
            end (int): End time step for fitting MSD.

        Print and return conductivity.
        """
        if start == -1 and end == -1:
            start, end, beta = choose_msd_fitting_region(self.cond_array, self.time_array)
        else:
            beta, _ = get_beta(self.cond_array, self.time_array, start, end)
        # print info on fitting
        time_units = ""
        if self.units == "real":
            time_units = "ps"
        elif self.units == "lj":
            time_units = "tau"
        print(f"Start of linear fitting regime: {start} ({self.time_array[start]} {time_units})")
        print(f"End of linear fitting regime: {end} ({self.time_array[end]} {time_units})")
        print(f"Beta value (fit to MSD = t^\u03B2): {beta} (\u03B2 = 1 in the diffusive regime)")
        cond = conductivity_calculator(
            self.time_array, self.cond_array, self.nvt_v, self.name, start, end, self.temp, self.units
        )
        return cond

    def coord_num_array_single_species(
        self,
        species: str,
        distance: float,
        run_start: int,
        run_end: int,
        center_atom: str = "cation",
    ) -> np.ndarray:
        """Calculates the coordination number array of one ``species`` around the interested ``center_atom``.

        Args:
            species: The neighbor species.
            distance: The coordination cutoff distance.
            run_start: Start frame of analysis.
            run_end: End frame of analysis.
            center_atom: The solvation shell center atom. Default to "cation".

        Return:
             An array of coordination number in the frame range.
        """
        nvt_run = self.wrapped_run
        distance_dict = {species: distance}
        center_atoms = nvt_run.select_atoms(self.select_dict.get(center_atom))
        num_array = concat_coord_array(
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
        """Calculates the coordination number array of multiple species around the interested ``center_atom``.

        Args:
            distance_dict: A dict of coordination cutoff distance of the neighbor species.
            run_start: Start frame of analysis.
            run_end: End frame of analysis.
            center_atom: The solvation shell center atom. Default to "cation".

        Return:
             A diction containing the coordination number sequence of each specified neighbor species
             and the total coordination number sequence in the specified frame range.
        """
        nvt_run = self.wrapped_run
        center_atoms = nvt_run.select_atoms(self.select_dict.get(center_atom))
        num_array_dict = concat_coord_array(
            nvt_run,
            num_of_neighbor,
            center_atoms,
            distance_dict,
            self.select_dict,
            run_start,
            run_end,
        )
        return num_array_dict

    def coord_num_array_specific(
        self,
        distance_dict: Dict[str, float],
        run_start: int,
        run_end: int,
        center_atom: str = "cation",
        counter_atom: str = "anion",
    ) -> Dict[str, np.ndarray]:
        """Calculates the coordination number array of multiple species of specific
        coordination types (SSIP, CIP, AGG).

        Args:
            distance_dict: A dict of coordination cutoff distance of the neighbor species.
            run_start: Start frame of analysis.
            run_end: End frame of analysis.
            center_atom: The solvation shell center atom. Default to "cation".
            counter_atom: The neighbor counter ion species. Default to "anion".

        Return:
             A diction containing the coordination number sequence of each specified neighbor species
             and the total coordination number sequence in the specified frame range.
        """
        nvt_run = self.wrapped_run
        center_atoms = nvt_run.select_atoms(self.select_dict.get(center_atom))
        num_array_dict = concat_coord_array(
            nvt_run,
            num_of_neighbor_specific,
            center_atoms,
            distance_dict,
            self.select_dict,
            run_start,
            run_end,
            counter_atom=counter_atom,
        )
        return num_array_dict

    def write_solvation_structure(
        self,
        distance_dict: Dict[str, float],
        run_start: int,
        run_end: int,
        structure_code: int,
        write_freq: float,
        write_path: str,
        center_atom: str = "cation",
    ):
        """Writes out a series of desired solvation structures as ``*.xyz`` files

        Args:
            distance_dict: A dict of coordination cutoff distance of the neighbor species.
            run_start: Start frame of analysis.
            run_end: End frame of analysis.
            structure_code: An integer code representing the solvation
                structure, for example, 221 is two species A, two species B
                and one species C with the same order as in the ``distance_dict``.
            write_freq: Probability to write out files.
            write_path: Path to write out files.
            center_atom: The solvation shell atom. Default to "cation".
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
            )

    def coord_type_array(
        self,
        distance: float,
        run_start: int,
        run_end: int,
        center_atom: str = "cation",
        counter_atom: str = "anion",
    ) -> np.ndarray:
        """Calculates the solvation structure type (1 for SSIP, 2 for CIP,
        3 for AGG) array of the solvation structure ``center_atom`` (typically the cation).

        Args:
            distance: The coordination cutoff distance.
            run_start: Start frame of analysis.
            run_end: End frame of analysis.
            center_atom: The solvation shell center atom. Default to "cation".
            counter_atom: The neighbor counter ion species. Default to "anion".

        Return:
            An array of the solvation structure type in the specified frame range.
        """
        nvt_run = self.wrapped_run
        distance_dict = {counter_atom: distance}
        center_atoms = nvt_run.select_atoms(self.select_dict.get(center_atom))
        num_array = concat_coord_array(
            nvt_run,
            num_of_neighbor_simple,
            center_atoms,
            distance_dict,
            self.select_dict,
            run_start,
            run_end,
        )["total"]
        return num_array

    def angle_array(
        self,
        distance_dict: Dict[str, float],
        run_start: int,
        run_end: int,
        center_atom: str = "cation",
        cip=True,
    ):
        """
        Calculates the angle of a-c-b in the specified frame range.

        Args:
            distance_dict: A dict of coordination cutoff distance of the neighbor species.
                The dictionary key must be in the order of a, b, where a is the neighbor species
                used for determining coordination type, b is the other neighbor species, and the
                corresponding values are cutoff distance of a->c and b->c, where c is the center species.
            run_start: Start frame of analysis.
            run_end: End frame of analysis.
            center_atom: The center atom species.
            cip: Only includes contact ion pair structures with only one `a` and one `c` atoms.
                Default to True.

        Returns:
            An array of angles of a-c-b in the specified frames.
        """
        nvt_run = self.wrapped_run
        center_atoms = nvt_run.select_atoms(self.select_dict.get(center_atom))
        assert len(distance_dict) == 2, "Only distance a->c, b->c shoud be specified in the distance_dict."
        distance_dict[center_atom] = list(distance_dict.values())[0]
        ang_array = concat_coord_array(
            nvt_run,
            angular_dist_of_neighbor,
            center_atoms,
            distance_dict,
            self.select_dict,
            run_start,
            run_end,
            cip=cip,
        )["total"]
        return ang_array

    def coordination(
        self,
        species: str,
        distance: float,
        run_start: int,
        run_end: int,
        center_atom: str = "cation",
    ) -> pd.DataFrame:
        """Tabulates the coordination number distribution of one species
        around the solvation structure ``center_atom``.

        Args:
            species: The neighbor species.
            distance: The coordination cutoff distance.
            run_start: Start frame of analysis.
            run_end: End frame of analysis.
            center_atom: The solvation shell center atom. Default to "cation".

        Return:
             A dataframe of the species coordination number and corresponding percentage.
        """
        num_array = self.coord_num_array_single_species(species, distance, run_start, run_end, center_atom=center_atom)
        shell_component, shell_count = np.unique(num_array.flatten(), return_counts=True)
        combined = np.vstack((shell_component, shell_count)).T

        item_name = "Num of " + species + " within " + str(distance) + " " + "\u212B"
        item_list = []
        percent_list = []
        for i in range(len(combined)):
            item_list.append(str(int(combined[i, 0])))
            percent_list.append(f"{(combined[i, 1] / combined[:, 1].sum() * 100):.4f}%")
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
        """Calculates the integral of the radial distribution function of selected species around the ``center_atom``.

        Args:
            distance_dict: A dict of coordination cutoff distance of the neighbor species.
            run_start: Start frame of analysis.
            run_end: End frame of analysis.
            center_atom: The solvation shell center atom to calculate the radial distribution for. Default to "cation".

        Return:
             A dataframe of the species and the coordination number.
        """
        cn_values = self.coord_num_array_multi_species(distance_dict, run_start, run_end, center_atom=center_atom)
        item_name = "Species in first solvation shell"
        item_list = []
        cn_list = []
        for kw, val in cn_values.items():
            if kw != "total":
                shell_component, shell_count = np.unique(val.flatten(), return_counts=True)
                cn = (shell_component * shell_count / shell_count.sum()).sum()
                item_list.append(kw)
                cn_list.append(cn)
        df_dict = {item_name: item_list, "CN": cn_list}
        df = pd.DataFrame(df_dict)
        return df

    def coordination_type(
        self,
        distance: float,
        run_start: int,
        run_end: int,
        center_atom: str = "cation",
        counter_atom: str = "anion",
    ) -> pd.DataFrame:
        """Tabulates the percentage of each solvation structures (CIP/SSIP/AGG)

        Args:
            distance: The coordination cutoff distance.
            run_start: Start frame of analysis.
            run_end: End frame of analysis.
            center_atom: The solvation shell center atom. Default to "cation".
            counter_atom: The neighbor counter ion species. Default to "anion".

        Return:
             A dataframe of the solvation structure and percentage.
        """
        num_array = self.coord_type_array(
            distance, run_start, run_end, center_atom=center_atom, counter_atom=counter_atom
        )

        shell_component, shell_count = np.unique(num_array.flatten(), return_counts=True)
        combined = np.vstack((shell_component, shell_count)).T

        item_name = "Solvation structure"
        item_dict = {"1": "ssip", "2": "cip", "3": "agg"}
        item_list = []
        percent_list = []
        for i in range(len(combined)):
            item = str(int(combined[i, 0]))
            item_list.append(item_dict.get(item))
            percent_list.append(f"{(combined[i, 1] / combined[:, 1].sum() * 100):.4f}%")
        df_dict = {item_name: item_list, "Percentage": percent_list}
        df = pd.DataFrame(df_dict)
        return df

    def coordination_specific(
        self,
        distance_dict: Dict[str, float],
        run_start: int,
        run_end: int,
        center_atom: str = "cation",
        counter_atom: str = "anion",
    ) -> pd.DataFrame:
        """Calculates the integral of the coordiantion number of selected species
        in each type of solvation structures (CIP/SSIP/AGG)

        Args:
            distance_dict: A dict of coordination cutoff distance of the neighbor species.
            run_start: Start frame of analysis.
            run_end: End frame of analysis.
            center_atom: The solvation shell center atom. Default to "cation".
            counter_atom: The neighbor counter ion species. Default to "anion".

        Return:
             A dataframe of the solvation structure and percentage.
        """
        cn_values = self.coord_num_array_specific(
            distance_dict, run_start, run_end, center_atom=center_atom, counter_atom=counter_atom
        )
        item_name = "Species in first solvation shell"
        item_list = []
        ssip_list = []
        cip_list = []
        agg_list = []
        for kw, val in cn_values.items():
            if kw != "total":
                shell_component, shell_count = np.unique(val.flatten(), return_counts=True)
                cn = (shell_component * shell_count / shell_count.sum()).sum()
                if kw.startswith("ssip_"):
                    item_list.append(kw[5:])
                    ssip_list.append(cn)
                elif kw.startswith("cip_"):
                    cip_list.append(cn)
                else:
                    agg_list.append(cn)
        df_dict = {item_name: item_list, "CN in SSIP": ssip_list, "CN in CIP": cip_list, "CN in AGG": agg_list}
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
            species: The species for analysis. Default to "cation".

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
        Calculates the mean square displacement (MSD) of the ``center_atom`` according to coordination states.
        The returned ``free_array`` include the MSD when ``center_atom`` is not coordinated to ``binding_site``.
        The ``attach_array`` includes the MSD when ``center_atom`` is coordinated to ``binding_site``.

        Args:
            distance: The coordination cutoff distance.
            run_start: Start frame of analysis.
            run_end: End frame of analysis.
            largest: The largest time sequence to trace.
            center_atom: The solvation shell center atom. Default to "cation".
            binding_site: The species the ``center_atom`` coordinates to. Default to "anion".

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
            species: The species for analysis. Default to "cation".
        """
        a2 = 1e-20
        ps = 1e-12
        s_m_to_ms_cm = 10
        if percentage != 1:
            d = (msd_array[start] - msd_array[stop]) / (start - stop) / self.time_step / 6 * a2 / ps
            sigma = percentage * d * self.d_to_sigma * s_m_to_ms_cm
            print(f"Diffusivity of {(percentage * 100):.2f}% {species}: {d} m^2/s")
            if species.lower() == "cation" or species.lower() == "li":
                print(f"NE Conductivity of {(percentage * 100):.2f}% {species}: {sigma}mS/cm")
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
            distance_dict: A dict of coordination cutoff distance of the neighbor species.
            run_start: Start frame of analysis.
            run_end: End frame of analysis.
            center_atom: The solvation shell center atom to calculate the ACF for. Default to "cation".

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
        self, times: np.ndarray, acf_avg_dict: Dict[str, np.ndarray], cutoff_time: int
    ) -> Dict[str, np.floating]:
        """Calculates the residence time of selected species around cation

        Args:
            times: The time series.
            acf_avg_dict: A dict of ACFs of each species.
            cutoff_time: Cutoff time for fitting the exponential decay.

        Return:
             The residence time of each species in a dict.
        """
        return fit_residence_time(times, acf_avg_dict, cutoff_time, self.time_step)

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
            center_atom: The solvation shell center atom. Default to "cation".
            species: The neighbor species.
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
        binding_cutoff: float,
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
            binding_cutoff: Binding cutoff distance.
            hopping_cutoff: Hopping out cutoff distance.
            floating_atom: Floating atom species.
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
            neighbor_trj = neighbor_distance(
                nvt_run, ion, run_start, run_end, binding_site, self.select_dict, binding_cutoff
            )
            if mode == "full":
                sites, freq, steps = find_nearest(
                    neighbor_trj, self.time_step, binding_cutoff, hopping_cutoff, smooth=smooth
                )
            elif mode == "free":
                sites, freq, steps = find_nearest_free_only(
                    neighbor_trj, self.time_step, binding_cutoff, hopping_cutoff, smooth=smooth
                )
            else:
                raise ValueError("invalid mode")
            coords = []
            for step in steps:
                coord_ion = nvt_run.trajectory[step + run_start][ion.index]
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
        """Calculates the coordination number evolution of species around ``center_atom`` as a function of time,
        the coordination numbers are averaged over all time steps around events when the center_atom
        hopping to and hopping out from the ``binding_site``. If ``duplicate_run`` is given, it is also averaged over
        all duplicate runs.

        Args:
            distance_dict: A dict of coordination cutoff distance of the neighbor species.
            run_start: Start frame of analysis.
            run_end: End frame of analysis.
            lag_step: time steps to track before and after the hopping event
            binding_cutoff: Binding cutoff distance.
            hopping_cutoff: Detaching cutoff distance.
            smooth: The length of the smooth filter window. Default to 51.
            cool: The cool down frames between hopping in and hopping out.
            center_atom: The solvation shell center atom. Default to "cation".
            binding_site: The select_dict key of the binding site. Default to "anion".
            duplicate_run: Default to None.

        Return:
            A dictionary containing the number of trj logged, the averaged coordination number and standard deviation
            for each species, and the corresponding time sequence.
        """
        in_list: Dict[str, List[np.ndarray]] = {}
        out_list: Dict[str, List[np.ndarray]] = {}
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
        cn_dict = {}
        cn_dict["time"] = np.array([i * self.time_step - lag_step * self.time_step for i in range(lag_step * 2 + 1)])
        for k in list(distance_dict):
            if "in_count" not in cn_dict:
                cn_dict["in_count"] = np.array(in_list[k]).shape[0]
                cn_dict["out_count"] = np.array(out_list[k]).shape[0]
            k_dict = {}
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
        cluster_terminal: Union[str, List[str]],
        binding_cutoff: float,
        hopping_cutoff: float,
        floating_atom: str = "cation",
        cartesian_by_ref: np.ndarray = None,
        sym_dict: Dict[str, List[np.ndarray]] = None,
        sample: Optional[int] = None,
        smooth: int = 51,
        dim: str = "xyz",
    ) -> np.ndarray:
        """Calculates the heatmap matrix of floating ion around a cluster

        Args:
            run_start: Start frame of analysis.
            run_end: End frame of analysis.
            cluster_center: The center atom species of the cluster.
            cluster_terminal: The selection string for terminal atom species of the cluster
                (typically the binding site for the floating ion). The argument can be a str if
                all the terminal atoms have the same selection string and are equivalent, or a list
                if the terminal atoms are distinct and have different selection strings.
            binding_cutoff: Binding cutoff distance.
            hopping_cutoff: Detaching cutoff distance.
            floating_atom: The floating atom species.
            cartesian_by_ref: Transformation matrix between cartesian
                and reference coordinate systems. Default to None.
            sym_dict: Dictionary of symmetry operation dictionary. Default to None.
            sample: Number of samples desired. Default to None, which is no sampling.
            smooth: The length of the smooth filter window. Default to 51.
            dim: Desired dimensions to calculate heat map

        Return:
            An array of coordinates.
        """
        nvt_run = self.wrapped_run
        floating_atoms = nvt_run.select_atoms(self.select_dict.get(floating_atom))
        if isinstance(cluster_terminal, str):
            terminal_atom_type: Union[str, List[str]] = self.select_dict.get(cluster_terminal, "Not defined")
            assert terminal_atom_type != "Not defined", f"{cluster_terminal} not defined in select_dict"
        else:
            terminal_atom_type = []
            for species in cluster_terminal:
                atom_type = self.select_dict.get(species, "Not defined")
                assert atom_type != "Not defined", f"{species} not defined in select_dict"
                terminal_atom_type.append(atom_type)
        coord_list = np.array([[0, 0, 0]])
        for atom in tqdm(floating_atoms[:]):
            neighbor_trj = neighbor_distance(
                nvt_run, atom, run_start, run_end, cluster_center, self.select_dict, binding_cutoff
            )
            if not bool(neighbor_trj):
                continue
            sites, freq, steps = find_nearest(
                neighbor_trj, self.time_step, binding_cutoff, hopping_cutoff, smooth=smooth
            )
            if cartesian_by_ref is None:
                cartesian_by_ref = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            coords = heat_map(
                nvt_run,
                atom,
                sites,
                terminal_atom_type,
                cartesian_by_ref,
                run_start,
                run_end,
            )
            if not coords.size == 0:
                coord_list = np.concatenate((coord_list, coords), axis=0)
        coord_list = coord_list[1:]
        if sym_dict:
            return get_full_coords(coord_list, sample=sample, dim=dim, **sym_dict)
        return get_full_coords(coord_list, sample=sample, dim=dim)

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
