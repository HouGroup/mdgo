import MDAnalysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import MDAnalysis
from MDAnalysis import transformations
from MDAnalysis.analysis.distances import distance_array
from MDAnalysis.lib.distances import capped_distance
from tqdm import tqdm_notebook
from mdgo.conductivity import calc_cond, conductivity_calculator
from mdgo.coordination import \
    coord_shell_array, num_of_neighbor_one_li, num_of_neighbor_one_li_multi, \
    num_of_neighbor_one_li_simple, trajectory, find_nearest, \
    heat_map, get_full_coords
from mdgo.msd import total_msd, partial_msd, special_msd
from mdgo.residence_time import calc_neigh_corr, fit_residence_time
from mdgo.util import resnames, mass_to_el
from mdgo.rdf import RdfMemoizer
from mdgo.shell_functions import get_counts, get_pair_type, count_dicts, \
    get_radial_shell


class MdRun:

    def __init__(self, data_dir, unwrapped_dir, nvt_start,
                 time_step, name, select_dict, cation_charge=1, anion_charge=-1,
                 temperature=298.5, cond=True):
        """
        Base constructor.

        Args:
            data_dir (str): path to the data file.
            unwrapped_dir (str): dpath to the unwrapped dcd file.
            nvt_start (int): nvt start time step
            time_step (int or float): LAMMPS timestep
            name (str): name of the MD run
            select_dict: a dictionary of species selection
            cation_charge: charge of cation. Default to 1.
            anion_charge: charge of anion. Default to 1.
            cond (bool): Whether to calculate conductivity MSD. Default to True.

        """
        self.u_unwrapped = MDAnalysis.Universe(str(data_dir),
                                               str(unwrapped_dir),
                                               format="LAMMPS")
        self.u_wrapped = MdRun.transform_run(self.u_unwrapped, 'wrap')
        self.rdf_memoizer = RdfMemoizer(self.u_wrapped)
        self.nvt_start = nvt_start
        self.time_step = time_step
        self.name = name
        self.select_dict = select_dict
        self.nvt_steps = self.u_wrapped.trajectory.n_frames
        self.time_array = [i * self.time_step for i in range(self.nvt_steps)]
        self.cation_name = None
        self.anion_name = None
        self.cations = self.u_unwrapped.select_atoms(self.select_dict["cation"])
        self.anion_center = self.u_unwrapped.select_atoms(self.select_dict["anion"])
        self.anions = self.anion_center.residues.atoms
        self.cation_charge = cation_charge
        self.anion_charge = anion_charge
        self.electrolytes = None  # TODO: extract electrolyte and anion_center from select_dict
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
    def auto_constructor(cls, data_dir, unwrapped_dir, nvt_start,
                         time_step, name, residue_mass_dict, cation_name,
                         anion_name, anion_central_atom, electrolyte_names,
                         cation_charge=1, anion_charge=-1, temperature=298.5,
                         cond=True):

        empty_select_dict = {"cation": "", "anion": ""}
        run = MdRun(data_dir, unwrapped_dir, nvt_start, time_step,
                    name, empty_select_dict, cation_charge=cation_charge,
                    anion_charge=anion_charge, temperature=temperature,
                    cond=cond)
        select_dict = {"cation": f"resname {cation_name}",
                       "anion": f"resname {anion_name}"}
        electrolyte_selections = {name: f"resname {name}"
                                  for name in electrolyte_names}
        run.select_dict = {**select_dict, **electrolyte_selections}
        run.name_residues(residue_mass_dict)
        run.cations = run.u_unwrapped.select_atoms(f"resname {cation_name}")
        run.anions = run.u_unwrapped.select_atoms(f"resname {anion_name}")
        run.anion_center = \
            run.u_unwrapped.select_atoms(f"resname {anion_name} and "
                                         f"name {anion_central_atom}")
        run.electrolytes = {elyte: run.u_unwrapped.select_atoms(f'resname {elyte}')
                            for elyte in electrolyte_names}
        run.u_wrapped = run.transform_run(run.u_unwrapped, 'wrap')
        return run

    def get_cation_molarity(self):
        A3_2_L3 = 1e-27
        mol_cations = len(self.cations.residues) / 6.02214e23
        liters = self.nvt_v * A3_2_L3
        return round(mol_cations / liters, 2)

    @staticmethod
    def transform_run(universe, transformation):
        wrapped_run = universe.copy()
        all_atoms = wrapped_run.atoms
        assert transformation in ["wrap", "unwrap"]
        if transformation == "wrap":
            transform = transformations.wrap(all_atoms)
        elif transformation == "unwrap":
            transform = transformations.unwrap(all_atoms)
        wrapped_run.trajectory.add_transformations(transform)
        return wrapped_run

    def name_residues(self, residue_mass_dict):
        atom_names = mass_to_el(self.u_unwrapped.atoms.masses)
        self.u_unwrapped.add_TopologyAttr('name', values=atom_names)
        residue_names = resnames(self.u_unwrapped, residue_mass_dict)
        self.u_unwrapped.add_TopologyAttr('resname', values=residue_names)

    def get_init_dimension(self):
        """
        Returns the initial box dimension.
        """
        return self.wrapped_run.dimensions

    def get_nvt_dimension(self):
        """
        Returns the equilibrium box dimension.
        """
        return self.wrapped_run.trajectory[-1].dimensions

    def get_cond_array(self):
        """ Calculates the conductivity "mean square displacement".

        Returns an array of MSD values for each time in the trajectory.
        """
        nvt_run = self.unwrapped_run
        cations = nvt_run.select_atoms(self.select_dict["cation"])
        anions = nvt_run.select_atoms(self.select_dict["anion"])
        cond_array = calc_cond(nvt_run, anions, cations, self.nvt_start,
                               self.cation_charge, self.anion_charge)
        return cond_array

    def plot_cond_array(self, start, end, *runs):
        """ Plots the conductivity MSD as a function of time

            Args:
                start (int): start time step
                end (int): end time step
                runs (MdRun): other runs to be compared in the same plot
        """
        if self.cond_array is None:
            self.cond_array = self.get_cond_array()
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
        """ Calculates the Green-Kubo (GK) conductivity

        Args:
            start (int): start time step
            end (int): end time step

        Print conductivity in mS/cm.
        """
        conductivity_calculator(self.time_array, self.cond_array,
                                self.nvt_v, self.name, start, end)
        return None

    def coord_num_array_one_species(self, species, distance,
                                    run_start, run_end):
        """ Calculates the coordination number array of one species around
        the cation.

        Args:
            species (str): the interested species
            distance (int or float): the coordination cutoff distance
            run_start (int): start time step
            run_end (int): end time step

        Returns an array of the species coordination number for each time
        in the trajectory.
        """
        nvt_run = self.wrapped_run
        li_atoms = nvt_run.select_atoms(self.select_dict["cation"])
        num_array = coord_shell_array(nvt_run, num_of_neighbor_one_li,
                                      li_atoms, species, self.select_dict,
                                      distance, run_start, run_end)
        return num_array

    def coord_num_array_multi_species(self, species, distances,
                                      run_start, run_end):
        """ Calculates the coordination number array of multiple species around
        the cation

        Args:
            species (str): the interested species
            distances (dict): a dict of coordination cutoff distance of species
            run_start (int): start time step
            run_end (int): end time step

        Returns a python dict of arrays of coordination numbers of each species
         for each time in the trajectory.
        """
        nvt_run = self.wrapped_run
        li_atoms = nvt_run.select_atoms(self.select_dict["cation"])
        num_array = coord_shell_array(nvt_run, num_of_neighbor_one_li_multi,
                                      li_atoms, species, self.select_dict,
                                      distances, run_start, run_end)
        return num_array

    def coord_num_array_simple(self, species, distance, run_start, run_end):
        """ Calculates the solvation structure type (1 for ssIP, 2 for CIP,
        3 for AGG) array of the cation

        Args:
            species (str): the interested species
            distance (int or float): the coordination cutoff distance
            run_start (int): start time step
            run_end (int): end time step

        Returns an array of the solvation structure type
        for each time in the trajectory.
        """
        nvt_run = self.wrapped_run
        li_atoms = nvt_run.select_atoms(self.select_dict["cation"])
        num_array = coord_shell_array(nvt_run, num_of_neighbor_one_li_simple,
                                      li_atoms, species, self.select_dict,
                                      distance, run_start, run_end)
        return num_array

    def coordination_one_species(self, species, distance, run_start, run_end):
        """ Tabulates the percentage of each coordination number of one species
        around the cation

        Args:
            species (str): the interested species
            distance (int or float): the coordination cutoff distance
            run_start (int): start time step
            run_end (int): end time step

        Returns pandas.dataframe of the species coordination number
        and corresponding percentage.
        """
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

    def rdf_integral(self, species, distances, run_start, run_end):
        """ Calculate the integral of the radial distribution function of
        selected species

        Args:
            species (str): the interested species
            distances (dict): a dict of coordination cutoff distance of species
            run_start (int): start time step
            run_end (int): end time step

        Returns pandas.dataframe of the species and the coordination number.
        """

        cn_values = self.coord_num_array_multi_species(species, distances,
                                                       run_start, run_end)
        item_name = "species in first solvation shell"
        item_list = []
        cn_list = []
        for kw in cn_values.keys():
            if kw != "total":
                shell_component, shell_count \
                    = np.unique(cn_values[kw].flatten(), return_counts=True)
                cn = (shell_component * shell_count / shell_count.sum()).sum()
                item_list.append(kw)
                cn_list.append(cn)
        df_dict = {item_name: item_list, self.name: cn_list}
        df = pd.DataFrame(df_dict)
        return df

    def shell_simple(self, species, distance, run_start, run_end):
        """ Tabulates the percentage of each solvation structures (CIP/SSIP/AGG)

        Args:
            species (str): The interested species.
            distance (int or float): The coordination cutoff distance.
            run_start (int): Start time step.
            run_end (int): End time step.

        Returns pandas.dataframe of the solvation structure and percentage.
        """
        num_array = self.coord_num_array_simple(species, distance,
                                                run_start, run_end)

        shell_component, shell_count = np.unique(num_array.flatten(),
                                                 return_counts=True)
        combined = np.vstack((shell_component, shell_count)).T

        item_name = "solvation structure"
        item_dict = {"1": "ssip", "2": "cip", "3": "agg"}
        item_list = []
        percent_list = []
        for i in range(len(combined)):
            item = str(int(combined[i, 0]))
            item_list.append(item_dict[item])
            percent_list.append(str("%.4f" % (combined[i, 1] /
                                              combined[:, 1].sum() * 100))
                                + '%')
        df_dict = {item_name: item_list, 'Percentage': percent_list}
        df = pd.DataFrame(df_dict)
        return df

    def get_msd_all(self, start=None, stop=None, fft=True):
        """ Calculates the mean square displacement (MSD) of the cation

        Args:
            start (int): start time step
            stop (int): end time step
            fft (bool): Whether to use fft to calculate msd. Default to True.

        Returns an array of MSD values for each time in the trajectory
        """
        msd_array = total_msd(self.unwrapped_run, start=start, stop=stop,
                              select=self.select_dict["cation"], fft=fft)
        return msd_array

    def get_msd_by_length(self, distance, run_start, run_end):
        nvt_run = self.unwrapped_run
        li_atoms = nvt_run.select_atoms(self.select_dict["cation"])
        free_array, attach_array = special_msd(nvt_run, li_atoms,
                                               self.select_dict, distance,
                                               run_start, run_end)
        return free_array, attach_array

    def get_msd_partial(self, distance, run_start, run_end, largest=1000):
        nvt_run = self.unwrapped_run
        li_atoms = nvt_run.select_atoms(self.select_dict["cation"])
        free_array, attach_array = partial_msd(nvt_run, li_atoms, largest,
                                               self.select_dict, distance,
                                               run_start, run_end)
        return free_array, attach_array

    def get_d(self, msd_array, start, stop, percentage=1):
        """ Calculates the self-diffusion coefficient of the cation and
        the Nernst-Einstein conductivity

        Args:
            msd_array (numpy.array): msd array
            start (int): start time step
            stop (int): end time step
            percentage (int or float): The percentage of the cation.
                Default to 1.

        Print self-diffusion coefficient in m^2/s and NE conductivity in mS/cm.
        """
        a2 = 1e-20
        ps = 1e-12
        s_m_to_ms_cm = 10
        if percentage != 1:
            d = (msd_array[start] - msd_array[stop]) \
                / (start-stop) / self.time_step / 6 * a2 / ps
            sigma = percentage * d * self.d_to_sigma * s_m_to_ms_cm
            print("Diffusivity of", "%.2f" % (percentage * 100) + "% Li: ",
                  d, "m^2/s")
            print("NE Conductivity of", "%.2f" % (percentage * 100) + "% Li: ",
                  sigma, "mS/cm")
        else:
            d = (msd_array[start] - msd_array[stop]) \
                / (start - stop) / self.time_step / 6 * a2 / ps
            sigma = d * self.d_to_sigma * s_m_to_ms_cm
            print("Diffusivity of all Li:", d, "m^2/s")
            print("NE Conductivity of all Li:", sigma, "mS/cm")

    def get_neighbor_corr(self, species_list, distance, run_start, run_end):
        """ Calculates the neighbor auto-corelation function (ACF)
        of selected species around cation

        Args:
            species_list (list): list of species name
            distance (int or float): cutoff distance of neighbor
            run_start: start time step
            run_end (int): end time step

        Returns an array of the time series and a dict of ACFs of each species.
        """
        return calc_neigh_corr(self.wrapped_run, species_list, self.select_dict,
                               distance, self.time_step, run_start, run_end)

    @staticmethod
    def get_residence_time(species_list, times, acf_avg_dict, cutoff_time):
        """ Calculates the residence time of selected species around cation

        Args:
            species_list (list): list of species name
            times (np.ndarray): the time series
            acf_avg_dict: a dict of ACFs of each species
            cutoff_time (int): cutoff time for fitting the exponential decay

        Returns the residence time of each species.
        """
        return fit_residence_time(times, species_list,
                                  acf_avg_dict, cutoff_time)

    def get_neighbor_trj(self, run_start, run_end, li_atom, species, distance):
        """ Calculates the distance of cation-neighbor as a function of time

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
        return trajectory(self.wrapped_run, li_atom, run_start, run_end,
                          species, self.select_dict, distance)

    def get_hopping_freq_dist(self, run_start, run_end, species, distance,
                              hopping_cutoff, smooth=51):
        """ Calculates the cation hopping rate and hopping distance

        Args:
            run_start (int): Start time step.
            run_end (int): End time step.
            species (str): cation Binding site species.
            distance (int or float): Binding cutoff distance.
            hopping_cutoff: (int or float): Detaching cutoff distance.
            smooth (int): The length of the smooth filter window. Default to 51.

        Returns the cation average hopping rate and average hopping distance.
        """
        nvt_run = self.wrapped_run
        li_atoms = nvt_run.select_atoms(self.select_dict["cation"])
        freqs = []
        hopping_distance = []
        for li in tqdm_notebook(li_atoms[:]):
            neighbor_trj = trajectory(nvt_run, li, run_start, run_end, species,
                                      self.select_dict, distance)
            sites, freq, steps = find_nearest(neighbor_trj, self.time_step,
                                              distance, hopping_cutoff,
                                              smooth=smooth)

            coords = []
            for step in steps:
                coord_li = nvt_run.trajectory[step + 1000][li.id - 1]
                coords.append(coord_li)
            if len(coords) > 1:
                dists = []
                for i in range(len(coords) - 1):
                    dist = distance_array(coords[i + 1], coords[i],
                                          box=self.get_nvt_dimension())[0][0]
                    dists.append(dist)
                li_mean_dists = np.mean(dists)
                hopping_distance.append(li_mean_dists)
            freqs.append(freq)
        return np.mean(freqs), np.mean(hopping_distance)

    def get_heat_map(self, run_start, run_end, species, distance,
                     hopping_cutoff, cartesian_by_ref=None, bind_atom_type=None,
                     sym_dict=None, sample=None, smooth=51):
        """ Calculates the heatmap matrix of cation around a cluster/molecule

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
        li_atoms = nvt_run.select_atoms(self.select_dict["cation"])
        coord_list = np.array([[0, 0, 0]])
        for li in tqdm_notebook(li_atoms[:]):
            neighbor_trj = trajectory(nvt_run, li, run_start, run_end, species,
                                      self.select_dict, distance)
            sites, freq, steps = find_nearest(neighbor_trj, self.time_step,
                                              distance, hopping_cutoff,
                                              smooth=smooth)
            if bind_atom_type is None:
                bind_atom_type = self.select_dict["anion"]
            if cartesian_by_ref is None:
                cartesian_by_ref = np.array([[1, 0, 0],
                                             [0, 1, 0],
                                             [0, 0, 1]])
            coords = heat_map(nvt_run, li, sites, 4, bind_atom_type,
                              cartesian_by_ref, run_start, run_end)
            coord_list = np.concatenate((coord_list, coords), axis=0)
        if sym_dict:
            return get_full_coords(coord_list, **sym_dict, sample=sample)
        else:
            return get_full_coords(coord_list, sample=sample)

    def get_cluster_distance(self, run_start, run_end, neighbor_cutoff,
                             cluster_center="center"):
        """ Calculates the average distance of the center of clusters/molecules

        Args:
            run_start (int): Start time step.
            run_end (int): End time step.
            neighbor_cutoff (int of float): Upper limit of
                first nearest neighbor.
            cluster_center (str): species name of cluster center.
                Default to "center".
        """
        center_atoms = \
            self.wrapped_run.select_atoms(self.select_dict[cluster_center])
        trj = self.wrapped_run.trajectory[run_start:run_end:]
        means = []
        for ts in trj:
            distance_matrix = capped_distance(center_atoms.positions,
                                              center_atoms.positions,
                                              max_cutoff=neighbor_cutoff,
                                              box=ts.dimensions,
                                              return_distances=True)[1]
            distance_matrix[distance_matrix == 0] = np.nan
            means.append(np.nanmean(distance_matrix))
        return np.mean(means)
