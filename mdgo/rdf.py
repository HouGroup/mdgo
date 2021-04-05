import numpy as np
from MDAnalysis.analysis import rdf


class RdfMemoizer:

    def __init__(self, universe):
        """
        This class is a intended to make generating and storing RDF easy and fast for the
        MdRun core class.

        Args:
            universe:
        """
        self.u = universe
        self.rdfs = {}


    def _get_rdf(self, central_atom_type, neighbor_atom_type, time_step,
                 rdf_range=[1, 10], fresh_rdf=False):
        """
        A function to return rdfs for a given pair of atoms at a given time step.
        Memoizes generated rdfs to prevent unnecessary computation.
        Args:
            central_atom_type:
            neighbor_atom_type:
            time_step:
            rdf_range:
            fresh_rdf: set to True to deactivate memoization

        Returns:

        """
        rdf_key = (central_atom_type, neighbor_atom_type, time_step, rdf_range)
        if not fresh_rdf and (rdf_key in self.rdfs.keys()):
            return self.rdfs[rdf_key]
        else:
            central_atoms = self.u.select_atoms(f'type {central_atom_type}')
            neighbor_atoms = self.u.select_atoms(f'type {neighbor_atom_type}')
            self.u.trajectory[time_step]
            local_rdf = rdf.InterRDF(central_atoms, neighbor_atoms, range=rdf_range)
            local_rdf.run()
            self.rdfs[rdf_key] = local_rdf
            return self.rdfs[rdf_key]

    def rdf_data(self, central_atom_type, neighbor_atom_type, time_step,
                 rdf_range=[1, 10], fresh_rdf=False):
        """
        This yields the y-axis data of an rdf plot.
        Args:
            central_atom_type:
            neighbor_atom_type:
            time_step:
            rdf_range:
            fresh_rdf:

        Returns:

        """
        local_rdf = self._get_rdf(central_atom_type, neighbor_atom_type, time_step,
                                  tuple(rdf_range), fresh_rdf)
        rdf_values = local_rdf.rdf
        bin_values = local_rdf.bins
        return rdf_values, bin_values

    def rdf_integral_data(self, central_atom_type, neighbor_atom_type, time_step,
                          rdf_range=[1, 10], fresh_rdf=False):
        local_rdf = self._get_rdf(central_atom_type, neighbor_atom_type, time_step,
                                  tuple(rdf_range), fresh_rdf)
        rdf_values = local_rdf.rdf
        bin_values = local_rdf.bins
        integral_values = rdf_values.cumsum() / rdf_values.sum()
        return integral_values, bin_values

    def rdf_bins(self, central_atom_type, neighbor_atom_type, time_step,
                 rdf_range=[1, 10], fresh_rdf=False):
        """
        This yields the x-axis data of an rdf plot.

        Args:
            central_atom_type:
            neighbor_atom_type:
            time_step:
            rdf_range:
            fresh_rdf:

        Returns:

        """
        local_rdf = self._get_rdf(central_atom_type, neighbor_atom_type, time_step,
                                  rdf_range, fresh_rdf)
        return local_rdf.bins
