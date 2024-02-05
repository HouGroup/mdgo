# Copyright (c) Tingzheng Hou.
# Distributed under the terms of the MIT License.

"""A class for writing, overwriting, scaling charges of a LammpsData object."""

from __future__ import annotations

import numpy as np
from pymatgen.io.lammps.data import LammpsData


class ChargeWriter:
    """
    A class for write, overwrite, scale charges of a LammpsData object.
    TODO: Auto determine number of significant figures of charges
    TODO: write to obj or write separate charge file
    TODO: Read LammpsData or path.

    Args:
        data: The provided LammpsData obj.
        precision: Number of significant figures.
    """

    def __init__(self, data: LammpsData, precision: int = 10):
        """Base constructor."""
        self.data = data
        self.precision = precision

    def scale(self, factor: float) -> LammpsData:
        """
        Scales the charge in of the in self.data and returns a new one. TODO: check if non-destructive.

        Args:
            factor: The charge scaling factor

        Returns:
            A recreated LammpsData obj
        """
        items = {}
        items["box"] = self.data.box
        items["masses"] = self.data.masses
        atoms = self.data.atoms.copy(deep=True)
        atoms["q"] = atoms["q"] * factor
        assert np.around(atoms.q.sum(), decimals=self.precision) == np.around(
            self.data.atoms.q.sum() * factor, decimals=self.precision
        )
        digit_count = 0
        for q in atoms["q"]:
            rounded = self.count_significant_figures(q)
            if rounded > digit_count:
                digit_count = rounded
        print("No. of significant figures to output for charges: ", digit_count)
        items["atoms"] = atoms
        items["atom_style"] = self.data.atom_style
        items["velocities"] = self.data.velocities
        items["force_field"] = self.data.force_field
        items["topology"] = self.data.topology
        return LammpsData(**items)

    def count_significant_figures(self, number: float) -> int:
        """
        Count significant figures in a float.

        Args:
            number: The number to count.

        Returns:
            The number of significant figures.
        """
        number_str = repr(float(number))
        tokens = number_str.split(".")
        if len(tokens) > 2:
            raise ValueError(f"Invalid number '{number}' only 1 decimal allowed")
        if len(tokens) == 2:
            decimal_num = tokens[1][: self.precision].rstrip("0")
            return len(decimal_num)
        return 0
