# Copyright (c) Tingzheng Hou.
# Distributed under the terms of the MIT License.

"""
A class for retrieving water and ion force field parameters.
"""

from __future__ import annotations
from typing import Literal, Final

import os
import re
from dataclasses import dataclass

from monty.json import MSONable
from monty.serialization import loadfn
from pymatgen.core import Lattice, Structure
from pymatgen.core.ion import Ion
from pymatgen.io.lammps.data import ForceField, LammpsData, Topology, lattice_2_lmpbox


MODULE_DIR: Final[str] = os.path.dirname(os.path.abspath(__file__))
DATA_DIR: Final[str] = os.path.join(MODULE_DIR, "data")
DATA_MODELS: Final[dict] = {
    "water": {
        "spc": "water_spc.lmp",
        "spce": "water_spce.lmp",
        "tip3pew": "water_tip3p_ew.lmp",
        "tip3pfb": "water_tip3p_fb.lmp",
        "opc3": "water_opc3.lmp",
        "tip4p2005": "water_tip4p_2005.lmp",
        "tip4pew": "water_tip4p_ew.lmp",
        "tip4pfb": "water_tip4p_fb.lmp",
        "opc": "water_opc.lmp",
    },
}
WATER_SIGMA: Final[dict] = {
    "spc": 3.16557,
    "spce": 3.16557,
    "tip3p": 3.1507,
    "tip3pew": 3.188,
    "tip3pfb": 3.178,
    "opc": 3.16655,
    "opc3": 3.17427,
    "tip4p": 3.1536,
    "tip4p2005": 3.1589,
    "tip4pew": 3.16435,
    "tip4pfb": 3.1655,
}


@dataclass
class IonLJData(MSONable):
    """
    A lightweight dataclass for storing ion force field parameters. The data
    file ion_lj_params.json distributed with mdgo is a serialized list of these
    objects.

    Attributes:
        name: The name of the parameter set
        formula: formula of the ion, e.g. 'Li+'
        combining_rule: the method used to compute pairwise interaction parameters
            from single atom parameters. 'geometric' or 'LB' for Lorentz-Berthelot
        water_model: The water model for which the ion parameters were optimized.
        sigma: The Lennard Jones sigma value, in Å
        epsilon: The Lennard Jones epsilon value, in kcal/mol
    """

    name: Literal["jensen_jorgensen", "joung_cheatham", "li_merz"]
    formula: str
    combining_rule: Literal["geometric", "LB"]
    water_model: Literal[
        "spc",
        "spce",
        "tip3p",
        "tip3pew",
        "tip3pfb",
        "opc3",
        "tip4p2005",
        "tip4p",
        "tip4pew",
        "tip4pfb",
        "opc",
    ]
    sigma: float
    epsilon: float


class Aqueous:
    """
    A class for retrieving water and ion force field parameters.

    Available water models are:
        1. SPC
        2. SPC/E
        3. TIP3P-EW
        4. TIP3P-FB
        5. OPC3
        6. TIP4P-EW
        7. TIP4P-2005
        8. TIP4P-FB
        9. OPC

    Multiple sets of Lennard Jones parameters for ions are available as well.
    Not every set is available for every water model. The parameter sets included
    are:
        1. Jensen and Jorgensen, 2006 (abbreviation 'jj')
        2. Joung and Cheatham, 2008 (abbreviation 'jc')
        3. Li and Merz group, 2020 (abbreviation, 'lm')

    Examples:
        Retreive SPC/E water model:
        >>> spce_data = Aqueous.get_water()
        Retreive Li+ ion by Jensen and Jorgensen:
        >>> li_data = Aqueous.get_ion(model="jj", ion="li+")
        Retreive a customized water data file:
        >>> spce_data = Aqueous.get_ion(file_name="path/to/data/file")
    """

    @staticmethod
    def get_water(model: str = "spce") -> LammpsData:
        """
        Retrieve water model parameters.

        Args:
            model: Water model to use. Valid choices are "spc", "spce", "opc3",
                "tip3pew", "tip3pfb", "tip4p2005", "tip4pew", "tip4pfb", and "opc".
                (Default: "spce")
        Returns:
            LammpsData: Force field parameters for the chosen water model.
                If you specify an invalid water model, None is returned.
        """
        signature = "".join(re.split(r"[\W|_]+", model)).lower()
        if DATA_MODELS["water"].get(signature):
            return LammpsData.from_file(os.path.join(DATA_DIR, "water", DATA_MODELS["water"].get(signature)))
        raise ValueError("Water model not found. Please specify a customized data path or try another water model.\n")

    @staticmethod
    def get_ion(
        ion: Ion | str,
        parameter_set: str = "auto",
        water_model: str = "auto",
        mixing_rule: str | None = None,
    ) -> LammpsData:
        """
        Retrieve force field parameters for an ion in water.

        Args:
            ion: Formula of the ion (e.g., "Li+"). Not case sensitive. May be
                passed as either a string or an Ion object.
            parameter_set: Force field parameters to use for ions.
                Valid choices are:
                    1. "jj" for the Jensen and Jorgensen parameters (2006)"
                    2. "jc" for Joung-Cheatham parameters (2008)
                    3. "lm" for the Li and Merz group parameters (2020-2021)"
                The default parameter set is "auto", which assigns a recommended
                parameter set that is compatible with the chosen water model.
            water_model: Water model to use. Models must be given as a string
                (not case sensitive). "-" and "/" are ignored. Hence "tip3pfb"
                and "TIP3P-FB" are both valid inputs for the TIP3P-FB water model.
                Available water models are:
                    1. SPC
                    2. SPC/E
                    3. TIP3P-EW
                    4. TIP3P-FB
                    5. OPC3
                    6. TIP4P-EW
                    7. TIP4P-2005
                    8. TIP4P-FB
                    9. OPC
                The default water model is "auto", which assigns a recommended
                water model that is compatible with the chosen ion parameters. Other
                combinations are possible at your own risk. See documentation.

            When both the parameter_set and water_model are set to "auto", the function returns the
            Joung-Cheatham parameters for the SPC/E water model.

                For a systematic comparison of the performance of different water models, refer to

                    Sachini et al., Systematic Comparison of the Structural and Dynamic Properties of
                    Commonly Used Water Models for Molecular Dynamics Simulations. J. Chem. Inf. Model.
                    2021, 61, 9, 4521–4536. https://doi.org/10.1021/acs.jcim.1c00794

            mixing_rule: The mixing rule to use for the ion parameter. Default to None, which does not
                change the original mixing rule of the parameter set. Available choices are 'LB'
                (Lorentz-Berthelot or arithmetic) and 'geometric'. If the specified mixing rule does not
                match the default mixing rule of the parameter set, the output parameter will be converted
                accordingly.


        Returns:
            Force field parameters for the chosen water model.
        """
        alias = {"aq": "aqvist", "jj": "jensen_jorgensen", "jc": "joung_cheatham", "lm": "li_merz"}
        default_sets = {
            "spc": "N/A",
            "spce": "jc",
            "tip3p": "jc",
            "tip3pew": "N/A",
            "tip3pfb": "lm",
            "opc3": "lm",
            "tip4p2005": "N/A",
            "tip4p": "jj",
            "tip4pew": "jc",
            "tip4pfb": "lm",
            "opc": "lm",
            "jj": "tip4p",
            "jc": "spce",
            "lm": "tip4pfb",
        }
        water_model = water_model.replace("-", "").replace("/", "").lower()
        parameter_set = parameter_set.lower()

        if water_model == "auto" and parameter_set == "auto":
            water_model = "spce"
            parameter_set = "jc"
        elif parameter_set == "auto":
            parameter_set = default_sets.get(water_model, parameter_set)
            if parameter_set == "N/A":
                raise ValueError(
                    f"The {water_model} water model has no specifically parameterized ion parameter sets"
                    "Please try a different water model."
                )
        elif water_model == "auto":
            water_model = default_sets.get(parameter_set, water_model)

        parameter_set = alias.get(parameter_set, parameter_set)

        # Make the Ion object to get mass and charge
        if isinstance(ion, Ion):
            ion_obj = ion
        else:
            ion_obj = Ion.from_formula(ion.capitalize())

        # load ion data as a list of IonLJData objects
        ion_data = loadfn(os.path.join(DATA_DIR, "ion_lj_params.json"))

        # make sure the ion is in the DataFrame
        key = ion_obj.reduced_formula
        filtered_data = [d for d in ion_data if d.formula == key]
        if len(filtered_data) == 0:
            raise ValueError(f"Ion {key} not found in database. Please try a different ion.")

        # make sure the parameter set is in the DataFrame
        filtered_data = [d for d in filtered_data if d.name == parameter_set and d.water_model == water_model]
        if len(filtered_data) == 0:
            raise ValueError(
                f"No {parameter_set} parameters for water model {water_model} for ion {key}. "
                "See documentation and try a different combination."
            )

        if len(filtered_data) != 1:
            raise ValueError(
                f"Something is wrong: multiple ion data entries for {key}, {parameter_set}, and {water_model}"
            )

        # we only consider monatomic ions at present
        # construct a cubic LammpsBox from a lattice
        lat = Lattice([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        box = lattice_2_lmpbox(lat)[0]
        # put it in the center of a cubic Structure
        struct = Structure(lat, ion_obj, [[0.5, 0.5, 0.5]])
        # construct Topology with the ion centered in the box
        topo = Topology(struct, charges=[ion_obj.charge])

        # retrieve Lennard-Jones parameters
        # construct ForceField object
        sigma = filtered_data[0].sigma
        epsilon = filtered_data[0].epsilon
        if mixing_rule is None:
            pass
        else:
            default_mixing = filtered_data[0].combining_rule
            water_sigma = WATER_SIGMA.get(filtered_data[0].water_model)
            if mixing_rule.lower() in ["lb", "arithmetic", "lorentz-berthelot", "lorentz berthelot"]:
                mixing_rule = "LB"
            elif mixing_rule.lower() == "geometric":
                mixing_rule = "geometric"
            else:
                raise ValueError("Invalid mixing rule. Supported mixing rules are 'LB'(arithmetic) and 'geometric'. ")
            if default_mixing == mixing_rule:
                pass
            elif default_mixing == "LB" and mixing_rule == "geometric":
                sigma = ((water_sigma + sigma) / 2) ** 2 / water_sigma
                print(
                    "The parameter mixing rule has been converted from the original 'LB' to 'geometric'.\n"
                    "Please use the parameter set with caution!"
                )
            else:
                sigma = 2 * ((water_sigma * sigma) ** (1 / 2)) - water_sigma
                print(
                    "The parameter mixing rule has been converted from the original 'geometric' to 'LB'.\n"
                    "Please use the parameter set with caution!"
                )
        ff = ForceField([(str(e), e) for e in ion_obj.elements], nonbond_coeffs=[[epsilon, sigma]])

        return LammpsData.from_ff_and_topologies(box, ff, [topo], atom_style="full")
