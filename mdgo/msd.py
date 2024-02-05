# Copyright (c) Tingzheng Hou.
# Distributed under the terms of the MIT License.

"""
This module implements functions for calculating meen square displacement (MSD).


MSD FFT Algorithms in this section are adapted from DOI: 10.1051/sfn/201112010 and
http://stackoverflow.com/questions/34222272/computing-mean-square-displacement-using-python-and-fft#34222273
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

try:
    import MDAnalysis.analysis.msd as mda_msd
except ImportError:
    mda_msd = None
try:
    import tidynamics as td
except ImportError:
    td = None

import numpy as np
from tqdm.auto import trange

if TYPE_CHECKING:
    from MDAnalysis import AtomGroup, Universe
    from MDAnalysis.core.groups import Atom

__author__ = "Tingzheng Hou"
__version__ = "0.3.0"
__maintainer__ = "Tingzheng Hou"
__email__ = "tingzheng_hou@berkeley.edu"
__date__ = "Jul 19, 2021"


DIM = Literal["xyz", "xy", "yz", "xz", "x", "y", "z"]


def total_msd(
    nvt_run: Universe,
    start: int,
    end: int,
    select: str = "all",
    msd_type: DIM = "xyz",
    fft: bool = True,
    built_in: bool = True,
    center_of_mass: bool = True,
) -> np.ndarray:
    """
    From an MD Universe, calculates the MSD array of a group of atoms defined by select.

    Args:
        nvt_run: An MDAnalysis ``Universe`` containing unwrapped trajectory.
        start: Start frame of analysis.
        end: End frame of analysis.
        select: A selection string. Defaults to “all” in which case all atoms are selected.
        msd_type: Desired dimensions to be included in the MSD. Defaults to "xyz".
        fft: Whether to use FFT to accelerate the calculation. Default to True.
        built_in: Whether to use built in method to calculate msd or use function from mds. Default to True.
        center_of_mass: Whether to subtract center of mass at each step for atom coordinates. Default to True.

    Warning:
        To correctly compute the MSD using this analysis module, you must supply coordinates in the
        unwrapped convention. That is, when atoms pass the periodic boundary, they must not be
        wrapped back into the primary simulation cell.

    Returns:
        An array of calculated MSD.
    """
    if built_in:
        return onsager_ii_self(
            nvt_run, start, end, select=select, msd_type=msd_type, center_of_mass=center_of_mass, fft=fft
        )

    if not mda_msd:
        raise ValueError("MDAnalysis version too low, please update MDAnalysis.")
    if center_of_mass:
        raise ValueError(
            "Warning! MDAnalysis does not support subtracting center of mass. Calculating without subtracting..."
        )
    if fft and td is None:
        raise ImportError(
            """tidynamics was not found!

                        tidynamics is required to compute an FFT based MSD (default)

                        try installing it using pip eg:

                            pip install tidynamics

                        or set fft=False"""
        )
    return mda_msd_wrapper(nvt_run, start, end, select=select, msd_type=msd_type, fft=fft)


def autocorr_fft(x: np.ndarray) -> np.ndarray:
    """
    Calculates the autocorrelation function using the fast Fourier transform.

    Args:
        x: function on which to compute autocorrelation function

    Returns a numpy.array of the autocorrelation function
    """
    N = len(x)
    F = np.fft.fft(x, n=2 * N)  # 2*N because of zero-padding
    PSD = F * F.conjugate()
    res = np.fft.ifft(PSD)
    res = (res[:N]).real
    n = N * np.ones(N) - np.arange(0, N)
    return res / n


def msd_fft(r: np.ndarray) -> np.ndarray:
    """
    Calculates mean square displacement of the array r using the fast Fourier transform.

    Args:
        r: atom positions over time

    Returns a numpy.array containing the mean-squared displacement over time
    """
    N = len(r)
    D = np.square(r).sum(axis=1)
    D = np.append(D, 0)
    S2 = sum([autocorr_fft(r[:, i]) for i in range(r.shape[1])])
    Q = 2 * D.sum()
    S1 = np.zeros(N)
    for m in range(N):
        Q = Q - D[m - 1] - D[N - m]
        S1[m] = Q / (N - m)
    return S1 - 2 * S2


def msd_straight_forward(r: np.ndarray) -> np.ndarray:
    """
    Calculates mean square displacement of the array r using straight forward method.

    Args:
        r: atom positions over time

    Returns a numpy.array containing the mean-squared displacement over time
    """
    shifts = np.arange(len(r))
    msds = np.zeros(shifts.size)

    for i, shift in enumerate(shifts):
        diffs = r[: -shift if shift else None] - r[shift:]
        sqdist = np.square(diffs).sum(axis=1)
        msds[i] = sqdist.mean()

    return msds


def create_position_arrays(
    nvt_run: Universe, start: int, end: int, select: str = "all", center_of_mass: bool = True
) -> np.ndarray:
    """
    Creates an array containing the positions of all cations and anions over time.

    nvt_run: An MDAnalysis ``Universe`` containing unwrapped trajectory.
    start: Start frame of analysis.
    end: End frame of analysis.
    select: A selection string. Defaults to “all” in which case all atoms are selected.
    center_of_mass: Whether to subtract center of mass at each step for atom coordinates. Default to True.
    """
    time = 0
    atom_group = nvt_run.select_atoms(select)
    atom_positions = np.zeros((end - start, len(atom_group), 3))
    if center_of_mass:
        for _ts in nvt_run.trajectory[start:end]:
            atom_positions[time, :, :] = atom_group.positions - nvt_run.atoms.center_of_mass()
            time += 1
    else:
        for _ts in nvt_run.trajectory[start:end]:
            atom_positions[time, :, :] = atom_group.positions
            time += 1
    return atom_positions


def onsager_ii_self(
    nvt_run: Universe,
    start: int,
    end: int,
    select: str = "all",
    msd_type: DIM = "xyz",
    center_of_mass: bool = True,
    fft: bool = True,
) -> np.ndarray:
    """
    From a MD Universe, calculates the MSD array for the self component for
    a diagonal transport coefficient (L^{ii}).

    Args:
        nvt_run: An MDAnalysis ``Universe`` containing unwrapped trajectory.
        start: Start frame of analysis.
        end: End frame of analysis.
        select: A selection string. Defaults to “all” in which case all atoms are selected.
        msd_type: Desired dimensions to be included in the MSD. Defaults to "xyz".
        center_of_mass: Whether to subtract center of mass at each step for atom coordinates. Default to True.
        fft: Whether to use FFT to accelerate the calculation. Default to True.

    Warning:
        To correctly compute the MSD using this analysis module, you must supply coordinates in the
        unwrapped convention. That is, when atoms pass the periodic boundary, they must not be
        wrapped back into the primary simulation cell.

    Returns:
        An array of "MSD" corresponding to the L^{ii}_{self} transport coefficient at each time
    """
    atom_positions = create_position_arrays(nvt_run, start, end, select=select, center_of_mass=center_of_mass)
    ii_self = np.zeros(end - start)
    n_atoms = np.shape(atom_positions)[1]
    dim = parse_msd_type(msd_type)
    if fft:
        for atom_num in range(n_atoms):
            r = atom_positions[:, atom_num, dim[0] : dim[1] : dim[2]]
            msd_temp = msd_fft(np.array(r))  # [start:end] bug fix, please confirm
            ii_self += msd_temp
    else:
        for atom_num in range(n_atoms):
            r = atom_positions[:, atom_num, dim[0] : dim[1] : dim[2]]
            msd_temp = msd_straight_forward(np.array(r))  # [start:end]
            ii_self += msd_temp
    return np.array(ii_self) / n_atoms


def mda_msd_wrapper(
    nvt_run: Universe, start: int, end: int, select: str = "all", msd_type: DIM = "xyz", fft: bool = True
) -> np.ndarray:
    """
    From a MD Universe, calculates the MSD array of a group of atoms using MDAnalysis's msd module..

    Args:
        nvt_run: An MDAnalysis ``Universe`` containing unwrapped trajectory.
        start: Start frame of analysis.
        end: End frame of analysis.
        select: A selection string. Defaults to “all” in which case all atoms are selected.
        msd_type: Desired dimensions to be included in the MSD. Defaults to "xyz".
        fft: Whether to use FFT to accelerate the calculation. Default to True.

    Warning:
        To correctly compute the MSD using this analysis module, you must supply coordinates in the
        unwrapped convention. That is, when atoms pass the periodic boundary, they must not be
        wrapped back into the primary simulation cell.

    Returns:
        An array of calculated MSD.
    """
    msd_calculator = mda_msd.EinsteinMSD(nvt_run, select=select, msd_type=msd_type, fft=fft)
    msd_calculator.run(start=start, stop=end)
    try:
        total_array = msd_calculator.results.timeseries
    except AttributeError:
        total_array = msd_calculator.timeseries

    return total_array


def parse_msd_type(msd_type: DIM) -> list[int]:
    """
    Sets up the desired dimensionality of the MSD.

    msd_type: Desired dimensions to be included in the MSD.

    Returns:
        A list of indexing number for slicing x, y, z coordinates.
    """
    keys = {
        "x": [0, 1, 1],
        "y": [1, 2, 1],
        "z": [2, 3, 1],
        "xy": [0, 2, 1],
        "xz": [0, 3, 2],
        "yz": [1, 3, 1],
        "xyz": [0, 3, 1],
    }

    msd_type_str = str(msd_type).lower()

    try:
        dim = keys[msd_type_str]
    except KeyError:
        raise ValueError(f"invalid msd_type: {msd_type_str} specified, please specify one of xyz, xy, xz, yz, x, y, z")
    return dim


def _total_msd(nvt_run: Universe, run_start: int, run_end: int, select: str = "all") -> np.ndarray:
    """
    A native MSD calculator. Uses the conventional algorithm.

    Args:
        nvt_run: An MDAnalysis ``Universe`` containing unwrapped trajectory.
        select: A selection string. Defaults to “all” in which case all atoms are selected.
        run_start: Start frame of analysis.
        run_end: End frame of analysis.

    Returns:
        An array of calculated MSD.
    """
    trj_analysis = nvt_run.trajectory[run_start:run_end:]
    li_atoms = nvt_run.select_atoms(select)
    all_list = []
    for li_atom in li_atoms:
        coords = []
        for ts in trj_analysis:
            current_coord = ts[li_atom.id - 1]
            coords.append(current_coord)
        all_list.append(np.array(coords))
    return msd_from_frags(all_list, run_end - run_start - 1)


def msd_from_frags(coord_list: list[np.ndarray], largest: int) -> np.ndarray:
    """
    Calculates the MSD using a list of fragments of trajectory with the conventional algorithm.

    Args:
        coord_list: A list of trajectory.
        largest: The largest interval of time frame for calculating MSD.

    Returns:
        The MSD series.
    """
    msd_dict: dict[int | np.integer, np.ndarray] = {}
    for state in coord_list:
        n_frames = state.shape[0]
        lag_times = np.arange(1, min(n_frames, largest))
        for lag in lag_times:
            disp = state[:-lag, :] - state[lag:, :]
            sqdist = np.square(disp).sum(axis=-1)
            if lag in msd_dict:
                msd_dict[lag] = np.concatenate((msd_dict[lag], sqdist), axis=0)
            else:
                msd_dict[lag] = sqdist
    timeseries = []
    time_range = len(msd_dict) + 1
    msds_by_state = np.zeros(time_range)
    for kw in range(1, time_range):
        msds = msd_dict.get(kw)
        assert msds is not None
        msds_by_state[kw] = msds.mean()
        timeseries.append(msds_by_state[kw])
    return np.array(timeseries)


def states_coord_array(
    nvt_run: Universe,
    atom: Atom,
    select_dict: dict[str, str],
    distance: float,
    run_start: int,
    run_end: int,
    binding_site: str = "anion",
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Cuts the trajectory of an atom into fragments. Each fragment contains consecutive timesteps of coordinates
    of the atom in either attached or free state. The Attached state is when the atom coordinates with the
    ``binding_site`` species (distance < ``distance``), and vice versa for the free state.
    TODO: check if need wrapped trj.

    Args:
        nvt_run: An MDAnalysis ``Universe`` containing unwrapped trajectory.
        atom: The Atom object to analyze.
        select_dict: A dictionary of atom species selection, where each atom species name is a key
            and the corresponding values are the selection language.
        distance: The coordination cutoff distance.
        run_start: Start frame of analysis.
        run_end: End frame of analysis.
        binding_site: The species the ``atom`` coordinates to.

    Returns:
        Two list of coordinates arrays containing where each coordinates array is a consecutive trajectory fragment
        of atom in a certain state. One for the attached state, the other for the free state.
    """
    trj_analysis = nvt_run.trajectory[run_start:run_end:]
    attach_list = []
    free_list = []
    coords = []
    prev_state = None
    prev_coord = None
    for ts in trj_analysis:
        selection = (
            "(" + select_dict[binding_site] + ") and (around " + str(distance) + " index " + str(atom.id - 1) + ")"
        )
        shell = nvt_run.select_atoms(selection, periodic=True)
        current_state = 0
        if len(shell) > 0:
            current_state = 1
        current_coord = ts[atom.id - 1]

        if prev_state:
            if current_state == prev_state:
                coords.append(current_coord)
                prev_coord = current_coord
            else:
                if len(coords) > 1:
                    if prev_state:
                        attach_list.append(np.array(coords))
                    else:
                        free_list.append(np.array(coords))
                prev_state = current_state
                coords = []
                coords.append(prev_coord)
                coords.append(current_coord)
                prev_coord = current_coord
        else:
            coords.append(current_coord)
            prev_state = current_state
            prev_coord = current_coord

    if len(coords) > 1:
        if prev_state:
            attach_list.append(np.array(coords))
        else:
            free_list.append(np.array(coords))

    return attach_list, free_list


def partial_msd(
    nvt_run: Universe,
    atoms: AtomGroup,
    largest: int,
    select_dict: dict[str, str],
    distance: float,
    run_start: int,
    run_end: int,
    binding_site: str = "anion",
) -> tuple[list[np.ndarray] | None, list[np.ndarray] | None]:
    """Calculates the mean square displacement (MSD) of the ``atoms`` according to coordination states.
    The returned ``free_data`` include the MSD when ``atoms`` are not coordinated to ``binding_site``.
    The ``attach_data`` includes the MSD of ``atoms`` are not coordinated to ``binding_site``.

    Args:
        nvt_run: An MDAnalysis ``Universe`` containing unwrapped trajectory.
        atoms: The AtomGroup for
        largest: The largest interval of time frame for calculating MSD.
        select_dict: A dictionary of atom species selection, where each atom species name is a key
            and the corresponding values are the selection language.
        distance: The coordination cutoff distance between
        run_start: Start frame of analysis.
        run_end: End frame of analysis.
        binding_site: The species the ``atoms`` coordinates to.

    Returns:
        Two arrays of MSD in the trajectory
    """
    free_coords = []
    attach_coords = []
    for i in trange(len(atoms)):
        attach_coord, free_coord = states_coord_array(
            nvt_run, atoms[i], select_dict, distance, run_start, run_end, binding_site=binding_site
        )
        attach_coords.extend(attach_coord)
        free_coords.extend(free_coord)
    attach_data = None
    free_data = None
    if len(attach_coords) > 0:
        attach_data = msd_from_frags(attach_coords, largest)
    if len(free_coords) > 0:
        free_data = msd_from_frags(free_coords, largest)
    return free_data, attach_data
