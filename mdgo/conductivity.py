# coding: utf-8
# Copyright (c) Tingzheng Hou.
# Distributed under the terms of the MIT License.

import numpy as np
from tqdm.notebook import tqdm
from scipy import stats

__author__ = "Kara Fong, Tingzheng Hou"
__version__ = "1.0"
__maintainer__ = "Tingzheng Hou"
__email__ = "tingzheng_hou@berkeley.edu"
__date__ = "Feb 9, 2021"

"""
Algorithms in this section are adapted from DOI: 10.1051/sfn/201112010 and
http://stackoverflow.com/questions/34222272/computing-mean-square-displacement-using-python-and-fft#34222273
"""


def autocorr_fft(x):
    """ Calculates the autocorrelation function using the fast Fourier transform.

    Args:
        x (numpy.array): function on which to compute autocorrelation function

    Returns a numpy.array of the autocorrelation function
    """
    N = len(x)
    F = np.fft.fft(x, n=2 * N)  # 2*N because of zero-padding
    PSD = F * F.conjugate()
    res = np.fft.ifft(PSD)
    res = (res[:N]).real
    n = N * np.ones(N) - np.arange(0, N)
    return res / n


def msd_fft(r):
    """ Calculates mean square displacement of the array r using the fast Fourier transform.

    Args:
        r (numpy.array): atom positions over time

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


def calc_cond(u, anions, cations, run_start, cation_charge=1, anion_charge=-1):
    """Calculates the conductivity "mean square displacement" over time

    Note:
       Coordinates must be unwrapped (in dcd file when creating MDAnalysis Universe)

    Args:
        u: MDAnalysis universe
        anions: MDAnalysis AtomGroup containing all anions (assumes anions are single atoms)
        cations: MDAnalysis AtomGroup containing all cations (assumes cations are single atoms)
        run_start (int): index of trajectory from which to start analysis
        cation_charge (int): net charge of cation
        anion_charge (int): net charge of anion

    Returns a numpy.array containing conductivity "MSD" over time
    """
    # Current code assumes anion and cation selections are single atoms
    qr = []
    for ts in tqdm(u.trajectory[run_start:]):
        qr_temp = np.zeros(3)
        for anion in anions.atoms:
            qr_temp += anion.position * anion_charge
        for cation in cations.atoms:
            qr_temp += cation.position * cation_charge
        qr.append(qr_temp)
    msd = msd_fft(np.array(qr))
    return msd


def conductivity_calculator(time_array, cond_array, v, name, start, end, T=298.15, units='real'):
    """Calculates the overall conductivity of the system

    Args:
        time_array (numpy.array): times at which position data was collected in the simulation
        cond_array (numpy.array): conductivity "mean squared displacement"
        v (float): simulation volume (Angstroms^3)
        name (str): system name
        start (int): index at which to start fitting linear regime of the MSD
        end (int): index at which to end fitting linear regime of the MSD
        units (str): unit system (currently 'real' and 'lj' are supported)

    Returns the overall ionic conductivity (float)
    """
    # Unit conversions
    if units == 'real':
        A2cm = 1e-8  # Angstroms to cm
        ps2s = 1e-12  # picoseconds to seconds
        e2c = 1.60217662e-19  # elementary charge to Coulomb
        kb = 1.38064852e-23  # Boltzmann Constant, J/K
        convert = e2c * e2c / ps2s / A2cm * 1000
        cond_units = 'mS/cm'
    elif units == 'lj':
        kb = 1
        convert = 1
        cond_units = 'q^2/(tau sigma epsilon)'
    else:
        raise ValueError("units selection not supported")

    slope, _, _, _, _ = stats.linregress(time_array[start:end], cond_array[start:end])
    cond = slope / 6 / kb / T / v * convert

    print("Conductivity of " + name + ": " + str(cond) + " " + cond_units)

    return cond