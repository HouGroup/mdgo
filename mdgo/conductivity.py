# coding: utf-8
# Copyright (c) Tingzheng Hou.
# Distributed under the terms of the MIT License.

import numpy as np
from tqdm import tqdm_notebook
from scipy import stats

__author__ = "Kara Fong, Tingzheng Hou"
__version__ = "1.0"
__maintainer__ = "Tingzheng Hou"
__email__ = "tingzheng_hou@berkeley.edu"
__date__ = "Feb 9, 2021"


def autocorrFFT(x):
    """
    Calculates the position autocorrelation function using
    the fast Fourier transform.
    """
    N = len(x)
    F = np.fft.fft(x, n=2 * N)  # 2*N because of zero-padding
    PSD = F * F.conjugate()
    res = np.fft.ifft(PSD)
    res = (res[:N]).real  # now we have the autocorrelation in convention B
    n = N * np.ones(N) - np.arange(0, N)  # divide res(m) by (N-m)
    return res / n  # this is the autocorrelation in convention A


def msd_fft(r):
    """
    Calculates mean square displacement of the array r using
    the fast Fourier transform.
    """
    N = len(r)
    D = np.square(r).sum(axis=1)
    D = np.append(D, 0)
    S2 = sum([autocorrFFT(r[:, i]) for i in range(r.shape[1])])
    Q = 2 * D.sum()
    S1 = np.zeros(N)
    for m in range(N):
        Q = Q - D[m - 1] - D[N - m]
        S1[m] = Q / (N - m)
    return S1 - 2 * S2


def calc_cond(u, anions, cations, run_start, cation_charge=1, anion_charge=-1):
    """ Calculates the conductivity "mean square displacement" given
    an MDAnalysis universe (u) and a selection of atoms or molecules (sel)

    Args:
    u: MDAnalysis Universe
    sel: Selection of atoms or molecules. Should be
    a MDAnalysis AtomGroup.

    Returns an array of MSD values for each time in the trajectory.

    Note:
        Coordinates must be unwrapped (in dcd file when creating MDAnalysis
        Universe)
    """
    # Current code assumes anion and cation selections are single atoms
    qr = []
    for ts in tqdm_notebook(u.trajectory[run_start:]):
        qr_temp = np.zeros(3)
        for anion in anions.atoms:
            qr_temp += anion.position * anion_charge
        for cation in cations.atoms:
            qr_temp += cation.position * cation_charge
        qr.append(qr_temp)
    msd = (msd_fft(np.array(qr)))
    return msd


def conductivity_calculator(time_array, cond_array, v, name, start, end):
    # Unit conversions
    A2cm = 1e-8
    ps2s = 1e-12
    e2c = 1.60217662e-19
    convert = e2c*e2c/ps2s/A2cm*1000  # mS/cm
    kb = 1.38064852e-23
    T = 298.15

    # Calculate conductivity of mof
    slope_cond_avg, intercept_cond_avg, r_value, p_value, std_err \
        = stats.linregress(time_array[start:end], cond_array[start:end])
    cond_einstein_mof = slope_cond_avg/6/kb/T/v*convert
    error_mof = std_err/6/kb/T/v*convert

    print("GK Conductivity of " + name + ": " + str(cond_einstein_mof)
          + " ± " + str(error_mof) + " mS/cm")
