# coding: utf-8
# Copyright (c) Tingzheng Hou.
# Distributed under the terms of the MIT License.

"""Utilities for manipulating coordinates under periodic boundary conditions."""

from __future__ import annotations
import numpy as np

from MDAnalysis.core.groups import Atom


def atom_vec(atom1: Atom, atom2: Atom, dimension: np.ndarray) -> np.ndarray:
    """
    Calculate the vector of the positions from atom2 to atom1.

    Args:
        atom1: Atom obj 1.
        atom2: Atom obj 2.
        dimension: box dimension.

    Return:
        The obtained vector
    """
    vec = [0, 0, 0]
    for i in range(3):
        diff = atom1.position[i] - atom2.position[i]
        if diff > dimension[i] / 2:
            vec[i] = diff - dimension[i]
        elif diff < -dimension[i] / 2:
            vec[i] = diff + dimension[i]
        else:
            vec[i] = diff
    return np.array(vec)


def position_vec(
    pos1: list[float] | np.ndarray,
    pos2: list[float] | np.ndarray,
    dimension: list[float] | np.ndarray,
) -> np.ndarray:
    """
    Calculate the vector from pos2 to pos2.

    Args:
        pos1: Array of 3d coordinates 1.
        pos2: Array of 3d coordinates 2.
        dimension: box dimension.

    Return:
        The obtained vector.
    """
    vec: list[int | float | np.floating] = [0, 0, 0]
    for i in range(3):
        diff = pos1[i] - pos2[i]
        if diff > dimension[i] / 2:
            vec[i] = diff - dimension[i]
        elif diff < -dimension[i] / 2:
            vec[i] = diff + dimension[i]
        else:
            vec[i] = diff
    return np.array(vec)


def angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.floating:
    """
    Calculate the angle between three atoms.

    Args:
        a: Coordinates of atom A.
        b: Coordinates of atom B.
        c: Coordinates of atom C.

    Returns:
        The degree A-B-C.
    """
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle_in_radian = np.arccos(cosine_angle)
    return np.degrees(angle_in_radian)
