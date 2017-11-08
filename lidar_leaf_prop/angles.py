# -*- coding: utf-8 -*-
"""
@author: Matheus Boni Vicari (matheus.boni.vicari@gmail.com)
"""

import numpy as np
from numba import jit


@jit
def norm_to_angle(norm_vec):

    """
    Function to calculate the (acute) angles between an array of normal vectors
    and a purerely vertical vector (0, 0, 1).

    Parameters
    ==========
    norm_vec: numpy.ndarray
        n x 3 array of normal vectors.

    Returns
    =======
    angle: numpy.ndarray
        n x 1 array of angles from vertical vector.

    """

    # Initializing array of angles as nans.
    angles = np.full(norm_vec.shape[0], np.nan)

    # Iterating over normal vectors to calculate their angle from [0, 0, 1]
    # using the function angle. Stores results to array 'angles'.
    for i, nn in enumerate(norm_vec):
        angles[i] = angle([0., 0., 1.], nn)

    return rad_to_degree(angles)


@jit
def angle(vec1, vec2):

    """
    Function to calculate the angle between two vectors.

    Parameters
    ==========
    vec1: list or numpy.ndarray
        First vector.
    vec2: list or numpy.ndarray
        Second vector.

    Returns
    =======
    angle: float
        Angle between v1 and v2
    """

    angle = np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) *
                      np.linalg.norm(vec2)))

    return angle


@jit
def rad_to_degree(angles):

    """
    Function to convert an angle, or array of angles, from radians to degree.

    Parameters
    ==========
    angles: scalar or numpy.ndarray
        Angle or set of angles.

    Return
    ======
    degree_angles: scalar or numpy.ndarray
        Converted angle or set of angles.

    """

    return angles * (180 / np.pi)


@jit
def angle_from_zenith(angle):

    """
    Function to calculate the smallest angular distance of an angle from
    zenith.

    Parameters
    ==========
    angle: scalar
        Input angle value, in degrees.

    Returns
    =======
    angle_from_zenith: scalar
        Angular distance from zenith.

    """

    # Setting up angular zenith values.
    quad = np.array([0, 180, 360])

    # Calculates and returns shortest angular distance from zenith.
    return np.min(np.abs(angle - quad))
