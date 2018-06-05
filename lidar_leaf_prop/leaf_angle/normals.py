# -*- coding: utf-8 -*-
"""
Module to calculate normal vectors.

@author: Matheus Boni Vicari (matheus.boni.vicari@gmail.com)
"""

import numpy as np
from numba import jit


@jit
def from_triangle(tri):

    """
    Function to calculate the normal angle of a plane defined by 3 vertices.

    Parameters
    ==========
    tri: numpy.ndarray
        3 x 3 array containing vertices describing a plane.

    Returns
    =======
    normal_vec: numpy.ndarray
        1 x 3 array containing the normal vector of tri.

    """

    # Calculates cross product between line segments of vertices. Normalizes
    # and returns cross product.
    n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
    return n / np.linalg.norm(n)


@jit
def from_cloud(arr):

    """
    Function to calculate the normal vector of a fitted plane from a set
    of points.

    Parameters
    ==========
    arr: numpy.ndarray
        n x m coordinates of a set of points.

    Returns
    =======
    centroid: numpy.ndarray
        1 x m centroid coordinates of 'arr'.
    normal: numpy.ndarray
        1 x m normal vector of 'arr'.
    evals: numpy.ndarray
        1 x m eigenvalues of 'arr'.

    """

    # Calculating centroid coordinates of points in 'arr'.
    centroid = np.average(arr, axis=0)

    # Running SVD on centered points from 'arr'.
    _, evals, evecs = np.linalg.svd(arr - centroid)

    # Obtaining surface normal, defined as the eigenvector from the smallest
    # eigenvalue.
    normal = evecs[-1]

    return centroid, normal, evals
