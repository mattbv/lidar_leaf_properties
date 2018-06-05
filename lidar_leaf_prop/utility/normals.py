# -*- coding: utf-8 -*-
"""
Module to calculate normal vectors.

@author: Matheus Boni Vicari (matheus.boni.vicari@gmail.com)
"""

import numpy as np
from numba import jit
from nnsearch import set_nbrs_knn


def point_normals(arr, knn):

    # Obtaining neighborhood parameters.
    dist, indices = set_nbrs_knn(arr, arr, knn, return_dist=True)
    indices = indices.astype(int)

    # Initializing normals, centers and eigenvalues arrays.
    nn = np.full([arr.shape[0], 3], np.nan)
    cc = np.full([arr.shape[0], 3], np.nan)
    ss = np.full([arr.shape[0], 3], np.nan)

    # Looping over each neighborhood set and calculating normal parameters.
    for i, (idx, d) in enumerate(zip(indices, dist)):
        # Calculates and stores neihborhood centroids, normal vectors and
        # eigenvalues.
        C, N, S = normals_from_cloud(arr[idx])
        nn[i] = N
        cc[i] = C
        ss[i] = S

    norm_vec = nn.copy()
    mask = norm_vec[:, 2] < 0
    norm_vec[mask] = norm_vec[mask] * -1

    return norm_vec, cc, ss


@jit
def normals_from_triangle(tri):

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
def normals_from_cloud(arr):

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
