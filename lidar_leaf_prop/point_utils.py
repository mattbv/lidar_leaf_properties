# -*- coding: utf-8 -*-
"""
@author: Matheus Boni Vicari (matheus.boni.vicari@gmail.com)
"""

import numpy as np
import pandas as pd
from nnsearch import set_nbrs_knn


def get_diff(arr1, arr2):

    """
    Function to perform the intersection of two arrays, returning the
    entries not intersected between arr1 and arr2.

    Parameters
    ----------
    arr1: array
        N-dimensional array of points to intersect.
    arr2: array
        N-dimensional array of points to intersect.

    Returns
    -------
    arr: array
        Difference array between 'arr1' and 'arr2'.

    """

    # Asserting that both arrays have the same number of columns.
    assert arr1.shape[1] == arr2.shape[1]

    # Stacking both arrays.
    arr3 = np.vstack((arr1, arr2))

    # Creating a pandas.DataFrame from the stacked array.
    df = pd.DataFrame(arr3)

    # Removing duplicate points and keeping only points that have only a
    # single occurrence in the stacked array.
    diff = df.drop_duplicates(keep=False)

    return np.asarray(diff)


def remove_duplicates(arr, return_ids=False):

    """
    Function to remove duplicated rows from an array.

    Parameters
    ----------
    arr: array
        N-dimensional array (m x n) containing a set of parameters (n) over a
        set of observations (m).

    Returns
    -------
    unique: array
        N-dimensional array (m* x n) containing a set of unique parameters (n)
        over a set of unique observations (m*).

    """

    # Setting the pandas.DataFrame from the array (arr) data.
    df = pd.DataFrame({'x': arr[:, 0],
                       'y': arr[:, 1], 'z': arr[:, 2]})

    # Using the drop_duplicates function to remove the duplicate points from
    # df.
    unique = df.drop_duplicates(['x', 'y', 'z'])

    return np.asarray(unique).astype(float)


def get_center(arr):

    """
    Function to calculate the centroid coordinates of a set of points.

    Parameters
    ==========
    arr: numpy.ndarray
        n x dimensions set of point coordinates.

    Returns
    =======
    arr: numpy.ndarray
        1 x d centroid coordinates.

    """

    return np.min(arr, axis=0) + ((np.max(arr, axis=0) -
                                  np.min(arr, axis=0)) / 2)


def upscale_data(low_dens, high_dens, dist_threshold=0.02):

    """
    Function to obtain a higher density set of points based on their distance
    from points contained in a lower density cloud.

    Parameters
    ==========
    low_dens: numpy.ndarray
        n x 3 array contained a low(er) density point cloud.
    high_dens: numpy.ndarray
        n x 3 array contained a high(er) density point cloud.
    dist_threshold: float
        Maximum distance of points in high_dens to points in low_dens that are
        still considered a valid neighborhood.

    """

    # Obtaining neighborhoord parameters.
    dist, idx = set_nbrs_knn(low_dens[:, :3], high_dens[:, :3], 1)
    # Masking neighborhoods based on distance threshold.
    mask = np.where(dist <= dist_threshold)[0]

    return high_dens[mask]
