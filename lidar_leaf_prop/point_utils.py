# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 08:38:19 2016

@author: mathe
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


def get_base_point(arr):

    xbase = np.min(arr[:, 0]) + ((np.max(arr[:, 0]) -
                                  np.min(arr[:, 0])) / 2)
    ybase = np.min(arr[:, 1]) + ((np.max(arr[:, 1]) -
                                  np.min(arr[:, 1])) / 2)
    zbase = np.min(arr[:, 2])

    return np.array([xbase, ybase, zbase])


def get_center(arr):
    return np.min(arr, axis=0) + ((np.max(arr, axis=0) -
                                  np.min(arr, axis=0)) / 2)


def upscale_data(low_dens, high_dens, dist_threshold=0.02):

    dist, idx = set_nbrs_knn(low_dens[:, :3], high_dens[:, :3], 1)
    mask = np.where(dist <= dist_threshold)[0]

    return high_dens[mask]
