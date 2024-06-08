# -*- coding: utf-8 -*-
"""
@author: Matheus Boni Vicari (matheus.boni.vicari@gmail.com)
"""

import numpy as np
from numba import jit
from scipy.spatial.distance import cdist

from leafproperties.nnsearch import set_nbrs_rad


@jit
def dist_outlier(arr, dist):
    cd = cdist(arr, arr)

    xt, yt = np.tril_indices(cd.shape[0], -1)

    mean_cd = np.mean(cd[xt, yt])
    std_cd = np.std(cd[xt, yt])

    mask = dist <= (mean_cd + std_cd)

    return mask


@jit
def evals_ratio(points, evecs, evals, t_threshold=0.02):
    ratio = (evals.T / np.sum(evals, axis=1)).T
    t = np.min(ratio, axis=1)

    t_mask = t <= t_threshold

    return t_mask


@jit
def angles_majority(points, angles, radius, weighted=True):
    distance, indices = set_nbrs_rad(points, points, radius)

    new_angles = np.zeros([angles.shape[0]])

    for i, (d, ids) in enumerate(zip(distance, indices)):
        if weighted is False:
            new_angles[i] = np.mean(angles[ids])

        elif weighted is True:
            mask = angles[ids] != 0
            if np.sum(mask > 0):
                new_angles[i] = np.average(angles[ids][mask], weights=(1 / d[mask]))

    return new_angles
