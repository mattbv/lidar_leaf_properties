# -*- coding: utf-8 -*-
"""
Main module to estimate leaf angle distribution from a point cloud.

@author: Matheus Boni Vicari (matheus.boni.vicari@gmail.com)
"""

import numpy as np

from leafproperties.angles import norm_to_angle
from leafproperties.filters import evals_ratio
from leafproperties.normals import normals_from_cloud
from leafproperties.point_utils import remove_duplicates


def angle_from_points(points, knn, r_thres=0.1):
    # Removing duplicate points from arr.
    points = remove_duplicates(points[:, :3])

    # Calculating normal vectors and eigenvalues for each point in arr.
    normal_vectors, cc, evals = normals_from_cloud(points)

    # Masking normals based on eigenvalues ratios.
    filter_mask = evals_ratio(points[:, :3], normal_vectors, evals, r_thres)

    # Calculating absolute angles between each normal vector and horizontal
    # plane [0, 0, 1].
    abs_angles = norm_to_angle(normal_vectors)

    #    # Setting bad quality points' angles (detected by filtering normals) to 0.
    abs_angles[~filter_mask] = np.nan

    return abs_angles
