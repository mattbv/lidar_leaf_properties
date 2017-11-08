# -*- coding: utf-8 -*-
"""
Main module to estimate leaf angle distribution from a point cloud.

@author: Matheus Boni Vicari (matheus.boni.vicari@gmail.com)
"""

import numpy as np
from scipy.spatial.distance import cdist
from filters import evals_ratio
from angles import norm_to_angle
from angles import angle_from_zenith
from nnsearch import set_nbrs_knn
from point_utils import get_center
from point_utils import remove_duplicates
from normals import from_cloud as cloud_normal


def angle_from_points(points, knn, r_thres=0.1):

    # Removing duplicate points from arr.
    points = remove_duplicates(points)

    # Calculating normal vectors and eigenvalues for each point in arr.
    normal_vectors, cc, evals = point_normals(points, knn)

    # Masking normals based on eigenvalues ratios.
    filter_mask = evals_ratio(points[:, :3], normal_vectors, evals,
                              r_thres)

    # Calculating absolute angles between each normal vector and horizontal
    # plane [0, 0, 1].
    abs_angles = norm_to_angle(normal_vectors)

    # Calculating zenithal angle, here defined as smallest angle from the
    # vertical axis (0, 180 or 360).
    zenith_angles = np.array([angle_from_zenith(x) for x in abs_angles])

#    # Setting bad quality points' angles (detected by filtering normals) to 0.
    zenith_angles[~filter_mask] = np.nan

    return zenith_angles


def point_normals(arr, knn):

    # Getting center of point cloud. This will be later used to correct
    # orientation of normal vectors.
    center = get_center(arr)

    # Obtaining neighborhood parameters.
#    dist, indices = set_nbrs_rad(arr, arr, rad, return_dist=True)
    dist, indices = set_nbrs_knn(arr, arr, knn, return_dist=True)

    # Initializing normals, centers and eigenvalues arrays.
    nn = np.full([arr.shape[0], 3], np.nan)
    cc = np.full([arr.shape[0], 3], np.nan)
    ss = np.full([arr.shape[0], 3], np.nan)

    # Looping over each neighborhood set and calculating normal parameters.
    for i, (idx, d) in enumerate(zip(indices, dist)):
        # Calculates and stores neihborhood centroids, normal vectors and
        # eigenvalues.
        C, N, S = cloud_normal(arr[idx])
        nn[i] = N
        cc[i] = C
        ss[i] = S

    # Calculating distance from each point in 'arr' to 'center'.
    arr_to_center = cdist(arr, center.reshape([1, 3]))

    # Creating a scaled set of normal vectors. These scaled vectors witll
    # be used to detect if the normal vectors are pointed inwards or outwards.
    scaled_evecs = nn * (0.1 * arr_to_center)

    # Calculating distance of each scalled vector to center.
    norm_to_base = cdist((arr + scaled_evecs), center.reshape([1, 3]))

    # Creating correction coefficients. Sets 1 where vectors are pointed
    # outwards (desired direction) and -1 where vectores are pointed inwards.
    corr_coeff = np.where(norm_to_base < arr_to_center, -1, 1)

    # Correcting (if necessary) normal vectors by multiplying them to
    # correction coefficients. This will make sure to invert vectors pointed
    # inwards.
    norm_vec = nn * corr_coeff

    return norm_vec, cc, ss
