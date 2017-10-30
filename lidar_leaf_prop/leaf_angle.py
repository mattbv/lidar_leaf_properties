# -*- coding: utf-8 -*-
"""
@author: Matheus
"""

import numpy as np
from scipy import linalg as LA
from scipy.spatial.distance import cdist
import mayavi.mlab as mlab
from normals import tri_normal
from filters import evals_ratio
from angles import norm_to_angle
from angles import angle_from_zenith
from nnsearch import set_nbrs_rad
from point_utils import get_center


def angle_from_points(points, rad):

    # Removing duplicate points from arr.
    arr = remove_duplicates(arr)

    # Calculating normal vectors and eigenvalues for each point in arr.
    normal_vectors, eigenvals = point_normals(arr, rad)

    # Masking normals based on eigenvalues ratios.
    filter_mask = evals_ratio(arr[:, :3], normal_vectors, eigenvals)

    # Calculating absolute angles between each normal vector and horizontal
    # plane [0, 0, 1].
    abs_angles = norm_to_angle(normal_vectors)

    # Calculating zenithal angle, here defined as smallest angle from the
    # vertical axis (0, 180 or 360).
    zenith_angle = angle_from_zenith(abs_angles)

    # Setting bad quality points' angles (detected by filtering normals) to 0.
    zenith_angle[~filter_mask] = 0

    return zenith_angle


def angle_from_mesh(facets):

    # Calculating normal vectors and eigenvalues for facet in facets.
    normal_vectors, eigenvals = mesh_normals(facets)

    # Calculating absolute angles between each normal vector and horizontal
    # plane [0, 0, 1].
    abs_angles = norm_to_angle(normal_vectors)

    # Calculating zenithal angle, here defined as smallest angle from the
    # vertical axis (0, 180 or 360).
    zenith_angle = angle_from_zenith(abs_angles)

    return zenith_angle


def point_normals(arr, rad):

    base = get_center(arr)

    dist, indices = set_nbrs_rad(arr, arr, rad, return_dist=True)

    evecs = np.zeros([arr.shape[0], 3])
    evals = np.zeros([arr.shape[0], 3])

    for i, (idx, d) in enumerate(zip(indices, dist)):

        masked_ids = idx[d <= (np.mean(d) + np.std(d))]

        if len(masked_ids) >= 3:
            cov = np.cov(arr[masked_ids].T)
            e_vals, e_vecs = LA.eig(cov)
            ev_id = np.argmin(e_vals)
            evecs[i] = e_vecs[ev_id]
            evals[i] = e_vals

    arr_to_base = cdist(arr, base.reshape([1, 3]))

    scaled_evecs = evecs * (0.1 * arr_to_base)

    norm_to_base = cdist((arr + scaled_evecs), base.reshape([1, 3]))
    corr_coeff = np.where(norm_to_base < arr_to_base, -1, 1)
    norm_vec = evecs * corr_coeff

    return norm_vec, evals


def mesh_normals(facets):

    coords = np.asarray(facets[['vx', 'vy', 'vz']])

    base = get_center(coords)

    n_facets = coords.shape[0] / 3

    evecs = np.zeros([n_facets, 3])
    center = np.zeros([n_facets, 3])

    for i, (j) in enumerate(range(0, coords.shape[0], 3)):
        facet_coords = coords[j:j+3, :]
        if facet_coords.shape[0] == 3:
            evecs[i] = tri_normal(facet_coords)
            center[i] = np.mean(facet_coords, axis=0)

    arr_to_base = cdist(center, base.reshape([1, 3]))

    scaled_evecs = evecs * (0.1 * arr_to_base)

    norm_to_base = cdist((center + scaled_evecs), base.reshape([1, 3]))
    corr_coeff = np.where(norm_to_base < arr_to_base, -1, 1)
    norm_vec = evecs * corr_coeff

    return norm_vec, center
