#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Matheus Boni Vicari (matheus.boni.vicari@gmail.com)
"""

import numpy as np
import sys
sys.path.append('..')
from utility.nnsearch import set_nbrs_knn
from utility.point_utils import remove_duplicates
from utility.normals import normals_from_cloud
from scipy.spatial import Delaunay
from sklearn.decomposition import PCA


def triangulate_cloud(arr, knn, eval_threshold, dist_threshold):
    
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
    
    evals = np.array([(i/np.sum(i)) for i in ss])
    
    evals_mask = evals <= eval_threshold
    
    valid_ids = np.where(evals_mask)[0]
    
    triangles = []
    
    for i in valid_ids:
        nbr_ids = indices[i]
        nbr_dist = dist[i]
        tri_ids = nbr_ids[nbr_dist <= dist_threshold]
        try:
            X = arr[tri_ids]
            pca = PCA(2).fit(X)
            Y = pca.transform(X)
    
            tri = Delaunay(Y)
            simp = tri.simplices
    
            for s in simp:
                if 0 in s:
                    triangles.append(tri_ids[s])
        except:
            pass                  

    tri_sorted = np.array([np.sort(i) for i in triangles])
    tri_sorted = remove_duplicates(tri_sorted)
    
    return tri_sorted.astype(int)


def expand_triangulation(tri):

    tri_mesh = np.zeros([len(tri) * 3, 3], dtype=int)
    for j, t in enumerate(tri):
        base_id = 3 * j
        tri_mesh[base_id, :] = [t[0], t[1], t[2]]
        tri_mesh[base_id + 1, :] = [t[1], t[2], t[0]]
        tri_mesh[base_id + 2, :] = [t[2], t[0], t[1]]

    return tri_mesh
