# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 09:41:51 2017

@author: Matheus
"""

import mayavi.mlab as mlab
import numpy as np
from test_new_approach import calculate_eigen, filt_normals, norm_to_angle, angle_from_zenith, calculate_eigen_mesh, set_nbrs_knn
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
from scipy.stats import kendalltau
import seaborn as sns
sns.set(style="ticks")


def test_plane():

    test = np.zeros([100, 3])
    test[:, :2] = np.random.rand(100, 2)

    norm_vec, evals = calculate_eigen(test[:, :3], 0.3)

    mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
    mlab.points3d(test[:, 0], test[:, 1], test[:, 2], scale_factor=0.05)
    mlab.quiver3d(test[:, 0], test[:, 1], test[:, 2],
                  norm_vec[:, 0], norm_vec[:, 1], norm_vec[:, 2],
                  color=(1, 0, 0))


def test_double_plane():

    test1 = np.zeros([100, 3])
    test1[:, :2] = np.random.rand(100, 2)

    test2 = np.zeros([100, 3])
    test2[:, 1:] = np.random.rand(100, 2)

    test = np.vstack((test1, test2))

    norm_vec, evals = calculate_eigen(test[:, :3], 0.3)
    ratio = (evals.T / np.sum(evals, axis=1)).T
    t = np.min(ratio, axis=1)

    mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
    mlab.points3d(test[:, 0], test[:, 1], test[:, 2], t,
                  scale_mode='none', scale_factor=0.05)
    mlab.quiver3d(test[:, 0], test[:, 1], test[:, 2],
                  norm_vec[:, 0], norm_vec[:, 1], norm_vec[:, 2],
                  color=(1, 0, 0))


def test_leaf_block():

    leaf = np.loadtxt(r'D:\Dropbox\PhD\Scripts\phd-geography-ucl\lidar_data\tls\tls_leaf_prop\tls_leaf_prop\data/test_leaf_2.txt')
    norm_vec, evals = calculate_eigen(leaf[:, :3], 0.01)
    pts, vec, val = filt_normals(leaf, norm_vec, evals)

    angles = norm_to_angle(vec)


    mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
    mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], angles,
                  scale_mode='none', scale_factor=0.001)
    mlab.quiver3d(pts[:, 0], pts[:, 1], pts[:, 2],
                  vec[:, 0], vec[:, 1], vec[:, 2],
                  color=(1, 0, 0))


def test_leaf_angles():

    leaf = np.loadtxt(r'D:\Dropbox\PhD\Scripts\phd-geography-ucl\lidar_data\tls\tls_leaf_prop\tls_leaf_prop\data/test_leaf_2.txt')
    norm_vec, evals = calculate_eigen(leaf[:, :3], 0.01)
    pts, vec, val = filt_normals(leaf, norm_vec, evals)

    angles = norm_to_angle(vec)
    zenith_angles = np.array([angle_from_zenith(x) for x in angles])

    plt.figure()
    plt.hist([angles, zenith_angles], 10)


def test_leaf_facets():

    facets_file = r'D:\Dropbox\PhD\Data\librat\rushworth\parsed_data\obj/1001_etri_erecto_obj_facets_.csv'
    triangulation_file = r'D:\Dropbox\PhD\Data\librat\rushworth\parsed_data\obj/1001_etri_erecto_triangulation.npy'

    facets = pd.read_csv(facets_file)
    leaf_facets = facets.loc[facets['material'] == 'new_leaf']
    triangulation = np.load(triangulation_file)

    norm, centers = calculate_eigen_mesh(leaf_facets)

    mlab.triangular_mesh(facets['vx'], facets['vy'], facets['vz'], triangulation)
    mlab.quiver3d(centers[:, 0], centers[:, 1], centers[:, 2],
                  norm[:, 0], norm[:, 1], norm[:, 2],
                  color=(1, 0, 0))

    angles = norm_to_angle(norm)
    zenith_angles = np.array([angle_from_zenith(x) for x in angles])

    plt.figure()
    plt.hist([angles, zenith_angles], 10)


def test_compare():

    facets_file = r'D:\Dropbox\PhD\Data\librat\rushworth\parsed_data\obj/1001_etri_erecto_obj_facets_.csv'
    triangulation_file = r'D:\Dropbox\PhD\Data\librat\rushworth\parsed_data\obj/1001_etri_erecto_triangulation.npy'

    facets = pd.read_csv(facets_file)
    leaf_facets = facets.loc[facets['material'] == 'new_leaf']
    triangulation = np.load(triangulation_file)

    norm, centers = calculate_eigen_mesh(leaf_facets)

    angles = norm_to_angle(norm)
    zenith_angles = np.array([angle_from_zenith(x) for x in angles])

    leaf = np.loadtxt(r'D:\Dropbox\PhD\Scripts\phd-geography-ucl\lidar_data\tls\tls_leaf_prop\tls_leaf_prop\data/test_leaf_1.txt')
    norm_vec, evals = calculate_eigen(leaf[:, :3], 0.008)
    pts, vec, val = filt_normals(leaf, norm_vec, evals)

    angles1 = norm_to_angle(vec)
    zenith_angles1 = np.array([angle_from_zenith(x) for x in angles1])

    plt.figure()
    plt.hist([angles, zenith_angles], 10)
    plt.figure()
    plt.hist([angles1, zenith_angles1], 10)


def test_compare_pointwise():

    facets_file = r'D:\Dropbox\PhD\Data\librat\rushworth\parsed_data\obj/1001_etri_erecto_obj_facets_.csv'
    triangulation_file = r'D:\Dropbox\PhD\Data\librat\rushworth\parsed_data\obj/1001_etri_erecto_triangulation.npy'

    facets = pd.read_csv(facets_file)
    leaf_facets = facets.loc[facets['material'] == 'new_leaf']
    triangulation = np.load(triangulation_file)

    norm, centers = calculate_eigen_mesh(leaf_facets)

    angles = norm_to_angle(norm)
    zenith_angles = np.array([angle_from_zenith(x) for x in angles])

    leaf = np.loadtxt(r'D:\Dropbox\PhD\Scripts\phd-geography-ucl\lidar_data\tls\tls_leaf_prop\tls_leaf_prop\data/test_leaf_1.txt')
    norm_vec, evals = calculate_eigen(leaf[:, :3], 0.008)
    pts, vec, val = filt_normals(leaf, norm_vec, evals)

    angles1 = norm_to_angle(vec)
    zenith_angles1 = np.array([angle_from_zenith(x) for x in angles1])


    indices = set_nbrs_knn(centers, pts, 1, False)
    new_angles = angles[indices]


    sns.jointplot(zenith_angles[indices.flatten()], zenith_angles1, kind="hex", stat_func=kendalltau, color="#4CB391")

#    plt.figure()
#    plt.hist([angles, zenith_angles], 10)
#    plt.figure()
#    plt.hist([angles1, zenith_angles1], 10)


def test_compare_batch():

    rad = 0.008

    facets_folder = r'E:\test_leaf_prop\facets/'
    points_folder = r'E:\test_leaf_prop\points/'
    output_folder = r'E:\test_leaf_prop\figures/'

    facets_files = glob.glob(facets_folder + '*facets_.csv')
    points_files = glob.glob(points_folder + '*.txt')

    for f, p in zip(facets_files, points_files):

        fname = os.path.splitext(os.path.basename(f))[0]
        pname = os.path.splitext(os.path.basename(p))[0].split('_')[-1]

        assert fname.split('_')[0] == pname

        output_file = output_folder + fname + '.png'

        if not os.path.isfile(output_file):
            try:
                facets = pd.read_csv(f)
                leaf_facets = facets.loc[facets['material'] == 'new_leaf']
                facets_norm, facets_centers = calculate_eigen_mesh(leaf_facets)

                facets_angles = norm_to_angle(facets_norm)
                facets_zenith_angles = np.array([angle_from_zenith(x) for x in
                                                 facets_angles])

                points = np.loadtxt(p)
                points_norm, points_evals = calculate_eigen(points[:, :3], rad)
                pts, vec, val = filt_normals(points[:, :3], points_norm,
                                             points_evals)

                points_angles = norm_to_angle(vec)
                points_zenith_angles = np.array([angle_from_zenith(x) for x in
                                                 points_angles])

                upscale_indices = set_nbrs_knn(facets_centers, pts, 1, False)
        #        new_facet_angles = facets_angles[upscale_indices.flatten()]
                new_facet_zenith_angles = facets_zenith_angles[upscale_indices.flatten()]

                sns_plot = sns.jointplot(new_facet_zenith_angles,
                                         points_zenith_angles, kind="hex",
                                         stat_func=kendalltau, color="#4CB391")

                sns_plot.savefig(output_folder + fname + '.png')
                plt.close()
            except Exception as e:
                print e
                pass


if __name__ == "__main__":

#    test_plane()
#    test_double_plane()
#    test_leaf_block()
#    test_leaf_angles()
#    test_leaf_facets()
#    test_compare()
#    test_compare_pointwise()
    test_compare_batch()
