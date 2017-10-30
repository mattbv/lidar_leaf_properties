# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 16:32:38 2017

@author: Matheus
"""

import numpy as np


def norm_to_angle(norm_vec):

    angles = np.zeros(norm_vec.shape[0])

    for i, nn in enumerate(norm_vec):
        angles[i] = angle([0., 0., 1.], nn, True)

    return rad_to_degree(angles)


def angle(v1, v2, acute):
    # v1 is your firsr vector
    # v2 is your second vector
    angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) *
                      np.linalg.norm(v2)))

    return angle


def rad_to_degree(arr):
    return arr * (180 / np.pi)


def angle_from_zenith(angle):
    quad = np.array([0, 180, 360])
    return np.min(np.abs(angle - quad))
