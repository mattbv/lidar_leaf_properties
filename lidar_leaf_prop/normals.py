# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 16:25:37 2017

@author: Matheus
"""

import numpy as np
from scipy import linalg as LA
from numba import jit


@jit
def eigen_normals(arr):
    cov = np.cov(arr.T)
    e_vals, e_vecs = LA.eig(cov)

    return e_vals, e_vecs


@jit
def tri_normal(tri):

    n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
    return n / np.linalg.norm(n)
