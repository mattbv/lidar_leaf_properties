# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 17:21:23 2017

@author: Matheus
"""

from numba import jit
import numpy as np


@jit
def tri_normal(tri):

    n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
    return n / np.linalg.norm(n)
