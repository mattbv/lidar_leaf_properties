# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 16:34:10 2017

@author: Matheus
"""

import numpy as np


def evals_ratio(points, evecs, evals, t_threshold=0.02, d_threshold=1):

    ratio = (evals.T / np.sum(evals, axis=1)).T
    t = np.min(ratio, axis=1)

    t_mask = t <= t_threshold

    return t_mask
