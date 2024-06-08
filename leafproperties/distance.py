#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 11:04:37 2018

@author: matheus
"""

import numpy as np


def euclidean_distance(p1, p2):
    return np.hypot(p1[0] - p2[0], p1[1] - p2[1])
