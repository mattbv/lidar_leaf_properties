#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 11:03:16 2018

@author: matheus
"""

import numpy as np
from distance import euclidean_distance as dist


def triangle_area(vertices):
    
    side_a = dist(vertices[0], vertices[1])
    side_b = dist(vertices[1], vertices[2])
    side_c = dist(vertices[2], vertices[0])
    s = 0.5 * ( side_a + side_b + side_c)
    
    return np.sqrt(s * (s - side_a) * (s - side_b) * (s - side_c))
