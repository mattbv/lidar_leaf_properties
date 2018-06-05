#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 11:07:50 2018

@author: matheus
"""

import numpy as np
from leaf_geometry import triangulate_cloud
import sys
sys.path.append('..')
from utility.area import triangle_area
from utility.point_utils import remove_duplicates

def area_from_points(arr, knn, r_thres=0.1, d_thresh=0.04):
    
    arr = remove_duplicates(arr[:, :3])
    
    triangulation = triangulate_cloud(arr, knn, r_thres, d_thresh)
    tri_area = triangulation_area(arr, triangulation)
    
    return np.nansum(tri_area)
    

def triangulation_area(arr, tri):
    
    area = []
    for t in tri:
        area.append(triangle_area(arr[t].astype(float)))
        
    return area
