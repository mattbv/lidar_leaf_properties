# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 08:18:04 2017

@author: Matheus
"""

import glob

import numpy as np

fname = '1178'
folder = r'E:\test_leaf_prop\points\%s/' % (fname)
print(folder)

files = glob.glob(folder + '*.txt')

for i, f in enumerate(files):

    print('Processing: %s' % f)

    data = np.loadtxt(f, delimiter=',')

    classes = np.unique(data[:, 3])

    leaf_data = data[data[:, 3] == np.max(classes)]

    out_arr = np.vstack((leaf_data[:, 1], leaf_data[:, 0], leaf_data[:, 2],
                         leaf_data[:, 3])).T

    np.savetxt(folder + '%s_%s.txt' % (fname, i), out_arr, fmt='%1.4f')
