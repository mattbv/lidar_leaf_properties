=====================
lidar_leaf_properties
=====================

This is a library for extracting structural properties about tree leaves represented by point clouds.

The lidar_leaf_properties library is being developed as part of my PhD research, supervised by Dr. Mat Disney, in the Department of Geography at University College London (UCL). My research 
is funded through Science Without Borders from the National Council of Technological and Scientific Development (10.13039/501100003593) – Brazil (Process number 233849/2014-9). 

Any questions or suggestions, feel free to contact me using one of the following e-mails: matheus.boni.vicari@gmail.com or matheus.vicari.15@ucl.ac.uk


To-Do:
- Add documentation
- Add tests

Installation
------------
To install the package, run the following command in the terminal:
        
        .. code-block:: bash

                python setup.py install
        
Usage
-----
To use the package, run the following command in the terminal:

        .. code-block:: python

                import leafproperties

                # Load the point cloud
                pc = np.loadtxt('path/to/pointcloud.txt')[:, :3]  # Load the point cloud and keep only the x, y, and z columns

                leaf_angles = leafproperties.leaf_angle.angle_from_points(pc, knn=10)


