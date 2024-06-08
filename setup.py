# Copyright (c) 2017-2024, Matheus Boni Vicari
# All rights reserved.
#
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.


__author__ = "Matheus Boni Vicari"
__copyright__ = "Copyright 2017-2024"
__credits__ = ["Matheus Boni Vicari"]
__license__ = "GPL3"
__version__ = "0.1"
__maintainer__ = "Matheus Boni Vicari"
__email__ = "matheus.boni.vicari@gmail.com"
__status__ = "Development"

from setuptools import find_packages, setup


def readme():
    with open("README.rst") as f:
        return f.read()


with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="leafproperties",
    version="0.1",
    author="Matheus Boni Vicari",
    author_email="matheus.boni.vicari@gmail.com",
    packages=find_packages("leafproperties"),
    entry_points={},
    url="https://github.com/mattbv/lidar_leaf_properties",
    license="LICENSE.txt",
    description="Extracts and estimates information from leaf points in a TLS point cloud.",
    long_description=readme(),
    classifiers=["Programming Language :: Python", "Topic :: Scientific/Engineering"],
    keywords="TLS, LiDAR, leaf, point cloud, vegetation, forest, ecology, 3D, point cloud, point cloud processing, point cloud analysis, point cloud features, point cloud features extraction, point cloud features estimation",
    install_requires=required,
)
