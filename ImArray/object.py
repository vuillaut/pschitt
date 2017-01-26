"""
ImArray is a software to image an object in the sky by an array of ground Telescopes
Copyright (C) 2016  Thomas Vuillaume

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>

The author may be contacted @
thomas.vuillaume@lapp.in2p3.fr
"""


import numpy as np
from math import *
import matplotlib.pyplot as plt
import geometry as geo



def linear_segment(shower_top, shower_bot, n):
    """
    Homogeneous repartition of points following a linear segment
    :param shower_top: array
    :param shower_bot: array
    :param n: number of points
    :return: array of point arrays
    """
    top = np.array(shower_top)
    vec = top - np.array(shower_bot)
    l = np.linspace(0,1,n)
    shower = []
    for i in l:
        shower.append(top - vec * i)
    return np.array(shower)


def random_surface_sphere(shower_center, shower_radius, n):
    """
    Random repartition of points on a sphere surface
    :param shower_center: array - center of the sphere
    :param shower_radius: float - radius of the sphere
    :param n: number of points
    :return: list of point arrays
    """
    shower = []
    theta,phi = pi * np.random.random_sample(n), 2. * pi * np.random.random_sample(n)
    x = shower_center[0] + shower_radius * np.sin(theta) * np.cos(phi)
    y = shower_center[1] + shower_radius * np.sin(theta) * np.sin(phi)
    z = shower_center[2] + shower_radius * np.cos(theta)
    for i in np.arange(n):
        shower.append([x[i],y[i],z[i]])
    return shower


def random_surface_ellipsoide(shower_center, shower_length, shower_width, n):
    """
    Random repartition of points on an ellipsoid surface
    :param shower_center: array - center of the ellipsoid
    :param shower_length: float - length of the ellipsoid
    :param shower_width: float - width of the ellipsoid
    :param n: int - number of points
    :return: list of point arrays
    """
    shower = []
    theta,phi = pi * np.random.random_sample(n), 2. * pi * np.random.random_sample(n)
    x = shower_center[0] + shower_width * np.sin(theta) * np.cos(phi)
    y = shower_center[1] + shower_width * np.sin(theta) * np.sin(phi)
    z = shower_center[2] + shower_length * np.cos(theta)
    for i in np.arange(n):
        shower.append([x[i],y[i],z[i]])
    return shower


def random_ellipsoide_alongz(shower_center, shower_length, shower_width, n):
    """
    Random repartition of points in an ellipsoid aligned with the Z axis
    :param shower_center: array - center of the ellipsoid
    :param shower_length: float - length of the ellipsoid
    :param shower_width: float - width of the ellipsoid
    :param n: int - number of points
    :return: list of point arrays
    """
    shower = []
    theta,phi = pi * np.random.random_sample(n), 2. * pi * np.random.random_sample(n)
    q,p = shower_length * np.random.random_sample(n), shower_width * np.random.random_sample(n)
    x = shower_center[0] + p/2. * np.sin(theta) * np.cos(phi)
    y = shower_center[1] + p/2. * np.sin(theta) * np.sin(phi)
    z = shower_center[2] + q/2. * np.cos(theta)
    for i in np.arange(n):
        shower.append([x[i],y[i],z[i]])
    return shower


def shifted_ellipsoide_v1(shower_center, shower_length, shower_width, n, p, origin_altitude):
    '''
    p = impact point coordinates
    '''
    shower = random_ellipsoide_alongz(shower_center, shower_length, shower_width, n)
    for point in shower:
        point[0] += p[0]*(- point[2]/origin_altitude + 1.0)
        point[1] += p[1]*(- point[2]/origin_altitude + 1.0)
    return shower

def plot3d(shower):
    X = []
    Y = []
    Z = []
    for points in shower:
        X.append(points[0])
        Y.append(points[1])
        Z.append(points[2])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X, Y, Z)
    plt.show()


def random_ellipsoide(shower_top_altitude, shower_length, shower_width, alt, az, impact_point, n):
    """
    Compute a list of N random points in an ellipsoid. The ellipsoid comes from direction (alt,az) and goes through impact_point
    :param shower_top_altitude: array
    :param shower_length: float
    :param shower_width: float
    :param alt: float
    :param az: float
    :param impact_point: array
    :param n: int - number of points
    :return: list of 3-floats arrays
    """
    shower_center = [0,0,shower_top_altitude - shower_length/2.]
    shower = random_ellipsoide_alongz(shower_center, shower_length, shower_width, n)
    shower_rot = []
    for point in shower:
        point_rot = geo.rotation_matrix_z(az) * geo.rotation_matrix_y(pi/2. - alt) * np.matrix(point).T
        #rotpoint = geo.rotation_matrix_y(rot)*np.matrix(point).T
        shower_rot.append(np.array(np.asarray(point_rot.T)[0]) + np.array(impact_point))
    return np.array(shower_rot)
