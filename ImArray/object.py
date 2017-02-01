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
#from math import *
import matplotlib.pyplot as plt
import geometry as geo
import math
import random



class shower:
    """
    Class handling shower object
    """
    def __init__(self):
        """
        Init
        """
        self.type = "Shower"
        self.tab = np.array([])


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


def linear_segment_2(shower_top, shower_bot, n):
    """
    Homogeneous repartition of points following a linear segment
    :param shower_top: array
    :param shower_bot: array
    :param n: number of points
    :return: tuple (X,Y,Z) of the points coordinates
    """
    top = np.array(shower_top)
    vec = top - np.array(shower_bot)
    l = np.linspace(0,1,n)
    x = np.empty(n)
    y = np.empty(n)
    z = np.empty(n)
    for idx,i in enumerate(l):
        point = top - vec * i
        x[idx] = point[0]
        y[idx] = point[1]
        z[idx] = point[2]
    return (x, y, z)



def random_surface_sphere(shower_center, shower_radius, n):
    """
    Random repartition of points on a sphere surface
    :param shower_center: array - center of the sphere
    :param shower_radius: float - radius of the sphere
    :param n: number of points
    :return: list of point arrays
    """
    shower = []
    theta,phi = math.pi * np.random.random_sample(n), 2. * math.pi * np.random.random_sample(n)
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
    theta,phi = math.pi * np.random.random_sample(n), 2. * math.pi * np.random.random_sample(n)
    x = shower_center[0] + shower_width * np.sin(theta) * np.cos(phi)
    y = shower_center[1] + shower_width * np.sin(theta) * np.sin(phi)
    z = shower_center[2] + shower_length * np.cos(theta)
    for i in np.arange(n):
        shower.append([x[i],y[i],z[i]])
    return shower


def random_ellipsoide_alongz(shower_center, shower_length, shower_width, n):
    """
    Random repartition of points in an ellipsoid aligned with the Z axis

    Parameters
    ----------
    shower_center: position in space of the ellipsoid center
    shower_length: length of the ellipsoid = float
    shower_width: width of the ellipsoid = float
    n: number of points forming the shower = int

    Returns
    -------
    list of points forming the shower (3-floats arrays)
    """
    shower = []
    theta,phi = math.pi * np.random.random_sample(n), 2. * math.pi * np.random.random_sample(n)
    q,p = shower_length * np.random.random_sample(n), shower_width * np.random.random_sample(n)
    x = shower_center[0] + p/2. * np.sin(theta) * np.cos(phi)
    y = shower_center[1] + p/2. * np.sin(theta) * np.sin(phi)
    z = shower_center[2] + q/2. * np.cos(theta)
    for i in np.arange(n):
        shower.append([x[i],y[i],z[i]])
    return shower


def shifted_ellipsoide_v1(shower_center, shower_length, shower_width, n, p, origin_altitude):
    """
    First version of the ellipsoid shower - depreciated
    Parameters
    ----------
    shower_center: central point of the ellipsoid
    shower_length: length of the ellipsoid
    shower_width: width of the ellipsoid
    n: number of points forming the shower
    p: impact point
    origin_altitude: altitude of the first interaction

    Returns
    -------
    shower: list of points (3-floats arrays)
    """
    shower = random_ellipsoide_alongz(shower_center, shower_length, shower_width, n)
    for point in shower:
        point[0] += p[0]*(- point[2]/origin_altitude + 1.0)
        point[1] += p[1]*(- point[2]/origin_altitude + 1.0)
    return shower

def plot3d(shower):
    """
    Display a 3d plot of a shower

    Parameters
    ----------
    shower: list of position points (3-floats arrays)
    """
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
    Parameters
    ----------
    shower_top_altitude: position of the first interaction point = 3-floats array
    shower_length: length of the ellipsoide = float
    shower_width: width of the ellipsoide = float
    alt: altitude angle of the shower
    az: azimuthal angle of the shower
    impact_point: point on the shower axis
    n: number of points forming the shower

    Returns
    -------
    list of points in the shower (3-floats arrays)
    """
    shower_center = [0,0,shower_top_altitude - shower_length/2.]
    shower = random_ellipsoide_alongz(shower_center, shower_length, shower_width, n)
    shower_rot = []
    for point in shower:
        point_rot = geo.rotation_matrix_z(az) * geo.rotation_matrix_y(math.pi/2. - alt) * np.matrix(point).T
        #rotpoint = geo.rotation_matrix_y(rot)*np.matrix(point).T
        shower_rot.append(np.array(np.asarray(point_rot.T)[0]) + np.array(impact_point))
    return np.array(shower_rot)


def Gaisser_Hillas(Nmax, X1, Xmax, X):
    lamb = 70.
    A = Xmax - X1
    return Nmax * math.pow((X - X1) / A, A / lamb) * math.exp((Xmax - X) / lamb)


def NKG(X1, X, r):
    rm = 118.
    t = (X - X1) / 36.7
    tmax = 1.7 + 0.76 * (math.log(1e12 / 81e6) - math.log(1))
    s = 2 * t / (t + tmax)

    # print(r/118)
    # print(s-2)
    # G1=math.pow(r/118.,s-2)
    Cs = math.gamma(4.5 - s) / (math.pi * 2 * rm * rm * math.gamma(s) * math.gamma(4.5 - 2 * s))
    G = 1e6 * Cs * math.pow(r / rm + 1, s - 4.5) * math.pow((r / rm), (s - 2))
    return G


def gcm2distance(X, theta):
    if X >= 631.1:
        return -9941.8638 * math.log((186.5562 + (X * math.cos(theta))) / 1222.6562) / math.cos(theta)

    if X > 271.7 and X < 631.1:
        return -8781.5355 * math.log((94.9199 + (X * math.cos(theta))) / 1144.9069) / math.cos(theta)

    if X > 3.039 and X <= 271.7:
        return -6361.4304 * math.log((-0.61289 + (X * math.cos(theta))) / 1305.5948) / math.cos(theta)

    if X >= 0.00182 and X <= 3.039:
        return -7721.7016 * math.log((X * math.cos(theta)) / 540.1778) / math.cos(theta)


def Get_pos_Gaisser_Hillas(Npart, alt, az, impact_point):
    shower = []
    f = []
    N = []
    Xinterp = []
    Rinterp = []
    dist = []
    r = []
    radius = []
    ang = []
    x = []
    y = []
    z = []

    for i in range(1000):
        f.append(i)

    r = np.arange(1, 350., 10)
    print(r)

    # plt.plot(r)
    # plt.show()
    for j in f:
        N.append(Gaisser_Hillas(Npart, 0., 600, j))
    Nsum = np.cumsum(N)

    for k in range(Npart):
        D = []
        yval = random.uniform(0, max(Nsum))
        Xinterp.append(np.interp(yval, Nsum, f))
        for m in range(35):
            D.append(NKG(0, np.interp(yval, Nsum, f), r[m]))
        Dsum = np.cumsum(D)
        # plt.plot(r,D)
        # plt.yscale("log")
        # plt.show()

        # rand2=random.uniform(0, max(D))

        # plt.plot(r,D)
        # plt.xlim(1e-1, 1e2)
        # plt.ylim(min(Dsum), max(Dsum))
        ##plt.xscale('log')
        # plt.yscale('log')
        # plt.show()

        yval2 = random.uniform(min(D), max(D))
        dist.append(gcm2distance(np.interp(yval, Nsum, f), pi / 2 - alt))

        radius.append(np.interp(yval2, r, D))
        # print(min(D),max(D),random.uniform(min(D), max(D)))
        x.append(radius[k] * cos(random.uniform(0, 2 * math.pi)))
        y.append(radius[k] * sin(random.uniform(0, 2 * math.pi)))
        z.append(dist[k])
    # plt.plot(dist,radius)
    # plt.show()
    for i in range(Npart):
        shower.append([x[i], y[i], z[i]])
    shower_rot = []
    for point in shower:
        point_rot = geo.rotation_matrix_z(az) * geo.rotation_matrix_y(pi / 2. - alt) * np.matrix(point).T
        # rotpoint = geo.rotation_matrix_y(rot)*np.matrix(point).T
        shower_rot.append(np.array(np.asarray(point_rot.T)[0]) + np.array(impact_point))
    return np.array(shower_rot)