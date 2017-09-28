# Licensed under a 3-clause BSD style license - see LICENSE.rst


import numpy as np
import matplotlib.pyplot as plt
from . import geometry as geo
from math import pi
from copy import copy



class shower:
    """
    Class handling shower object
    """
    def __init__(self):
        """
        Init
        """
        self.type = "Shower"
        self.altitude_first_interaction = 0
        self.alt = np.deg2rad(70)
        self.az = 0
        self.impact_point = [0,0,0]
        self.energy_primary = 0
        self.number_of_particles = 10
        self.particles = np.empty((3,self.number_of_particles))

    def linear_segment(self, shower_first_interaction, shower_bot):
        """
        Homogeneous repartition of points following a linear segment
        Parameters
        ----------
        shower_first_interaction: 1D Numpy array - position of the first interaction point
        shower_bot: 1D Numpy array - position of the bottom of the shower
        """
        self.particles = linear_segment(shower_first_interaction, shower_bot, self.number_of_particles)


    def random_surface_sphere(self, shower_center, shower_radius):
        """
        Random repartition of points on a sphere surface

        Parameters
        ----------
        shower_center: 1D Numpy array
        shower_radius: float
        """
        self.particles = random_surface_sphere(shower_center, shower_radius, self.number_of_particles)


    def random_surface_ellipsoide(self, shower_center, shower_length, shower_width):
        self.particles = random_surface_ellipsoide(shower_center, shower_length, shower_width, self.number_of_particles)


    def random_ellipsoide_alongz(self, shower_center, shower_length, shower_width):
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
        Numpy array (3,n) - positions of particles in shower
        """
        self.particles = random_ellipsoide_alongz(shower_center, shower_length, shower_width, self.number_of_particles)


    def gaussian_ellipsoide_alongz(self, shower_center, shower_length, shower_width):
        """

        Parameters
        ----------
        shower_center
        shower_length
        shower_width
        """
        self.particles = gaussian_ellipsoide_alongz(shower_center, shower_length, shower_width, self.number_of_particles)


    def gaussian_ellipsoide(self, shower_top_altitude, shower_length, shower_width):
        """

        Parameters
        ----------
        shower_top_altitude
        shower_length
        shower_width
        """
        self.particles = gaussian_ellipsoide(shower_top_altitude, shower_length, shower_width, \
                                         self.alt, self.az, self.impact_point, self.number_of_particles)


    def shower_rot(self, alt, az):
        """
        Rotate the shower
        Parameters
        ----------
        alt: float
        az: float
        """
        self.particles = shower_array_rot(self.particles, alt, az)


    def plot3d(self):
        """
        Make a 3d plot
        """
        plot3d(self.particles)


    def random_ellipsoide(self, shower_top_altitude, shower_length, shower_width):
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
        self.particles = random_ellipsoide(shower_top_altitude, shower_length, shower_width, \
                                       self.alt, self.az, self.impact_point, self.number_of_particles)







def linear_segment(shower_top, shower_bot, n):
    """
    Homogeneous repartition of points following a linear segment
    Parameters
    ----------
    shower_first_interaction: 1D Numpy array - position of the first interaction point
    shower_bot: 1D Numpy array - position of the bottom of the shower
    n: int - number of points in shower

    Returns
    -------
    Numpy array (3,n) - positions of particles in shower
    """
    vec = shower_bot - shower_top
    l = np.linspace(0, 1, n)
    return shower_top + vec * np.array([l ,l ,l]).T


def random_surface_sphere(shower_center, shower_radius, n):
    """
    Random repartition of points on a sphere surface
    Parameters
    ----------
    shower_center: 1D Numpy array
    shower_radius: float
    n: int - number of particles in shower

    Returns
    -------
    Numpy array (3,n) - positions of particles in shower
    """
    theta = pi * np.random.random_sample(n)
    phi = 2. * pi * np.random.random_sample(n)
    x = shower_center[0] + shower_radius * np.sin(theta) * np.cos(phi)
    y = shower_center[1] + shower_radius * np.sin(theta) * np.sin(phi)
    z = shower_center[2] + shower_radius * np.cos(theta)
    return np.array([x,y,z]).T


def random_surface_ellipsoide(shower_center, shower_length, shower_width, n):
    """
    Random repartition of points on an ellipsoid surface

    Parameters
    ----------
    shower_center: 1D numpy array - position of the center of the sphere
    shower_length: float
    shower_width: float
    n: int - number of particles in shower

    Returns
    -------
    Numpy array (3,n) - positions of particles in shower
    """
    theta,phi = pi * np.random.random_sample(n), 2. * pi * np.random.random_sample(n)
    x = shower_center[0] + shower_width * np.sin(theta) * np.cos(phi)
    y = shower_center[1] + shower_width * np.sin(theta) * np.sin(phi)
    z = shower_center[2] + shower_length * np.cos(theta)
    return np.array([x, y, z]).T


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
    Numpy array (3,n) - positions of particles in shower
    """
    theta, phi = pi * np.random.random_sample(n), 2. * pi * np.random.random_sample(n)
    q, p = shower_length * np.random.random_sample(n), shower_width * np.random.random_sample(n)
    x = shower_center[0] + p/2. * np.sin(theta) * np.cos(phi)
    y = shower_center[1] + p/2. * np.sin(theta) * np.sin(phi)
    z = shower_center[2] + q/2. * np.cos(theta)
    return np.array([x, y, z]).T


def gaussian_ellipsoide_alongz(shower_center, shower_length, shower_width, n):
    """
    Gaussian repartition of points in an ellipsoid aligned with the Z axis
    Shower length and width correspond to 3 sigma of the gaussian

    Parameters
    ----------
    shower_center: position in space of the ellipsoid center
    shower_length: length of the ellipsoid = float
    shower_width: width of the ellipsoid = float
    n: number of points forming the shower = int

    Returns
    -------
    Numpy array (3,n) - positions of particles in shower
    """
    x = np.random.normal(shower_center[0], shower_width / 5., n)
    y = np.random.normal(shower_center[1], shower_width / 5., n)
    z = np.random.normal(shower_center[2], shower_length / 5., n)

    return np.array([x, y, z]).T


def gaussian_ellipsoide(shower_top_altitude, shower_length, shower_width, alt, az, impact_point, n):
    """
    N random points following a gaussian repartition in an ellipsoid. The ellipsoid comes from direction (alt,az) and goes through impact_point
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
    shower_center = [0, 0, shower_top_altitude - shower_length/2.]
    shower = gaussian_ellipsoide_alongz(shower_center, shower_length, shower_width, n)
    return shower_array_rot(shower, alt, az) + np.array(impact_point)


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


def shower_array_rot(shower_array, alt, az):
    """
    Given a series of point on the Z axis, perform a rotation of alt around Y and az around Z

    Parameters
    ----------
    shower_array: numpy array of shape (N,3) giving N points coordinates
    alt: altitude shower direction - float
    az: azimuth shower direction - float

    Returns
    -------
    Numpy array of shape (N,3) giving N points coordinates
    """
    rotated_shower_array = geo.rotation_matrix_z(az) * geo.rotation_matrix_y(pi / 2. - alt) * shower_array.T
    return np.array(rotated_shower_array.T)


def rotated_shower(shower, alt, az):
    """
    Return a rotated shower object from a shower object and a direction (alt, az)
    Parameters
    ----------
    shower: shower class object

    Returns
    -------
    copy of the given shower but rotated
    """
    rot_shower = copy(shower)
    rot_shower.particles = shower_array_rot(shower.particles, shower.alt, shower.az)
    return rot_shower



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
    shower_center = [0, 0, shower_top_altitude - shower_length/2.]
    shower = random_ellipsoide_alongz(shower_center, shower_length, shower_width, n)
    return shower_array_rot(shower, alt, az) + np.array(impact_point)


