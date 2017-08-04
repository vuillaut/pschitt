# Licensed under a 3-clause BSD style license - see LICENSE.rst


import numpy as np
import matplotlib.pyplot as plt
from . import geometry as geo
import math
import random
import scipy.special as ss
from copy import copy
from scipy.special import gamma as sgamma



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
        self.array = np.empty((3,self.number_of_particles))

    def linear_segment(self, shower_first_interaction, shower_bot):
        """
        Homogeneous repartition of points following a linear segment
        Parameters
        ----------
        shower_first_interaction: 1D Numpy array - position of the first interaction point
        shower_bot: 1D Numpy array - position of the bottom of the shower
        """
        self.array = linear_segment(shower_first_interaction, shower_bot, self.number_of_particles)


    def random_surface_sphere(self, shower_center, shower_radius):
        """
        Random repartition of points on a sphere surface

        Parameters
        ----------
        shower_center: 1D Numpy array
        shower_radius: float
        """
        self.array = random_surface_sphere(shower_center, shower_radius, self.number_of_particles)


    def random_surface_ellipsoide(self, shower_center, shower_length, shower_width):
        self.array = random_surface_ellipsoide(shower_center, shower_length, shower_width, self.number_of_particles)


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
        self.array = random_ellipsoide_alongz(shower_center, shower_length, shower_width, self.number_of_particles)


    def gaussian_ellipsoide_alongz(self, shower_center, shower_length, shower_width):
        """

        Parameters
        ----------
        shower_center
        shower_length
        shower_width
        """
        self.array = gaussian_ellipsoide_alongz(shower_center, shower_length, shower_width, self.number_of_particles)


    def gaussian_ellipsoide(self, shower_top_altitude, shower_length, shower_width):
        """

        Parameters
        ----------
        shower_top_altitude
        shower_length
        shower_width
        """
        self.array = gaussian_ellipsoide(shower_top_altitude, shower_length, shower_width, \
                                         self.alt, self.az, self.impact_point, self.number_of_particles)


    def shower_rot(self, alt, az):
        """
        Rotate the shower
        Parameters
        ----------
        alt: float
        az: float
        """
        self.array = shower_array_rot(self.array, alt, az)


    def plot3d(self):
        """
        Make a 3d plot
        """
        plot3d(self.array)


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
        self.array = random_ellipsoide(shower_top_altitude, shower_length, shower_width, \
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
    theta = math.pi * np.random.random_sample(n)
    phi = 2. * math.pi * np.random.random_sample(n)
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
    theta,phi = math.pi * np.random.random_sample(n), 2. * math.pi * np.random.random_sample(n)
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
    theta, phi = math.pi * np.random.random_sample(n), 2. * math.pi * np.random.random_sample(n)
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
    rotated_shower_array = geo.rotation_matrix_z(az) * geo.rotation_matrix_y(math.pi / 2. - alt) * shower_array.T
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
    rot_shower.array = shower_array_rot(shower.array, shower.alt, shower.az)
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


def gaisser_hillas_reduced(x, eps):
    """
    Reduced distribution of Gaisser-Hillas
    Parameters
    ----------
    x: float or 1D numpy array - reduced atmospheric depth
    eps: float - reduced parameter

    Returns
    -------
    distribution - same type as x
    """
    assert np.all(x>=-1) and eps>0
    return np.nan_to_num(np.power(1+x, eps) * np.exp(-eps *x))


def gaisser_hillas(X, Nmax, Xmax, X1, lam):
    """
    Gaisser-Hillas distribution
    Parameters
    ----------
    X: float or 1D array - atmospheric depth
    Nmax: Number of particles at max(N)
    Xmax: Atmospheric length at max(N)
    X1: Atmospheric depth
    lam: Mean atmospheric depth (usually 70g/cm2)

    Returns
    -------
    Gaisser-Hillas distribution - same type as X
    """
    assert np.all(X>=X1) and Xmax>=X1 and Nmax > 0 and lam >0
    W = Xmax - X1
    x = (X-Xmax)/W
    eps = W / lam
    return Nmax * Ngh_s(x, eps)


def gaisser_hillas_reduced_integral(eps, x_min, x_max):
    """
    Analytical integral of the reduced gaisser-hillas function between x_min and x_max
    Parameters
    ----------
    eps: reduced parameter
    x_min: reduced atmospheric depth - float or 1D Numpy array
    x_max: reduced atmospheric depth - float or 1D Numpy array
    Returns
    -------
    float or 1D Numpy array
    """
    assert np.all(x_min >= -1) and np.all(x_max >= -1)
    return np.exp(eps) * eps**(-(eps+1)) * ss.gamma(1+eps) * (ss.gammaincc(1+eps, eps*(1+x_min)) - ss.gammaincc(1+eps, eps*(1+x_max)))


def gaisser_hillas_integral(X, Nmax, Xmax, X1, lam):
    """
    Analytical integral of the gaisser-hillas function between x_min and x_max
    Parameters
    ----------
    X: atmospheric depth - float or 1D Numpy array
    Nmax: float
    Xmax float
    X1: float
    lam: float

    Returns
    -------
    type(X)
    """
    assert np.all(X >= X1) and Xmax > X1
    W = Xmax - X1
    eps = W / lam
    return W * Nmax * gaisser_hillas_reduced_integral(eps, -1, (X-Xmax)/W)


def gaisser_hillas_integral_0_inf(Nmax, Xmax, X1, lam):
    """
    Analytical integral of the Gaisser-Hillas distribution between 0 and +inf
    Parameters
    ----------
    Nmax: float
    Xmax: float
    X1: float
    lam: float

    Returns
    -------
    float
    """
    W = Xmax - X1
    eps = W / lam
    return Nmax * W * np.exp(eps) * eps**(-1-eps) * ss.gamma(1+eps)


def gaisser_hillas_integral_inverse(Y, Nmax, Xmax, X1, lam):
    """

    Parameters
    ----------
    Y: float or 1D Numpy array
    Nmax: float
    Xmax: float
    X1: float
    lam: float

    Returns
    -------
    type(Y)
    """
    W = Xmax - X1
    eps = W / lam
    c1 = ss.gamma(1 + eps)
    c2 = Nmax * W * np.exp(eps) * eps**(-1-eps)
    return Xmax + W * (- 1 + ss.gammainccinv(1+eps, (c1 - Y/c2)/ss.gamma(1+eps)) / eps)


def Normed_Gaisser_Hillas(X1, Xmax, X):
    lamb = 70.
    A = Xmax - X1
    return np.power((X - X1) / A, A / lamb) * np.exp((Xmax - X) / lamb)


def NKG(X1, X, r):
    rm = 118.
    t = (X - X1) / 36.7
    tmax = 1.7 + 0.76 * 4.0915
    s = 2 * t / (t + tmax)

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
    # def gaisser_hillas_shower(shower_top_altitude, shower_length, alt, az, impact_point, Npart):
    """

    Parameters
    ----------
    shower_top_altitude
    shower_length
    alt
    az
    impact_point
    Npart

    Returns
    -------

    """
    shower = np.empty([Npart, 3])
    N = []
    impact_point = np.array(impact_point)
    Xinterp = []
    # dist = []
    # radius = []
    # x = []
    # y = []
    # z = []
    dist = np.empty(Npart)
    radius = np.empty(Npart)
    x = np.empty(Npart)
    y = np.empty(Npart)
    z = np.empty(Npart)

    Nmax = 1
    Xmax = 600
    X1 = 0
    lam = 70

    f = np.arange(1000)
    r = np.arange(1, 350., 10)


    # for j in f: ### why 1000 ? -> param
    #     N.append(Normed_Gaisser_Hillas(0., 600, j))
    #     #N[j] = Gaisser_Hillas(Npart, 0., 600, j)
    # Nsum = np.cumsum(N)

    # N = gaisser_hillas(f, Nmax, Xmax, X1, lam)
    Nsum = gaisser_hillas_integral(f, Nmax, Xmax, X1, lam)



    for k in range(Npart):
        D = []
        yval = random.uniform(0, max(Nsum))
        Xinterp.append(np.interp(yval, Nsum, f))
        for m in range(35):
            D.append(NKG(0, np.interp(yval, Nsum, f), r[m]))

        # Dsum = np.cumsum(D) ### not used

        yval2 = random.uniform(min(D), max(D))

        #dist.append(gcm2distance(np.interp(yval, Nsum, f), math.pi / 2. - alt))
        dist[k] = gcm2distance(np.interp(yval, Nsum, f), math.pi / 2. - alt)

        #radius.append(np.interp(yval2, r, D))
        radius[k] = np.interp(yval2, r, D)

        # x.append(radius[k] * np.cos(random.uniform(0, 2 * math.pi)))
        # y.append(radius[k] * np.sin(random.uniform(0, 2 * math.pi)))
        # z.append(dist[k])
        # x[k] = radius[k] * math.cos(random.uniform(0, 2 * math.pi))
        # y[k] = radius[k] * math.sin(random.uniform(0, 2 * math.pi))
        # z[k] = dist[k]
        ### are you sure you don't need the same angle for x and y ???

    angles = np.random.uniform(0, 2. * math.pi, Npart)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    z = dist

    for i in range(Npart):
        #shower.append([x[i], y[i], z[i]])
        shower[i] = [x[i], y[i], z[i]]

    return shower
    #return shower_array_rot(shower, alt, az, impact_point)


def radial_distribution(r, R):
    """
    radial distribution of particles in a showe
    Parameters
    ----------
    r: float or 1D Numpy array
    R: float

    Returns
    -------
    type(r)
    """
    return 2 * r * R ** 2 / (r ** 2 + R ** 2) ** 2


def radial_distribution_integral(r_min, r_max, R):
    """
    Integral of radial_distribution between r_min and r_max
    Parameters
    ----------
    r_min: float or 1D numpy array
    r_max: float or 1D numpy array
    R: float

    Returns
    -------
    type(r)
    """
    return 1/(1+(x_min/R)**2) - 1/(1+(x_max/R)**2)


def radial_distribution_cumulative(r, R):
    """
    Cumulative of the radial distribution
    Parameters
    ----------
    r: float or 1D numpy array
    R: float

    Returns
    -------
    type(r)
    """
    return f_integral(0, x, R)


def radial_distribution_cumulative_inverse(y, R):
    """

    Parameters
    ----------
    y: float or 1D numpy array
    R: float

    Returns
    -------
    type(y)
    """
    return R * np.sqrt(y/(1-y))


def gaisser_hillas_shower(shower_top_altitude, shower_length, alt, az, impact_point, Npart):
    """

    Parameters
    ----------
    shower_top_altitude
    shower_length
    alt
    az
    impact_point
    Npart

    Returns
    -------

    """
    shower = np.empty([Npart, 3])
    N = []
    impact_point = np.array(impact_point)
    Xinterp = []
    # dist = []
    # radius = []
    # x = []
    # y = []
    # z = []
    dist = np.empty(Npart)
    radius = np.empty(Npart)
    x = np.empty(Npart)
    y = np.empty(Npart)
    z = np.empty(Npart)

    Nmax = 1
    Xmax = 600
    X1 = 0
    lam = 70

    f = np.arange(1000)
    r = np.arange(1, 350., 10)

    # for j in f: ### why 1000 ? -> param
    #     N.append(Normed_Gaisser_Hillas(0., 600, j))
    #     #N[j] = Gaisser_Hillas(Npart, 0., 600, j)
    # Nsum = np.cumsum(N)

    # N = gaisser_hillas(f, Nmax, Xmax, X1, lam)
    Nsum = gaisser_hillas_integral(f, Nmax, Xmax, X1, lam)

    Y = np.random.uniform(0, 1, Npart)

    # max_ngh = gaisser_hillas_integral(f.max(), Nmax, Xmax, X1, lam)
    max_ngh = gaisser_hillas_integral_0_inf(Nmax, Xmax, X1, lam)
    NZ = gaisser_hillas_integral_inverse(Y * max_ngh, Nmax, Xmax, X1, lam)

    R = 30
    Y = np.random.uniform(0, 1, Npart)
    NR = radial_distribution_cumulative_inverse(Y, R)

    for k in range(Npart):
        dist[k] = gcm2distance(NZ[k], math.pi / 2. - alt)

    angles = np.random.uniform(0, 2. * math.pi, Npart)
    x = NR * np.cos(angles)
    y = NR * np.sin(angles)
    z = dist

    # shower = np.array([x,y,z])
    shower = np.empty([Npart,3])
    for i in range(Npart):
        shower[i] = [x[i], y[i], z[i]]

    return shower
    # return shower_array_rot(shower, alt, az) + np.array(impact_point)


def gil_distribution(ep, a, t, x1):
    """
    ## TO DO : comment / add names
    ## log or log10 ?
    ## t or X1 as parameter ?
    ## t can be computed in external function
    ## Good ref ?

    Greisen-Iljina-Linsley (GIL) parametrisation for hadronic showers longitudinal profile
    ref: Catalano O. et al. Proc. of 27th ICRC, Hamburg (Germany) p.498, 2001

    Parameters
    ----------
    ep: float - primary particle energy [TeV]
    a: float - mass of the primary
    t: float or 1D Numpy array
    x1: float - first interaction depth

    Returns
    -------

    """
    ec = 0.081 # critical energy [TeV]
    X0 = 36.7 # interaction depth [g/cm2]
    el = 1.45 # normalisation energy [TeV]

    s = shower_age(X, x1, ep, a)
    n = ep/el * np.exp(t * (1 - 2 * np.log(s)) - tmax)
    return n


def shower_age(x, x1, ep, a):
    """
    Compute the shower age as defined in GIL parameterisation
    ref: Catalano O. et al. Proc. of 27th ICRC, Hamburg (Germany) p.498, 2001, 2001

    Parameters
    ----------
    x: float or 1D Numpy array - interaction depth
    x1: float - first interaction depth
    ep: float - energy of the primary
    a: float - atomic mass of the primary

    Returns
    -------
    shower age s - X type (float or 1D Numpy array)
    """
    x0 = 36.7  # interaction depth [g/cm2]
    ec = 0.081  # critical energy [TeV]

    t = (x - x1)/ x0
    tmax = 1.7 + 0.76 * (np.log(ep/ec) - np.log(a))

    s = 2 * t / (t + tmax)
    return s


def nkg_distribution(X, x1, r, ep, a):
    """

    Parameters
    ----------
    X
    x1
    r: distance to shower axis
    ep: float - emergy of the primary
    a: float - atomic mass of the primary

    Returns
    -------

    """
    rm = 118. ### why ?
    rm = moliere_radius(x0, z)
    s = shower_age(x, x1, ep, a)

    Cs = sgamma(4.5 - s) / (math.pi * 2 * rm * rm * sgamma(s) * sgamma(4.5 - 2 * s))

    ## Ns ?
    G = 1e6 * Cs * np.power(1 + r/rm, s-4.5) * np.power(r/rm, s - 2)
    return G


def radial_exp_distribution(N0, rm):
    """
    Gives an exponantially decreasing distribution N(r) = N0 * exp(-beta * r)
    with beta = ln(10)/rm so that 90% of the particles are contained in the molière radius rm

    Parameters
    ----------
    N0: Total number of particles in the shower
    rm: Molière radius [m]

    Returns
    -------
    1D Numpy array of length N0. Gives a distribution of radius following an exponantial decrease
    """
    assert N0 > 0

    ln10 = np.log(10)

    return N0 * rm / ln10 * (1 - np.exp(-ln10 * r / rm))



def moliere_radius_approx(x0, z):
    """
    Molière radius = radius containing on average 90% of the shower's energy deposition
    Approximation given by [ref]

    Parameters
    ----------
    x0: float - radiation length
    z: float - atomic number

    Returns
    -------
    moliere radius - float
    """
    rm = 0.0265 * x0  * (z + 1.2)

    return rm



def longo_distribution(ep, ):

    ec = 0.086 # electrons critical energy [TeV]
    y = ep/ec # normalised energy
    lny = np.log(y)
    beta = 1.0 / (1.53 + 0.01 * lny)
    alpha = beta * (2.16 + 0.99 * lny)
    t = radiation_length(x, 0)

    n = E0 * beta * np.power(beta * t, alpha - 1) * np.exp(-beta * t) / sgamma(alpha)


def radiation_length(x, x1):
    """
    Radiation length

    Parameters
    ----------
    x: interaction depth - float or 1D Numpy array
    x1: first interaction depth [g/cm2]

    Returns
    -------
    radiation length t - same type as x
    """
    # Radiation length via Bremsstrahlung of electrons in the air [g/cm2]:
    x0 = 36.7
    t = (x - x1)/x0
    return t