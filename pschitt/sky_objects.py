# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from . import geometry as geo
from . import emission as em
import math
from math import pi
from copy import copy

# MODIFICATION 17/4/18 #

"""
Constants
----------
Ec_eV: Electron critical energy [eV]
Ec_J: Electron critical energy [J]
Xo: Radiation length of air [g/cm^2]
H: Scale height of the atmosphere [m]
g: Acceleration due to gravity at sea level [m/s^2]
Po: Atmospheric pressure at sea level [Pa]
alpha: Electromagnetic coupling constant [none]
c: Speed of light in a vacuum [m/s]
m_e: Mass of an electron [kg]
Es_J: ??? [J] 
po = Density of air at sea level [kg/m^3]
po_g = Density of air at sea level [g/cm^3]
Rm_0 = Moliere radius at sea level [m]
binfactor = Scaling factor for sky bins [none]
binsize = Size of sky bins [g/cm^2]
"""
Ec_eV = 8.6e7 
Ec_J = 8.6e7*(1.602e-19) 
Xo = 37.2 
H = 7250 # !Needs refining!
g = 9.81 # !Needs refining!
Po = 101325 # !Needs refining!
alpha = 137**(-1)
c = 2.99792458e8
m_e = 9.11e-31 
Es_J = m_e*(c**2)*(math.sqrt(4*math.pi*alpha))
po = 1.225 # !Needs refining! 
po_g = po/1000
Rm_0 = 54.6575699745 
binfactor = (1+(9/7))/2
binsize = binfactor*Xo

# END OF MODIFICATION 17/4/18 #

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
        self.height_of_first_interaction = 0 # Added 18/4/18
        self.number_of_particles = 10
        self.particles = np.empty((3,self.number_of_particles))
        self.particles_angular_emission_profile = em.angular_profile_constant
        self.particles_angular_emission_profile_kwargs = {'c':1}

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

# MODIFICATION 17,21/4/18 #

    def ShowerMax_depth(self):
        """
        Parameters
        ----------
        self.energy_primary
        """
        self.depth_shower_max = ShowerMax_depth(self.energy_primary)
        return self.depth_shower_max
        
    def showermax(self):
        """
        Parameters
        ----------
        self.energy_primary
        """
        self.height_shower_max = showermax(self.energy_primary)
        return self.height_shower_max

    def moliere_radius_ShMax(self):
        """
        Parameters
        ----------
        self.energy_primary
        """
        self.RMol_ShMax = moliere_radius_ShMax(self.energy_primary)
        return self.RMol_ShMax

    def particles_in_bins(self):  #21/4/18
        """
        Parameters
        ----------
        self.height_of_first_interaction
        self.energy_primary
        """
        self.number_in_bins = particles_in_bins(self.height_of_first_interaction, self.energy_primary)
        return self.number_in_bins

# END OF MODIFICATION 17,21/4/18 #

# MODIFICATION 16,18,19/4/18 #

    def scaled_ellipsoide_alongz(self):
        """
        Parameters
        ----------
        shower_centre
        RMol_ShMax
        """
        self.particles = scaled_ellipsoide_alongz(self.energy_primary, self.height_of_first_interaction)

    def scaled_ellipsoide(self):
        """
        Parameters
        ----------
        ShMax
        """
        self.particles = scaled_ellipsoide(self.energy_primary, self.height_of_first_interaction, self.alt, self.az, self.impact_point)

# END OF MODIFICATION 16,18,19/4/18 #

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

    def set_emission_profile(self, emission_profile, **kwargs):
        self.emission_profile = emission_profile(**kwargs)



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

# MODIFICATION 17,21/4/18 #

def ShowerMax_depth(E):	
    """
    Calculates the atmospheric depth at which the shower maximum occurs given the primary photon energy.

    Parameters
    ----------
    E: energy of the primary photon [eV] - int
    Ec_J: Electron critical energy [J] - float
    binsize: Size of sky bins [g/cm^2] - float

    Returns
    ----------
    depthmax: depth of maximum [g/cm^2] - float
    """
    Einp_J = E*1.602e-19 # Convert input energy into joules 
    depthmax = ((math.log(Einp_J/Ec_J))*binsize)/(math.log(2)) # Changed Xo to binsize 25/4/18
    return depthmax

def showermax(E):
    """
    Calculates height of shower maximum above sea level.
    
    Parameters
    ----------
    E: energy of the primary photon [eV] - float
    H: scale height of the atmosphere [m] - int
    g: Acceleration due to gravity at sea level [m/s^2] - float
    Po: Air pressure at sea level [Pa] - int

    Returns
    ----------
    ShMax: height of shower maximum [m asl] - float
    """
    ShMax = (-H*math.log((ShowerMax_depth(E)*10*g)/(Po)))
    return ShMax

def moliere_radius_ShMax(E):
    """
    Calculates the moliere radius at height of shower maximum.
    
    Parameters
    ----------
    E: Energy of the primary photon [eV] - int
    po_g: Density of air at sea level [g/cm^3] - float 
    H: Scale height of the atmosphere [m] - int
    Xo: Radiation length of air [g/cm^2] - float
    Es_J: ??? [J] - float
    Ec_J: Electron critical energy [J] - float
    
    Returns
    ----------
    R_m: Moliere radius at height of maximum [m]
    """
    p = (po_g*math.exp(-showermax(E)/H))
    R_m = (Xo*Es_J)/(p*Ec_J)
    return R_m

def particles_in_bins(h_init, E):
    """
    Slices shower into bins of size 'binsize' and calculates number of electrons and photons in each bin.
    
    Parameters
    ----------
    h_init: Height of first photon interaction [m] - float
    depth_shower_max: Atmospheric depth of shower maximum [g/cm^2] - float
    
    Returns
    ----------
    num: Numpy array of electron number and photon number in each bin [none] - floats
    N_tot: The total number of particles in the shower to be simulated - int
    """
    depth_first_interaction = ((Po*math.exp(-h_init/H))/g)/10
    n = int(math.ceil((ShowerMax_depth(E)-depth_first_interaction)/binsize))
    print('Number of sky bins: ', n)
    def propagate():
        ne = [0]
        nph =[1]
        counter = 0
        while True:
            ne.append(ne[-1]+(2*nph[-1]))
            nph.append(ne[-2])
            counter += 1
            yield [ne[:], nph[:]]
    prop = propagate()

    for i, particles in enumerate(prop):
        if (i+1)==n: 
            #print i, particles
            num = np.array(particles).T
            break
    N_tot = int(sum(i for [i,j] in num))
    print('Particles in n bins: ', num)
    print('Total number of electrons: ', N_tot) 
    return num, N_tot

# END OF MODIFICATION 17,21/4/18 #

# MODIFICATION 16,18,19,23/4/18 #

def scaled_ellipsoide_alongz(E, h_init): # Name change 19/4/18
    """
    Lateral distribution of particles described by a gaussian distribution where 90% lie within one moliere 	   radius at sea level.
    Longitudinal distribution described by a laplacian distribution distribution folded about shower maximum and scaled to the difference between the height of first interaction and shower maximum. 

    Parameters
    ----------
    E: Energy of Primary Photon [eV] - float
    h_init: Height of first interaction [m] - float
    n: Number of particles in shower [none] - int

    Returns
    -------
    Numpy array (3,n) - positions of particles in shower
    """
    # old scale for x and y: scale=((2*RMol_ShMax)/1.64485362692)
    ShMax = showermax(E) # Added 18/4/18
    shower_center = [0, 0, ShMax] # Added 18/4/18
    RMol_ShMax = moliere_radius_ShMax(E) # Added 17/4/18
    n = particles_in_bins(h_init,E)[1] # Added 23/4/18
    x = np.random.normal(loc=shower_center[0], scale=((2*Rm_0)/1.64485362692), size=n) # Changed 19/4/18
    y = np.random.normal(loc=shower_center[1], scale=((2*Rm_0)/1.64485362692), size=n) # Changed 19/4/18
    z1 = np.random.laplace(loc=shower_center[2], scale=((h_init-ShMax)/4), size=n) # Changed 19/4/18
    z=[]   
    for i in z1:
        if i-ShMax >=0:
                z.append(i)
        else: 
                z.append(i+(2*abs(i-ShMax))) # Added 19/4/18
    return np.array([x, y, z]).T

def scaled_ellipsoide(E, h_init, alt, az, impact_point): # Name change 19/4/18
    """
    n random points following gaussian lateral and laplacian longitudinal distributions. Ellipsoid originates 	    from direction (alt, az) and goes through impact point.  

    Parameters
    ----------	
    E: Energy of Primary Photon [eV] - float
    h_init: Height of first interaction [m] - float
    alt: altitude angle of shower [deg] - float
    az: azimuthal angle of shower [deg] - float 
    impact_point: point where shower axis intersects the ground [none] - np.array
    n: number of particles in shower [none] - int

    Returns
    -------
    List of points in the shower (3-floats arrays)
    """
    shower = scaled_ellipsoide_alongz(E, h_init) 
    return shower_array_rot(shower, alt, az) + np.array(impact_point)

# END MODIFICATION 16,18,19,23/4/18#

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

"""
## THE FOLLOWING IS FOR TEST PURPOSES ONLY 18,19/4/18 ##

shower = shower()
shower.energy_primary = 5e10
#shower.number_of_particles = int(1e4)
shower.height_of_first_interaction = 25000
shower.scaled_ellipsoide()
shower.particles_in_bins()
Q = shower.particles
print(Q)
"""

