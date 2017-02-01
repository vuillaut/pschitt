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
import math
import CameraImage as ci

DEF = -10000

#Camera sizes in meters
#LST_RADIUS = 1
#MST_RADIUS = 1
#SST_RADIUS = 0.5
LST_FOCALE = 16
MST_FOCALE = 10
SST_FOCALE = 5

'''
Camera Types
# 0 = dragon
# 1 = nectar
# 2 = flash
# 3 = sct
# 4 = astri
# 5 = dc
# 6 = gct
'''


class Telescope:
    """
    Class describing the telescope and its camera
    """
    id = 0
    def __init__(self, center, normal, camera_type):
        Telescope.id += 1
        self.id = Telescope.id
        self.center = np.array(center)
        #self.normal = np.array(list(sympy.Matrix(normal).normalized().evalf()))
        self.normal = np.array(normal)/np.linalg.norm(normal)
        self.camera_type = camera_type
        self.pixpos_filename = "data/PosPixel_{0}.txt".format(camera_type)
        self.pixel_tab = ci.read_pixel_pos(self.pixpos_filename)
        self.signal_hist = np.zeros(len(self.pixel_tab[0]))

        if(camera_type>3):
            self.type="sst"
            #self.camera_size = SST_RADIUS   ## camera size is now given by the pixels positions
            self.focale = SST_FOCALE
        elif(camera_type>0):
            self.type="mst"
            #self.camera_size = MST_RADIUS
            self.focale = MST_FOCALE
        else:
            self.type="lst"
            #self.camera_size = LST_RADIUS
            self.focale = LST_FOCALE

        self.camera_center = self.center + self.normal * self.focale
        self.camera_size = math.sqrt((self.pixel_tab[0]**2 +self.pixel_tab[0]**2).max())


    def display_info(self):
        """Just display some info about telescope and camera"""
        print('')
        print("Telescope number ", self.id)
        print("Type ", self.type)
        print("Camera type:", self.camera_type)
        print("Center ", self.center)
        print("normal ", self.normal)
        print("Focale ", self.focale)
        print("Pixel position datafile ", self.pixpos_filename)


    def pointing_object(self, point):
        """
        set the pointing direction to a point in space
        """
        self.normal = np.array(point) - self.camera_center
        self.normal = self.normal / np.sqrt((self.normal ** 2).sum())



def plane_array(point, normal):
    """
    Given a point and a the normal vector of a plane, compute the 4 parameters (a,b,c,d) of the plane equation
    a*x + b*y + c*z + d = 0
    Parameters
    ----------
    point: 3-floats Numpy array
    normal: 3-floats Numpy array

    Returns
    -------
    4-floats array
    """
    d = - normal.dot(point)
    return np.append(normal, d)


def is_point_in_plane(point, plane, eps=1e-6):
    """
    Check if a given point is in a plane
    Parameters
    ----------
    point: 3-floats Numpy array
    plane: 4-floats Numpy array giving the plane coefficients of the cartesian equation a*x+b*y+c*z = d
    eps: float - desired accuracy

    Returns
    -------
    Boolean - True if the given point is in the plane, False otherwise
    """
    #if the point is in the plane, it must check the plane equation
    if(math.fabs( (point * plane[:3]).sum() + plane[3]) < eps):
        return True
    else:
        return False


def is_point_in_camera_plane(point, telescope, eps=1e-6):
    """
    Check if a point is the camera plane of the telescope
    Parameters
    ----------
    point: 3-floats array - point to check
    telescope: telescope class
    eps: accuracy

    Returns
    -------
    Boolean - True if the point is in the camera plane, False otherwise
    """
    return is_point_in_plane(point, camera_plane(telescope))


def point_distance(point1, point2):
    """
    Compute the distance between two points in space
    Parameters
    ----------
    point1: cartesian coordinates = 3-floats array
    point2: cartesian coordinates = 3-floats array

    Returns
    -------
    distance between the two points = float
    """
    return math.sqrt(((np.array(point2)-np.array(point1))**2).sum())


def vector_norm(vector):
    """
    Compute the norm of a vector
    Parameters
    ----------
    vector: 3-floats array

    Returns
    -------
    float: norm of the vector
    """
    return np.linalg.norm(vector)


def is_point_visible(point, telescope):
    """
    Test if a point is inside the camera radius
    Parameters
    ----------
    point: 3-floats array
    telescope: telescope class

    Returns
    -------
    Boolean - True if the point is visible by the camera of the telescope
    """
    if(point_distance(point, camera_center(telescope)) < telescope.camera_size):
        return True
    else:
        return False


def site_to_camera_cartesian_old(point, telescope):
    """
    Given cartesian coordinates of a point in the site frame,
    compute the cartesian coordinates in the camera frame
    Parameters
    ----------
    point: 3-floats array
    telescope: telescope class

    Returns
    -------
    Numpy array
    """
    o = point - camera_center(telescope)
    [e1,e2,e3] = camera_base(telescope)
    # x = np.dot(o,e1)
    # y = np.dot(o,e2)
    # z = np.dot(o,e3)
    x = my_3d_dot(o, e1)
    y = my_3d_dot(o, e2)
    z = my_3d_dot(o, e3)
    return np.array([x,y,z])


def site_to_camera_cartesian(shower, telescope):
    """
    Same as before but for whole shower
    Parameters
    ----------
    shower
    telescope

    Returns
    -------

    """
    o = shower - camera_center(telescope)
    [e1, e2, e3] = camera_base(telescope)
    x = o.dot(e1)
    y = o.dot(e2)
    z = o.dot(e3)
    return np.array([x, y, z]).T



def normal_to_altaz(normal):
    """
    given a vector, return the corresponding (altitude,azimuth) pointing direction
    Parameters
    ----------
    normal: cartesian coordinates of the vector = 3-floats array

    Returns
    -------
    altitude, azimuth = angles of the vector pointing direction
    """
    alt = math.asin(normal[2])
    if normal[0]==0:
        az = math.pi/2
    else:
        az = math.atan2(normal[1],normal[0])
    return alt, az


def altaz_to_normal(alt,az):
    """
    given a pointing direction angles (altitude, azimuth),
    return a unit vector pointing in this direction
    Parameters
    ----------
    alt: altitude angle = float
    az: azimuth angle = float

    Returns
    -------
    cartesian coordinates of the unit vector pointing to (alt,az) = 3-floats array
    """
    return np.array([math.cos(alt) * math.cos(az), math.cos(alt) * math.sin(az), math.sin(alt)])


def cartesian_to_altaz(vec):
    """
    :param vec: array
    :return: array
    """
    x = vec[0]
    y = vec[1]
    z = vec[2]
    rho = math.sqrt(x**2 + y**2 + z**2)
    alt = math.asin(z/rho)
    az = math.acos(x/rho)
    if y<0:
        az = az - math.pi
    return [rho,alt,az]


def altaz_to_cartesian(vec):
    """
    :param vec: array
    :return: array
    """
    rho = vec[0]
    alt = vec[1]
    az = vec[2]
    x = rho * math.cos(alt) * math.cos(az)
    y = rho * math.cos(alt) * math.sin(az)
    z = rho * math.sin(alt)
    return [x,y,z]

def camera_base(telescope):
    """
    :param telescope: telescope class
    :return: array
    """
    # if not(np.array_equal(telescope.normal,[0,0,1])):
    #     e2 = np.cross([0,0,1],telescope.normal)
    #     e1 = np.cross(e2,telescope.normal)
    # else:
    #     e1 = [1,0,0]
    #     e2 = [0,1,0]
    #e2 = np.cross([0,0,1],telescope.normal)
    #e1 = np.cross(e2,telescope.normal)
    e2 = my_3d_cross([0,0,1],telescope.normal)
    e1 = my_3d_cross(e2,telescope.normal)
    e3 = list(telescope.normal)
    #return [e1/np.linalg.norm(e1),e2/np.linalg.norm(e2),e3/np.linalg.norm(e3)]
    return [my_normed_vec(e1),my_normed_vec(e2),my_normed_vec(e3)]


def my_3d_cross(vec1, vec2):
    x = vec1[1] * vec2[2] - vec1[2] * vec2[1]
    y = vec1[2] * vec2[0] - vec1[0] * vec2[2]
    z = vec1[0] * vec2[1] - vec1[1] * vec2[0]
    return [x,y,z]

def my_3d_norm(vec):
    return math.sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2])

def my_normed_vec(vec):
    norm = my_3d_norm(vec)
    return [vec[0]/norm, vec[1]/norm, vec[2]/norm]

def my_3d_dot(vec1, vec2):
    return vec1[0]*vec2[0] + vec1[1]*vec2[1] + vec1[2]*vec2[2]


def camera_plane(telescope):
    """
    :param telescope: telescope class
    :return: plane array
    """
    return plane_array(camera_center(telescope), telescope.normal)


def camera_center(telescope):
    """
    :param telescope: telescope class
    :return: array - position of the camera center in the site frame
    """
    return telescope.center + telescope.normal * telescope.focale


def plan_focale_image(telescope):
    """
    :param telescope: telescope class
    :return: SymPy plane
    """
    return equation_plane(telescope.center-telescope.focale*telescope.normal, telescope.normal)


def matrices_inter(point_line1, point_line2, point_plane, normal_plane):
    # type: (Numpy array, Numpy array, Numpy array, Numpy array) -> Numpy Arrays (matrices)
    """
    Given two points of a line, a point in a plane and a normal vector of the plane, return the matrices A,B
    necessary to find the intersection of the line and the plane
    Parameters
    ----------
    point_line1: 3-floats Numpy array - first point of the line
    point_line2: 3-floats Numpy array - second point of the line
    point_plane: 3-floats Numpy array - plane point
    normal_plane: 3-floats Numpy array - plane normal vector in cartesian coordinates

    Returns
    -------
    Tuple of matrices (A,B)
    """
    D = point_line2 - point_line1
    a = np.array([[0, D[2], -D[1]] , [-D[2], 0, D[0]], [normal_plane[0], normal_plane[1], normal_plane[2]]])
    b = np.array([D[2] * point_line1[1] - D[1] * point_line1[2], D[0] * point_line1[2] - D[2] * point_line1[0],
         point_plane[0] * normal_plane[0] + point_plane[1] * normal_plane[1] + point_plane[2] * normal_plane[2]])
    return a, b


def image_point_pfi_old(point, telescope):
    """
    Compute the coordinates of the image (in the image focal plane) of a given point
    Parameters
    ----------
    point: 3-floats array
    telescope: telescope class

    Returns
    -------
    3-floats Numpy array
    """
    a, b = matrices_inter(np.array(point), telescope.center, telescope.center-telescope.focale*telescope.normal, telescope.normal)
    inter = np.linalg.solve(a,b)
    return inter


def image_point_pfi(point, telescope):
    """
    Faster way to compute the coordinates of the image (in the image focal plane) of a given point
    Parameters
    ----------
    point: 3-floats array
    telescope: telescope class

    Returns
    -------
    3-floats Numpy array
    """
    e = (telescope.center - point) / np.linalg.norm(telescope.center - point)
    I = telescope.center + telescope.focale/(np.dot(telescope.normal, -e)) * e
    return I


def image_shower_pfi(shower, telescope):
    """
    Compute the image of a shower in the image focal plan as seen by a telescope
    Parameters
    ----------
    shower: numpy ndarray representing a list of points
    telescope: telescope class

    Returns
    -------
    list of points coordinates [X,Y,Z] of the shower image
    """
    e = (telescope.center - shower)
    norm = np.linalg.norm(telescope.center - shower, axis=1, keepdims=True)
    e = e / norm
    theta = e.dot(-telescope.normal)[np.newaxis].T
    I = telescope.center + telescope.focale/theta * e
    return I


def image_shower_pfo(shower, telescope):
    image_pfi = image_shower_pfi(shower, telescope)
    return image_pfi + 2.0 * telescope.focale * telescope.normal


def image_shower_pfi_old(shower, telescope):
    """
    Compute the image of a shower in the image focal plan as seen by a telescope
    Parameters
    ----------
    shower: list of points coordinates [X,Y,Z]
    telescope: telescope class

    Returns
    -------
    list of points coordinates [X,Y,Z] of the shower image
    """
    #ex = (telescope.center - shower[0]) / np.linalg.norm(telescope.center - point)
    ex = (telescope.center[0] - shower[0])
    ey = (telescope.center[1] - shower[1])
    ez = (telescope.center[2] - shower[2])
    norm = np.sqrt(ex**2 + ey**2 + ez**2)

    ex = ex/norm
    ey = ey/norm
    ez = ez/norm

    theta = -( telescope.normal[0] * ex + telescope.normal[1] * ey + telescope.normal[2] * ez)

    ix = telescope.center[0] + (telescope.focale / theta) * ex
    iy = telescope.center[1] + (telescope.focale / theta) * ey
    iz = telescope.center[2] + (telescope.focale / theta) * ez

    return ix, iy, iz


def image_shower_pfo_old(shower, telescope):
    """
    Compute the image of a shower in the object focal plane (plane of the camera) as seen by a telescope
    Parameters
    ----------
    shower: list of points coordinates [X,Y,Z]
    telescope: telescope class

    Returns
    -------
    list of points coordinates [X,Y,Z] of the shower image
    """
    ix, iy, iz = image_shower_pfi(shower, telescope)
    ix += 2.0 * telescope.focale * telescope.normal[0]
    iy += 2.0 * telescope.focale * telescope.normal[1]
    iz += 2.0 * telescope.focale * telescope.normal[2]
    return ix, iy, iz


def image_point_pfo(point, telescope):
    """
    Compute the coordinates of the image in the object focal plane) of a given point
    :param point: 3-floats array
    :param telescope: telescope class
    :return: 3-floats array
    """
    image_pfi = image_point_pfi(point, telescope)
    return image_pfi + 2.0 * telescope.focale * telescope.normal


def load_telescopes(filename, normal = [0,0,1]):
    """
    Load telescopes positions and pointing direction from data file
    :param filename: string - name of the data file
    :param normal: array - pointing direction
    :return: list of telescope classes
    """
    tels = []
    with open(filename, 'r') as f:
        read_data = f.readlines()
    for line in read_data:
        if not line[0]=='#':
            t = line.split()
            tel = Telescope([float(t[1]),float(t[2]),float(t[3])],normal,int(t[0]))
            tels.append(tel)
    print(len(tels), " telescopes loaded")
    return tels


def load_telescopes_flatfloor(filename, normal = [0,0,1]):
    """
    Load telescopes positions and pointing direction from data file. All the telescopes have the same altitude z=0
    Parameters
    ----------
    filename: string - datafile name with list of positions for the telescopes
    normal: pointing direction as a normal (to the camera plane) vector

    Returns
    -------
    list of telescope classes
    """
    tels = []
    with open(filename, 'r') as f:
        read_data = f.readlines()
    for line in read_data:
        t = line.split()
        if(float(t[2])==LST_FOCALE):
            teltype = "lst"
        elif(float(t[2])==MST_FOCALE):
            teltype = "mst"
        elif(float(t[2])==SST_FOCALE):
            teltype = "sst"
        tel = Telescope([float(t[0]),float(t[1]),0],normal,teltype)
        #for tel1 in tels:
        #    if not ((tel1.center==tel.center).all() and tel1.id==tel.id):
        tels.append(tel)
    print(len(tels), " telescopes loaded")
    return tels


def cross_cart(v1,v2):
    """
    cross product of two vectors in cartesian coordinates
    :param v1: array
    :param v2: array
    :return: array
    """
    x = v1[1]*v2[2] - v2[1]*v1[2]
    y = - v1[0]*v2[2] + v2[0]*v1[2]
    z = v1[0]*v2[1] - v2[0]*v1[1]
    return [x,y,z]


def rotation_matrix_x(theta):
    """
    Compute the rotation matrix for a rotation of an angle theta around the X axis
    :param theta: float, rotation angle
    :return: Numpy matrix
    """
    return np.matrix([[1,0,0],[0,math.cos(theta),-math.sin(theta)],[0,math.sin(theta),math.cos(theta)]])


def rotation_matrix_y(theta):
    """
    Compute the rotation matrix for a rotation of an angle theta around the Y axis
    :param theta: float, rotation angle
    :return: Numpy matrix
    """
    return np.matrix([[math.cos(theta),0,math.sin(theta)],[0,1,0],[-math.sin(theta),0,math.cos(theta)]])

def rotation_matrix_z(theta):
    """
    Compute the rotation matrix for a rotation of an angle theta around the Z axis
    :param theta: float, rotation angle
    :return: Numpy matrix
    """
    return np.matrix([[math.cos(theta),-math.sin(theta),0],[math.sin(theta),math.cos(theta),0],[0,0,1]])


def camera_ex(alt,az):
    return np.array([-math.sin(az), math.cos(az), 0])

def camera_ey(alt,az):
    return np.array([-math.sin(alt) * math.cos(az), -math.sin(alt) * math.sin(az), math.cos(alt)])

def camera_ez(alt,az):
    return np.array([math.cos(alt) * math.cos(az), math.cos(alt) * math.sin(az), math.sin(alt)])


def camera_frame_base(alt, az):
    M = np.column_stack((camera_ex(alt,az),camera_ey(alt,az),camera_ez(alt,az)))
    return np.mat(M)


def camera_frame_to_R(tel, alt, az, vector):
    V = np.matrix(vector).T
    M = camera_frame_base(alt, az)
    return (M*V).T.getA()[0] + tel.camera_center


def barycenter_in_R(tel, alt, az, cen_x, cen_y):
    return camera_frame_to_R(tel, alt, az, [cen_x, cen_y, 0])


def normal_vector_ellipse_plane(psi, alt, az):
    n = [
    math.sin(alt) * math.cos(az) * math.cos(psi) - math.sin(az) * math.sin(psi),
    math.sin(alt) * math.sin(az) * math.cos(psi) + math.cos(az) * math.sin(psi),
    -math.cos(alt) * math.cos(psi)
    ]
    return np.array(n)


def normal_vector_ellipse_plane2(psi, alt, az):
    bb = np.matrix([math.cos(psi), math.sin(psi), 0]).T
    camera_base = camera_frame_base(alt, az)
    bb2 = (camera_base * bb).T.getA()[0]
    n = np.cross(bb2,camera_ez(alt,az))
    return np.array(n)


def normal_vector_ellipse_plane3(psi, alt, az, barycenter_x, barycenter_y):
    camera_base = camera_frame_base(alt, az)
    ellipsevec = (camera_base * np.matrix([math.cos(psi), math.sin(psi), 0]).T).T.getA()[0]
    bco = (camera_base * np.matrix([-barycenter_x, -barycenter_y, 1]).T).T.getA()[0]
    n = np.cross(ellipsevec,bco)
    return n


def mask_without_cospatial_tels(filename, prec=10):
    """
    From a datafile with a list of telescopes positions, compute a mask with only non-cospatial telescopes
    save the result in a newfile  "data/UniquePositionTelescope.txt"
    Parameters
    ----------
    filename: datafile with list of positions
    prec: accuracy for the non-cospatiality
    """
    A = np.loadtxt(filename)
    B = [A[0]]
    C = [A[0][3]-1]
    unique = True
    for a in A:
        unique = True
        for b in B:
            if((np.abs(a[:3]-b[:3])<prec).all()):
                unique = False
        if unique:
            B.append(a)
            C.append(a[3]-1)
    np.savetxt("data/UniquePositionTelescope.txt",B,fmt='%3.3f',delimiter='\t')
    np.savetxt("data/MaskUnique.txt",C,fmt='%d')
    print("Number of unique telescopes : ", len(C))


def pointing_object(tel, Point):
    """
    Compute the direction (camera's normal) for a telescope to point to a given point in space
    Parameters
    ----------
    tel: telescope class
    Point: 3-float Numpy array

    Returns
    -------
    the direction as a normal vector = 3-floats numpy array
    """
    tel.normal = np.array(Point)-tel.camera_center
    tel.normal = tel.normal / np.sqrt((tel.normal**2).sum())
    return tel.normal


def pointing_object_array(alltel, Point):
    """
    Compute the direction (camera's normal) for each telescope of the array to point to a given point in space
    The new direction is set for each telescope class
    Parameters
    ----------
    alltel: list of telescopes class
    Point: 3-floats array
    """
    for tel in alltel:
        pointing_object(tel, Point)


def divergent_pointing_array(alltel, point):
    """
        Compute the direction (camera's normal) for each telescope of the array to point to a given point in space in a divergent manner
        (you probably want to point to something in the ground here)

    Parameters
    ----------
    alltel: list of telescope classes
    point: point = 3-floats array
    """
    pointing_object_array(alltel, point)
    for tel in alltel:
        tel.normal = -tel.normal


def telescopes_unicity(tel_list):
    """
    Check if all the telescopes in a list are unique
    Parameters
    ----------
    tel_list: list of telescopes classes

    Returns
    -------
    Boolean: True if all telescopes in the list are unique
    """
    bool = True
    for tel1 in tel_list:
        for tel2 in tel_list:
            if ((tel1.center == tel2.center).all() and tel1.id != tel2.id):
                print("Telescopes {0} and {1} are both at the same position {3}".format(tel1.id, tel2.id, tel1.center))
                bool = False
    return bool


def particle_cone_angle(particle_beta=1, air_index=1):
    """
    Compute the particle cone angle as a function of its energy
    Parameters
    ----------
    particle_position: 3-floats array
    particle_energy: float in eV

    Returns
    -------
    float: angle in radian
    """
    # 0.018 rad ~ 1 deg
    # theta = 1/beta*neta ?
    return 0.018


def particle_transmission_coefficient(particle_energy, particle_altitude):
    """

    Parameters
    ----------
    particle_energy: float Particle energy in eV

    Returns
    -------
    float: transmission coefficient
    """
    return 1


def is_particle_visible(particle_position, particle_direction, particle_energy, telescope):
    """
    Determine if a particle is visible by a telescope.

    Parameters
    ----------
    particle_position: 3-floats array - particle position
    particle_direction: 3-floats array - direction vector of the particle
    particle_energy: float
    telescope: telescope class

    Returns
    -------
    boolean: True if the particle is visible by the telescope
    """

    # vector particle-telescope:
    PT = telescope.camera_center - np.array(particle_position)
    PT = PT/np.linalg.norm(PT)

    # angle between particle direction and PT
    theta_pt = np.arccos(np.dot(particle_direction, PT)/np.linalg.norm(particle_direction))

    #particle emission transmission
    Tau = particle_transmission_coefficient(particle_energy, particle_position[2])

    return (theta_pt < particle_cone_angle(particle_energy)) & (np.random.rand() < Tau)


def cherenkov_ground_ellipse_parameters(particle_position, particle_direction, cherenkov_cone_angle):
    """
    compute the parameters of the ellipse created by the illumination of the ground by a cherenkov shower
    the ground is supposed to be given by z=0

    Parameters
    ----------
    particle_position: 3-floats array
    particle_direction: 3-floats array - direction vector
    cherenkov_cone_angle: float, angle in rad

    Returns
    -------

    """

    #center of the ellipse (= impact point)
    #x0 = (2. * ) / ()