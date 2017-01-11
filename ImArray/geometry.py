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
import sympy
from sympy import Point3D, Plane, Line3D
from sympy.physics.vector import *
from math import *


DEF = -10000

#Camera sizes in meters
LST_RADIUS = 1
MST_RADIUS = 1
SST_RADIUS = 0.5
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
    def __init__(self, center, normale, camera_type):
        Telescope.id += 1
        self.id = Telescope.id
        self.center = np.array(center)
        #self.normale = np.array(list(sympy.Matrix(normale).normalized().evalf()))
        self.normale = np.array(normale)/np.linalg.norm(normale)
        self.camera_type = camera_type
        self.pixpos_filename = "data/PosPixel_{0}.txt".format(camera_type)
        if(camera_type>3):
            self.type="sst"
            self.camera_size = SST_RADIUS
            self.focale = SST_FOCALE
        elif(camera_type>0):
            self.type="mst"
            self.camera_size = MST_RADIUS
            self.focale = MST_FOCALE
        else:
            self.type="lst"
            self.camera_size = LST_RADIUS
            self.focale = LST_FOCALE
        self.camera_center = self.center + self.normale * self.focale

    def display_info(self):
        """Just display some info about telescope and camera"""
        print('')
        print("Telescope number ", self.id)
        print("Type ", self.type)
        print("Camera type:", self.camera_type)
        print("Center ", self.center)
        print("Normale ", self.normale)
        print("Focale ", self.focale)



def plane_array(point, normale):
    d = - (np.array(point)*np.array(normale)).sum()
    return np.append(normale, d)


def is_point_in_plane(point, plane, eps=1e-6):
    """
    Check if a given point is in a plane
    :param point: array
    :param plane: array
    :param eps: allowed precision
    :return: boolean
    """
    #if the point is in the plane, it must check the plane equation
    if(fabs( (point * plane[:3]).sum() + plane[3]) < eps):
        return True
    else:
        return False

def is_point_in_camera_plane(point, telescope, eps=1e-6):
    """
    Check if a point is the camera plane of the telescope
    :param point: array
    :param telescope: class telescope
    :param eps: float, precision
    :return: boolean
    """
    return is_point_in_plane(point, camera_plane(telescope))

def point_distance(point1, point2):
    """
    Compute the distance between two points in space
    :param point1: array
    :param point2: array
    :return: float, distance
    """
    return sqrt(((np.array(point2)-np.array(point1))**2).sum())

def is_point_visible(point, telescope):
    """
    test if a point is inside the camera radius
    :param point: array
    :param telescope: telescope class
    :return: float
    """
    if(point_distance(point, camera_center(telescope)) < telescope.camera_size):
        return True
    else:
        return False


def altaz_coordinates_changes(alt_angle, az_angle, origine, point):
    """
    Change of coordinates from one frame to another
    :param alt_angle: float, altitude angle
    :param az_angle: float, azimtuh angle
    :param origine: 3D point
    :param point: 3D point
    :return: 3D point
    """
    R = ReferenceFrame('R')
    N = ReferenceFrame('N')
    N1 = R.orientnew('N1', 'Axis', [az_angle,R.z])
    N.orient(N1,'Axis', [alt_angle, R.x])
    return Point3D(R.dcm(N)*sympy.Matrix(point) + sympy.Matrix(origine), evaluate=False)


def site_to_camera_cartesian(point, telescope):
    """
    Given cartesian coordinates of a point in the site frame,
    compute the cartesian coordinates in the camera frame
    :param point: array
    :param telescope: telescope class
    :return: Numpy array
    """
    o = point - camera_center(telescope)
    [e1,e2,e3] = camera_base(telescope)
    x = np.vdot(o,e1)
    y = np.vdot(o,e2)
    z = np.vdot(o,e3)
    return np.array([x,y,z])

def normale_to_altaz(normale):
    """
    :param normale: array
    :return: tuple of 2 floats : Alt,Az
    """
    alt = asin(normale[2])
    if normale[0]==0:
        az = pi/2
    else:
        az = atan2(normale[1],normale[0])
    return (alt,az)

def altaz_to_normal(alt,az):
    """
    :param alt: float
    :param az: float
    :return: array
    """
    return [cos(alt)*cos(az),cos(alt)*sin(az),sin(alt)]

def cartesian_to_altaz(vec):
    """
    :param vec: array
    :return: array
    """
    x = vec[0]
    y = vec[1]
    z = vec[2]
    rho = sqrt(x**2 + y**2 + z**2)
    alt = asin(z/rho)
    az = acos(x/rho)
    if y<0:
        az = az - pi
    return [rho,alt,az]


def altaz_to_cartesian(vec):
    """
    :param vec: array
    :return: array
    """
    rho = vec[0]
    alt = vec[1]
    az = vec[2]
    x = rho*cos(alt)*cos(az)
    y = rho*cos(alt)*sin(az)
    z = rho*sin(alt)
    return [x,y,z]

def camera_base(telescope):
    """
    :param telescope: telescope class
    :return: array
    """
    if not(np.array_equal(telescope.normale,[0,0,1])):
        e2 = np.cross([0,0,1],telescope.normale)
        e1 = np.cross(e2,telescope.normale)
    else:
        e1 = [1,0,0]
        e2 = [0,1,0]
    e3 = list(telescope.normale)
    return [e1/np.linalg.norm(e1),e2/np.linalg.norm(e2),e3/np.linalg.norm(e3)]


def camera_plane(telescope):
    """
    :param telescope: telescope class
    :return: plane array
    """
    return plane_array(camera_center(telescope), telescope.normale)


def camera_center(telescope):
    """
    :param telescope: telescope class
    :return: array - position of the camera center in the site frame
    """
    return telescope.center + telescope.normale * telescope.focale


def equation_plane(point, normale):
    """
    :param point: array
    :param normale: array
    :return: SymPy Plane - cartesian equation of a plane
    """
    return Plane(Point3D(point, evaluate=False), normal_vector=list(normale), evaluate=False)


def plan_focale_image(telescope):
    """
    :param telescope: telescope class
    :return: SymPy plane
    """
    return equation_plane(telescope.center-telescope.focale*telescope.normale, telescope.normale)


def photon_line(point_objet, telescope):
    """
    :param point_objet: SymPy point or array
    :param telescope: telescope class
    :return: SymPy line
    """
    return Line3D(point_objet, telescope.center)


def matrices_inter(point_droite1, point_droite2, point_plan, normale_plan):
    """
    :param point_droite1:
    :param point_droite2:
    :param point_plan:
    :param normale_plan:
    :return:
    """
    D = point_droite2 - point_droite1
    a = [[0,D[2],-D[1]] , [-D[2],0,D[0]], [normale_plan[0],normale_plan[1],normale_plan[2]]]
    #a = [[0,-D[2],normale_plan[0]], [D[2],0,normale_plan[1]], [-D[1],D[0],normale_plan[2]]]
    b = [D[2]*point_droite1[1] - D[1]*point_droite1[2], D[0]*point_droite1[2] - D[2]*point_droite1[0], point_plan[0]*normale_plan[0]+point_plan[1]*normale_plan[1]+point_plan[2]*normale_plan[2]]
    return a,b


def image_point_pfi(point, telescope):
    """
    Compute the coordinates of the image (in the image focal plane) of a given point
    :param point: SymPy point or array
    :param telescope: telescope class
    :return: SymPy poiny
    """
    a,b = matrices_inter(np.array(point), telescope.center, telescope.center-telescope.focale*telescope.normale, telescope.normale)
    inter = np.linalg.solve(a,b)
    return inter


def image_point_pfo(point, telescope):
    """
    Compute the coordinates of the image in the object focal plane) of a given point
    :param point: SymPy point or array
    :param telescope: telescope class
    :return: SymPy poiny
    """
    image_pfi = image_point_pfi(point, telescope)
    return image_pfi + 2*telescope.focale*telescope.normale


def image_vec_pfi(point1, point2, telescope):
    """
    :param point1: SymPy point or array
    :param point2: SymPy point or array
    :param telescope: telescope class
    :return: SymPy line
    """
    im1 = image_point_pfi(point1, telescope)
    im2 = image_point_pfi(point2, telescope)
    return Line3D(im1,im2)


def image_vec_pfo(point1, point2, telescope):
    """
    :param point1: SymPy point or array
    :param point2: SymPy point or array
    :param telescope: telescope class
    :return: SymPy line
    """
    im1 = image_point_pfo(point1, telescope)
    im2 = image_point_pfo(point2, telescope)
    return Line3D(im1,im2)


def impact_point(shower_top, shower_bot, tel1, tel2):
    """
    :param shower_top: SymPy point or array
    :param shower_bot: SymPy point or array
    :param tel1: telescope class
    :param tel2: telescope class
    :return: SymPy line
    """
    line1 = image_vec_pfo(shower_top, shower_bot, tel1)
    line2 = image_vec_pfo(shower_top, shower_bot, tel2)
    return line1.intersection(line2)


def real_impact_point(shower_top, shower_bot):
    """
    compute intersection of shower true direction and plane z=0
    :param shower_top: SymPy point or array
    :param shower_bot: SymPy point or array
    :return: SymPy point
    """
    shower_dir = Line3D(shower_top, shower_bot)
    inter = shower_dir.intersection()


def load_telescopes(filename, normale = [0,0,1]):
    """
    Load telescopes positions and pointing direction from data file
    :param filename: string - name of the data file
    :param normale: array - pointing direction
    :return: list of telescope classes
    """
    tels = []
    with open(filename, 'r') as f:
        read_data = f.readlines()
    for line in read_data:
        if not line[0]=='#':
            t = line.split()
            tel = Telescope([float(t[1]),float(t[2]),float(t[3])],normale,int(t[0]))
            tels.append(tel)
    print(len(tels), " telescopes loaded")
    return tels


def load_telescopes_flatfloor(filename, normale = [0,0,1]):
    """
    Load telescopes positions and pointing direction from data file. All the telescopes have the same altitude z=0
    :param filename: string - data file
    :param normale: array - pointing direction
    :return: list of telescope classes
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
        tel = Telescope([float(t[0]),float(t[1]),0],normale,teltype)
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
    return np.matrix([[1,0,0],[0,cos(theta),-sin(theta)],[0,sin(theta),cos(theta)]])


def rotation_matrix_y(theta):
    """
    Compute the rotation matrix for a rotation of an angle theta around the Y axis
    :param theta: float, rotation angle
    :return: Numpy matrix
    """
    return np.matrix([[cos(theta),0,sin(theta)],[0,1,0],[-sin(theta),0,cos(theta)]])

def rotation_matrix_z(theta):
    """
    Compute the rotation matrix for a rotation of an angle theta around the Z axis
    :param theta: float, rotation angle
    :return: Numpy matrix
    """
    return np.matrix([[cos(theta),-sin(theta),0],[sin(theta),cos(theta),0],[0,0,1]])


def camera_ex(alt,az):
    return np.array([-sin(az), cos(az), 0])

def camera_ey(alt,az):
    return np.array([-sin(alt) * cos(az), -sin(alt) * sin(az), cos(alt)])

def camera_ez(alt,az):
    return np.array([cos(alt) * cos(az), cos(alt) * sin(az), sin(alt)])


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
    sin(alt) * cos(az) * cos(psi) - sin(az) * sin(psi),
    sin(alt) * sin(az) * cos(psi) + cos(az) * sin(psi),
    -cos(alt) * cos(psi)
    ]
    return np.array(n)


def normal_vector_ellipse_plane2(psi, alt, az):
    bb = np.matrix([cos(psi), sin(psi), 0]).T
    camera_base = camera_frame_base(alt, az)
    bb2 = (camera_base * bb).T.getA()[0]
    n = np.cross(bb2,camera_ez(alt,az))
    return np.array(n)


def normal_vector_ellipse_plane3(psi, alt, az, barycenter_x, barycenter_y):
    camera_base = camera_frame_base(alt, az)
    ellipsevec = (camera_base * np.matrix([cos(psi), sin(psi), 0]).T).T.getA()[0]
    bco = (camera_base * np.matrix([-barycenter_x, -barycenter_y, 1]).T).T.getA()[0]
    n = np.cross(ellipsevec,bco)
    return n

def mask_without_cospatial_tels(filename, prec=10):
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
