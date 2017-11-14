# Licensed under a 3-clause BSD style license - see LICENSE.rst


import numpy as np
from . import geometry as geo
from math import *



def hillas_parameters(pix_x, pix_y, image):
    """
    Hillas parameters calculation.
    Parameters
    ----------
    pix_x : 1D Numpy array - Pixel x-coordinate
    pix_y : 1D Numpy array - Pixel y-coordinate
    image : 1D Numpy array - signal in each pixel
    Returns
    -------
    hillas_parameters - 1D Numpy array
    """

    amplitude = image.sum()
    assert amplitude > 0

    momdata = np.row_stack([pix_x,
                            pix_y,
                            pix_x * pix_x,
                            pix_y * pix_y,
                            pix_x * pix_y]) * image

    moms = momdata.sum(axis=1) / amplitude

    # calculate variances

    vx2 = moms[2] - moms[0] ** 2
    vy2 = moms[3] - moms[1] ** 2
    vxy = moms[4] - moms[0] * moms[1]

    # common factors:

    dd = vy2 - vx2
    zz = np.sqrt(dd ** 2 + 4.0 * vxy ** 2)

    # miss

    uu = 1.0 + dd / zz
    vv = 2.0 - uu
    miss = np.sqrt((uu * moms[0] ** 2 + vv * moms[1] ** 2) / 2.0
                   - moms[0] * moms[1] * 2.0 * vxy / zz)

    # shower shape parameters

    width = np.sqrt(vx2 + vy2 - zz)
    length = np.sqrt(vx2 + vy2 + zz)
    azwidth = np.sqrt(moms[2] + moms[3] - zz)

    # rotation angle of ellipse relative to centroid

    tanpsi_numer = (dd + zz) * moms[1] + 2.0 * vxy * moms[0]
    tanpsi_denom = (2 * vxy * moms[1]) - (dd - zz) * moms[0]
    psi = ((np.pi / 2.0) + np.arctan2(tanpsi_numer, tanpsi_denom))

    # polar coordinates of centroid

    rr = np.hypot(moms[0], moms[1])
    phi = np.arctan2(moms[1], moms[0])

    return np.array([amplitude, moms[0], moms[1], length, width, rr, phi, psi, miss])




def reco_impact_parameter(hillas_parameters_1, tel1, hillas_parameters_2, alt2, az2, tel2, z=0):
    """
    Geometric reconstruction of the impact parameter
    Parameters
    ----------
    hillas_parameters_1: Hillas parameters of the first telescope
    alt1: pointing altitude of first telescope
    az1: pointing azimuth of first telescope
    tel1: first telescope class
    hillas_parameters_2: Hillas parameters of the second telescope
    alt2: pointing altitude of second telescope
    az2: pointing azimuth of second telescope
    tel2: second telescope class
    z: Plan in which to compute the impact parameter

    Returns
    -------
    1D Numpy array - position in space of the impact parameter
    """
    cen_x1 = hillas_parameters_1[1]
    cen_y1 = hillas_parameters_1[2]
    psi1 = hillas_parameters_1[6]
    cen_x2 = hillas_parameters_2[1]
    cen_y2 = hillas_parameters_2[2]
    psi2 = hillas_parameters_2[6]

    alt1, az1 = geo.normal_to_altaz(tel1.normal)
    alt2, az2 = geo.normal_to_altaz(tel2.normal)

    barycenter1 = geo.barycenter_in_R(tel1, cen_x1, cen_y1)
    barycenter2 = geo.barycenter_in_R(tel2, cen_x2, cen_y2)

    n1 = geo.normal_vector_ellipse_plane(psi1, alt1, az1)
    n2 = geo.normal_vector_ellipse_plane(psi2, alt2, az2)
    n3 = np.array([0, 0, 1])

    A = np.array([n1, n2, n3])
    B = np.array([np.dot(n1, barycenter1), np.dot(n2, barycenter2), z])
    X = np.linalg.solve(A,B)
    return X


def reco_impact_parameter_test(hillas_parameters_1, tel1, hillas_parameters_2, tel2, z=0):
    cen_x1 = hillas_parameters_1[1]
    cen_y1 = hillas_parameters_1[2]
    psi1 = hillas_parameters_1[6]
    cen_x2 = hillas_parameters_2[1]
    cen_y2 = hillas_parameters_2[2]
    psi2 = hillas_parameters_2[6]

    psi1 = hillas_parameters_1[7] + np.pi/2.
    psi2 = hillas_parameters_2[7] + np.pi/2.

    barycenter1 = geo.barycenter_in_R(tel1, cen_x1, cen_y1)
    barycenter2 = geo.barycenter_in_R(tel2, cen_x2, cen_y2)

    alt1, az1 = geo.normal_to_altaz(tel1.normal)
    alt2, az2 = geo.normal_to_altaz(tel2.normal)

    n1 = geo.normal_vector_ellipse_plane_new(psi1, alt1, az1)
    n2 = geo.normal_vector_ellipse_plane_new(psi2, alt2, az2)
    n3 = np.array([0, 0, 1])


    A = np.array([n1, n2, n3])
    B = np.array([np.dot(n1, barycenter1), np.dot(n2, barycenter2), z])
    X = np.linalg.solve(A, B)
    return X




def impact_parameter_average(alltel, HillasParameters):
    """
    Reconstruction of the impact parameter taking the directions intersection
    of telescopes two by two and averaging the result

    Parameters
    ----------
    alltel: list of telescopes
    HillasParameters: list of hillas_parameters

    Returns
    -------
    Space coordinates of the impact parameter - list of floats
    """
    P = []
    for i in np.arange(len(alltel)-1):
        for j in np.arange(i+1, len(alltel)):
            #if(j != i):
            hp1 = HillasParameters[i]
            hp2 = HillasParameters[j]
            p = reco_impact_parameter_test(hp1, alltel[i], hp2, alltel[j], 0)
            P.append(p)
    P = np.array(P)
    return [P[:,0].mean(), P[:,1].mean(), P[:,2].mean()]



def coef_hillas_ponderation(hillas_parameters_1, hillas_parameters_2):
    """
    Weight of two telescopes reconstruction
    Parameters
    ----------
    hillas_parameters_1: list of hillas parameters for camera 1
    hillas_parameters_2: list of hillas parameters for camera 2

    Returns
    -------
    Float - weight
    """
    phi1 = hillas_parameters_1[6]
    psi1 = hillas_parameters_1[7]
    intensity1 = hillas_parameters_1[0]
    width1 = hillas_parameters_1[4]
    length1 = hillas_parameters_1[3]
    phi2 = hillas_parameters_2[6]
    psi2 = hillas_parameters_2[7]
    intensity2 = hillas_parameters_2[0]
    width2 = hillas_parameters_2[4]
    length2 = hillas_parameters_2[3]
    gx1 = hillas_parameters_1[1]
    gy1 = hillas_parameters_1[2]
    gx2 = hillas_parameters_2[1]
    gy2 = hillas_parameters_2[2]

    r = (gx1*gx1 + gy1*gy1 + gx2*gx2 + gy2*gy2)**6

    if r < 1e-6:
        return fabs(sin(psi1 - psi2)) / ((1.0/intensity1 + 1.0/intensity2) + (width1/length1 + width2/length2))
    else:
        return fabs(sin(psi1 - psi2)) / ( (1.0/intensity1 + 1.0/intensity2 + width1/length1 + width2/length2) * r)


def impact_parameter_ponderated(alltel, HillasParameters):
    """
    Reconstruction of the impact parameter using the weigths from coef_hillas_ponderation
    Parameters
    ----------
    alltel: list of telescopes
    HillasParameters: list of hillas_parameters for each telescope

    Returns
    -------
    Coordinates of the impact parameter - list of floats
    """
    P = []
    C = []
    for i in np.arange(len(alltel)):
        for j in np.arange(len(alltel)):
            if(j != i):
                hp1 = HillasParameters[i]
                hp2 = HillasParameters[j]
                p = reco_impact_parameter_test(hp1, alltel[i], hp2, alltel[j], 0)
                c = coef_hillas_ponderation(hp1, hp2)
                P.append(p)
                C.append(c)

    P = np.array(P)
    C = np.array(C)

    return [np.average(P[:,0],weights=C), np.average(P[:,1],weights=C), np.average(P[:,2],weights=C)]


def array_hillas_parameters(tel_array, trigger_intensity):
    """
    Given an array of telescopes, compute the hillas parameteres for each camera images
    Hillas Parameters are computed if the telescope triggered (total intensity > trigger_intensity)
    Parameters
    ----------
    tel_array: list of telescope classes
    trigger_intensity: float - threshold to trigger a telescope

    Returns
    -------
    list of hillas parameters. Hillas parameters of each cameras are contained in a 1D Numpy array.
    list of triggered telescopes
    """
    HP = []
    triggered_telescopes = []
    for tel in tel_array:
        pixels_signal = tel.signal_hist
        if len(pixels_signal.nonzero()[0]) > 1 and pixels_signal.sum() > trigger_intensity:
            hp = hillas_parameters(tel.pixel_tab[:, 0], tel.pixel_tab[:, 1], pixels_signal)
            HP.append(hp)
            triggered_telescopes.append(tel)

    return HP, triggered_telescopes
