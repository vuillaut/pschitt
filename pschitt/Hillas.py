# Hillas functions copied from CTAPIPE https://github.com/cta-observatory/ctapipe/blob/master/ctapipe/reco/hillas.py

# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Hillas shower parametrization.
TODO:
-----
- Should have a separate function or option to compute 3rd order
  moments + asymmetry (which are not always needed)
- remove alpha calculation (which is only about (0,0), and make a get
  alpha function that does it from an arbitrary point given a
  pre-computed list of parameters
"""
import numpy as np
from astropy.units import Quantity
from collections import namedtuple
import astropy.units as u
import geometry as geo
from math import *


__all__ = [
    'MomentParameters',
    'HighOrderMomentParameters',
    'hillas_parameters',
]


MomentParameters = namedtuple(
    "MomentParameters",
    "size,cen_x,cen_y,length,width,r,phi,psi,miss"
)
"""Shower moment parameters up to second order.
See also
--------
HighOrderMomentParameters, hillas_parameters, hillas_parameters_2
"""

HighOrderMomentParameters = namedtuple(
    "HighOrderMomentParameters",
    "skewness,kurtosis,asymmetry"
)
"""Shower moment parameters of third order.
See also
--------
MomentParameters, hillas_parameters, hillas_parameters_2
"""

class HillasParameters():
    """
    Class to handle Hillas parameters
    """
    def __init__(self):
        phi = 0.
        intensity = 0.
        width = 0.
        length =0.
        gx = 0.
        gy = 0.


    def plot_ellipse(self):
        """
        To write: plot the ellipse in the camera
        """


def hillas_parameters_1(pix_x, pix_y, image):
    """Compute Hillas parameters for a given shower image.
    Reference: Appendix of the Whipple Crab paper Weekes et al. (1998)
    http://adsabs.harvard.edu/abs/1989ApJ...342..379W
    (corrected for some obvious typos)
    Parameters
    ----------
    pix_x : array_like
        Pixel x-coordinate
    pix_y : array_like
        Pixel y-coordinate
    image : array_like
        Pixel values corresponding
    Returns
    -------
    hillas_parameters : `MomentParameters`
    """

    pix_x = Quantity(np.asanyarray(pix_x, dtype=np.float64)).value
    pix_y = Quantity(np.asanyarray(pix_y, dtype=np.float64)).value
    image = np.asanyarray(image, dtype=np.float64)
    assert pix_x.shape == image.shape
    assert pix_y.shape == image.shape

    # Compute image moments
    _s = np.sum(image)
    m_x = np.sum(image * pix_x) / _s
    m_y = np.sum(image * pix_y) / _s
    m_xx = np.sum(image * pix_x * pix_x) / _s  # note: typo in paper
    m_yy = np.sum(image * pix_y * pix_y) / _s
    m_xy = np.sum(image * pix_x * pix_y) / _s  # note: typo in paper

    # Compute major axis line representation y = a * x + b
    S_xx = m_xx - m_x * m_x
    S_yy = m_yy - m_y * m_y
    S_xy = m_xy - m_x * m_y
    print(S_xy)
    d = S_yy - S_xx
    temp = d * d + 4 * S_xy * S_xy
    a = (d + np.sqrt(temp)) / (2 * S_xy)
    b = m_y - a * m_x

    # Compute Hillas parameters
    width_2 = (S_yy + a * a * S_xx - 2 * a * S_xy) / (1 + a * a)
    width = np.sqrt(width_2)
    length_2 = (S_xx + a * a * S_yy + 2 * a * S_xy) / (1 + a * a)
    length = np.sqrt(length_2)
    miss = np.abs(b / (1 + a * a))
    r = np.sqrt(m_x * m_x + m_y * m_y)
    phi = np.arctan2(m_y, m_x)
    psi = np.arctan2(b, b/a) + np.pi/2.

    # Compute azwidth by transforming to (p, q) coordinates
    sin_theta = m_y / r
    cos_theta = m_x / r
    q = (m_x - pix_x) * sin_theta + (pix_y - m_y) * cos_theta
    m_q = np.sum(image * q) / _s
    m_qq = np.sum(image * q * q) / _s
    azwidth_2 = m_qq - m_q * m_q
    azwidth = np.sqrt(azwidth_2)

    '''return MomentParameters(size=_s, cen_x=m_x, cen_y=m_y, length=length,
                            width=width, r=r, phi=phi, psi=None, miss=miss)
    '''
    return [_s,m_x,m_y,length,width,r,phi,psi,miss]


def hillas_parameters_2(pix_x, pix_y, image):
    """Compute Hillas parameters for a given shower image.
    Alternate implementation of `hillas_parameters` ...
    in the end we'll just keep one, but we're using Hilllas parameter
    computation as an example for performance checks.
    Parameters
    ----------
    pix_x : array_like
        Pixel x-coordinate
    pix_y : array_like
        Pixel y-coordinate
    image : array_like
        Pixel values corresponding
    Returns
    -------
    hillas_parameters : `MomentParameters`
    """

    assert image.sum() > 0

    #unit = pix_x.unit
    unit = 1.

    pix_x = Quantity(np.asanyarray(pix_x, dtype=np.float64)).value
    pix_y = Quantity(np.asanyarray(pix_y, dtype=np.float64)).value

    assert pix_x.shape == image.shape
    assert pix_y.shape == image.shape

    # Compute image moments (done in a bit faster way, but putting all
    # into one 2D array, where each row will be summed to calculate a
    # moment) However, this doesn't avoid a temporary created for the
    # 2D array

    size = image.sum()
    momdata = np.row_stack([pix_x,
                            pix_y,
                            pix_x * pix_x,
                            pix_y * pix_y,
                            pix_x * pix_y]) * image

    moms = momdata.sum(axis=1) / size

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
    psi = ((np.pi / 2.0) + np.arctan2(tanpsi_numer, tanpsi_denom))* u.rad

    # polar coordinates of centroid

    rr = np.hypot(moms[0], moms[1])
    phi = np.arctan2(moms[1], moms[0])

    '''return MomentParameters(size=size, cen_x=moms[0]*unit, cen_y=moms[1]*unit,
                            length=length*unit, width=width*unit, r=rr, phi=phi,
                            psi=psi.to(u.rad) , miss=miss*unit)
    '''
    return [size,moms[0],moms[1],length,width,rr,phi,psi.value,miss]

# use the 2 version by default
hillas_parameters = hillas_parameters_1

"""
def camera_frame_to_R(alt, az):
    '''
    return coordinates of base vectors of camera frame in global frame
    '''
    xx = [-sin(az), cos(az), 0]
    yy = [-sin(alt) * cos(az), -sin(az) * sin(alt), cos(alt)]
    zz = [cos(alt) * cos(az), cos(alt) * sin(az), sin(alt)]
    return np.array([xx, yy, zz])
"""



def reco_impact_parameter(hillas_parameters_1, alt1, az1, tel1, hillas_parameters_2, alt2, az2, tel2, z=0):
    cen_x1 = hillas_parameters_1[1]
    cen_y1 = hillas_parameters_1[2]
    psi1 = hillas_parameters_1[6]
    cen_x2 = hillas_parameters_2[1]
    cen_y2 = hillas_parameters_2[2]
    psi2 = hillas_parameters_2[6]

    barycenter1 = geo.barycenter_in_R(tel1, alt1, az1, cen_x1, cen_y1)
    barycenter2 = geo.barycenter_in_R(tel2, alt2, az2, cen_x2, cen_y2)

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
    phi1 = hillas_parameters_1[6]
    intensity1 = hillas_parameters_1[0]
    width1 = hillas_parameters_1[4]
    length1 = hillas_parameters_1[3]
    phi2 = hillas_parameters_2[6]
    intensity2 = hillas_parameters_2[0]
    width2 = hillas_parameters_2[4]
    length2 = hillas_parameters_2[3]
    gx1 = hillas_parameters_1[1]
    gy1 = hillas_parameters_1[2]
    gx2 = hillas_parameters_2[1]
    gy2 = hillas_parameters_2[2]

    r = (gx1*gx1 + gy1*gy1 + gx2*gx2 + gy2*gy2)**6;
    #print(gx1,gy1,gx2,gy2)
    if r < 1e-6:
        return fabs(sin(phi1 - phi2)) / ((1.0/intensity1 + 1.0/intensity2) + (width1/length1 + width2/length2));
    else:
        return fabs(sin(phi1 - phi2)) / ( (1.0/intensity1 + 1.0/intensity2 + width1/length1 + width2/length2) * r);

def impact_parameter_ponderated(alltel, HillasParameters):
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
                r12 = sqrt(hp1[1]**2 + hp1[2]**2)
                r22 = sqrt(hp2[1]**2 + hp2[2]**2)
                #if(r12>0.9*alltel[i].camera_size or r22>0.9*alltel[j].camera_size):
                #    c = 0
                C.append(c)

    P = np.array(P)
    C = np.array(C)
    #if(C.min()!=C.max()):
    #    C = (C - C.min())/(C.max()-C.min())

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