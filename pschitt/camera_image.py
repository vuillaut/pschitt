# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
Functions to compute and display camera images
"""


import numpy as np
from . import geometry as geo
from . import emission as em
from numba import jit
# from multiprocessing import Pool
# from functools import partial


def read_pixel_pos(filename):
    """
    Read pixel position for a camera from a data file
    Parameters
    ----------
    filename : string - data file name

    Returns
    -------
    [X,Y] numpy array with pixel positions
    """
    return np.loadtxt(filename, unpack=True).T

@jit
def find_closest_pixel(pos, pixel_tab):
    """
    Compute the closest pixel of a given position
    Parameters
    ----------
    pos : 1D Numpy array giving a position
    pixel_tab : N-dim Numpy array of pixels positions [X,Y]

    Returns
    -------
    Pixel index in the given pixel_tab, corresponding distance to the pixel
    """
    x = pixel_tab - pos
    D2 = x[:, 0] ** 2 + x[:, 1] ** 2
    return D2.argmin()


@jit
def photons_to_signal(photon_pos_tab, pixel_tab, pixel_radius):
    """
    Count the number of photons in each pixel of the pixel_tab (camera)

    Parameters
    ----------
    photon_pos_tab: Numpy 2D array shape (N,2) with the position of the photons in the camera frame
    pixel_tab: Numpy 2D array with the positions of the pixels in the camera frame
    pixel_radius: float, maximal radius of one pixel to consider that a photon can be counted

    Returns
    -------
    Numpy 1D array with the signal in each pixel
    """
    count = np.zeros(len(pixel_tab))
    d_max2 = (pixel_radius)**2
    for photon in photon_pos_tab:
        D2 = np.sum((pixel_tab - photon)**2, axis=1)
        if D2.min() < d_max2:
            count[D2.argmin()] += 1

    return count


# @jit
# def pixel_in_camera(distances, dist_max2):
#     argmin = distances.argmin()
#     # if distances[argmin] < dist_max2:
#     #     return distances.argmin()
#     # else:
#     #     return -1
#     return (-1 * distances[argmin] > dist_max2) + argmin * (distances[argmin] <= dist_max2)
#
#
# def photons_to_signal_2(photon_pos_tab, pixel_tab, pixel_radius):
#     """
#     alternative to photons_to_signal with smart coding.
#     performances are similar to photons_to_signal when @jit is used
#     Parameters
#     ----------
#     photon_pos_tab: Numpy 2D array shape (N,2) with the position of the photons in the camera frame
#     pixel_tab: Numpy 2D array with the positions of the pixels in the camera frame
#
#     Returns
#     -------
#     Numpy 1D array with the signal in each pixel
#     """
#     d_max2 = (pixel_radius) ** 2
#     pixel_id = np.array([pixel_in_camera(np.sum((p-pixel_tab)**2, axis=1), d_max2) for p in photon_pos_tab])
#     return np.bincount(pixel_id[pixel_id >= 0], minlength=len(pixel_tab))



def write_camera_image(pix_hist, filename="data/camera_image.txt"):
    """
    Save camera image in a file
    Parameters
    ----------
    pix_hist : array - pixels signal
    filename : string
    """
    np.savetxt(filename,pix_hist,fmt='%.5f')


def shower_image_in_camera_old(telescope, photon_pos_tab, pixel_pos_filename):
    """
    Depreciated
    :param telescope: telescope class
    :param photon_pos_tab: array
    :param pixel_pos_filename: string
    :return: array - pixel histogram
    """
    pixel_tab = read_pixel_pos(pixel_pos_filename)
    pix_hist = photon_count(photon_pos_tab, pixel_tab)
    return pix_hist


def add_noise_poisson(signal, lam=100):
    """
    Add Poisson noise to the image

    Parameters
    ----------
    signal : pixels signal - array
    lam : lambda for Poisson law - float

    Returns
    -------
    signal array
    """
    if(lam>=0):
        signal += np.random.poisson(lam, signal.size)
    return signal


def add_noise_gaussian(signal, lam=10):
    """
    Add Poisson noise to the image

    Parameters
    ----------
    signal : pixels signal - array
    lam : lambda for Poisson law - float

    Returns
    -------
    signal array
    """
    if lam > 0:
        signal += np.random.normal(0, lam, signal.size)
    return signal




def shower_image_in_camera(telescope, photon_pos_tab, lam=0, result_filename=None):
    """
    Compute the camera image given the positions of the photons in the camera frame.
    Poissonian noise can be added if lambda > 0
    The image can be saved in a textfile if a result_filename is given
    Parameters
    ----------
    telescope: telescope class
    photon_pos_tab: Numpy 2D array with the cartesian positions of the photons in the camera frame [[x1,y1],[x2,y2],...]
    lam: lambda for the poissonian noise
    result_filename: string, name of file to write the resulting image

    Returns
    -------
    Numpy 1D array with the photon count in each pixel
    """
    pixels_signal = photons_to_signal(photon_pos_tab, telescope.pixel_tab, telescope.pixel_radius)

    pixels_signal = add_noise_gaussian(pixels_signal, lam)
    if result_filename:
        write_camera_image(pix_hist, result_filename)
    return pixels_signal


def threshold_pass(pix_signal, threshold):
    """
    Test if image signal pass a threshold

    Parameters
    ----------
    pix_signal: photon count 1D Numpy array
    threshold: float

    Returns
    -------
    Boolean
    """
    if pix_signal.sum() > threshold:
        return True
    else:
        return False


def shower_camera_image(shower, tel, noise = 0, **kwargs):
    """
    Given a shower object and a telescope, compute the image of the shower in the camera
    Background noise can be added
    Parameters
    ----------
    shower: 3D Numpy array with a list of space position points composing the shower
    tel: telescope class

    Returns
    -------
    Numpy 1D array of the photon count in each pixel of the telescope camera
    """

    # Image the shower in the camera focale plane
    shower_image = geo.image_shower_pfo(shower.particles, tel)

    # Change reference frame to have particles images in the camera frame
    shower_cam = geo.site_to_camera_cartesian(shower_image, tel)

    # impact_distance = np.sqrt(np.sum((shower.impact_point - tel.mirror_center)**2))

    # Only part of the photons reach the telescope camera due to absorption
    break_angle = 0.018 # = 1 degree
    alpha = 0.75
    # mask = em.mask_transmitted_particles(tel, shower, em.angular_profile_exp_falloff, break_angle, alpha)
    mask = em.mask_transmitted_particles(tel, shower)
    photons_in_camera = shower_cam[:, [0, 1]][mask]

    tel.signal_hist = shower_image_in_camera(tel, photons_in_camera, noise)
    return tel.signal_hist


def array_shower_imaging(shower, tel_array, noise):
    """
    Given a shower object and an array of telescopes, compute the image of the shower in each camera
    Background noise can be added
    The camera signal is registered in all tel.signal_hist
    Parameters
    ----------
    shower: 3D Numpy array with a list of space position points composing the shower
    tel_array: Numpy array or list of telescope classes
    noise: float
    """

    for tel in tel_array:
        shower_camera_image(shower, tel, noise)


