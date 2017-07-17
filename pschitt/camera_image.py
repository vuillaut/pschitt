# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
Functions to compute and display camera images
"""


import numpy as np
from . import geometry as geo
from numba import jit
from . import emission as emi
from multiprocessing import Pool
from functools import partial


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



def photons_to_signal(photon_pos_tab, pixel_tab):
    """
    Count the number of photons in each pixel of the pixel_tab (camera)

    Parameters
    ----------
    photon_pos_tab: Numpy 2D array shape (N,2) with the position of the photons in the camera frame
    pixel_tab: Numpy 2D array with the positions of the pixels in the camera frame

    Returns
    -------
    Numpy 1D array with the signal in each pixel
    """
    count = np.zeros(len(pixel_tab))
    d_max2 = (pixel_tab[:, 0]**2 + pixel_tab[:, 1]**2).max()
    for photon in photon_pos_tab:
        if photon[0]**2 + photon[1]**2 < d_max2:
            x = pixel_tab - photon
            D2 = x[:, 0] ** 2 + x[:, 1] ** 2
            count[D2.argmin()] += 1

    return count


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


def shower_image_in_camera(telescope, photon_pos_tab, lam=0, impact_distance=0, result_filename=None):
    """
    Compute the camera image given the positions of the photons in the camera frame.
    Poissonian noise can be added if lambda > 0
    The image can be saved in a textfile if a result_filename is given
    Parameters
    ----------
    telescope: telescope class
    photon_pos_tab: Numpy 2D array with the cartesian positions of the the photons in the camera frame [[x1,y1],[x2,y2],...]
    lam: lambda for the poissonian noise
    result_filename: string, name of file to write the resulting image

    Returns
    -------
    Numpy 1D array with the photon count in each pixel
    """
    pixels_signal = photons_to_signal(photon_pos_tab, telescope.pixel_tab)
    pixels_signal = pixels_signal * emi.emission_coef(impact_distance)
    pixels_signal = add_noise_poisson(pixels_signal, lam)
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


def shower_camera_image(shower, tel, noise = 0, shower_direction=None):
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

    if shower_direction==None:
        direction = np.array(shower.array[shower.array[:, 2].argmin()] - shower.array[shower.array[:, 2].argmax()])
    else:
        direction = shower_direction
    direction = direction/np.sqrt((direction**2).sum())
    visible = geo.mask_visible_particles(tel, shower.array, direction)
    # visible = np.ones(len(shower), dtype=bool)
    shower_image = geo.image_shower_pfo(shower.array[visible], tel)
    shower_cam = geo.site_to_camera_cartesian(shower_image, tel)
    impact_distance = np.sqrt(np.sum((shower.impact_point - tel.mirror_center)**2))
    tel.signal_hist = shower_image_in_camera(tel, shower_cam[:, [0, 1]], noise, impact_distance=impact_distance)
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



def shower_camera_image_argorder(shower, noise, tel):
    shower_camera_image(shower, tel, noise)
    return None


def array_shower_imaging_multiproc(shower, tel_array, noise):
    """
    TEST
    Given a shower object and an array of telescopes, compute the image of the shower in each camera
    Background noise can be added
    The camera signal is registered in all tel.signal_hist
    Parameters
    ----------
    shower: 3D Numpy array with a list of space position points composing the shower
    tel_array: Numpy array or list of telescope classes
    noise: float
    """

    pool = Pool(processes=4)
    func = partial(shower_camera_image_argorder, shower, noise)
    pool.map(func, tel_array)
    pool.close()
    pool.join()
