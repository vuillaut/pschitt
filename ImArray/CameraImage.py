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

"""
Functions to compute and display camera images
"""


import numpy as np
import geometry as geo
from numba import jit


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
    #D = np.linalg.norm(pixel_tab-pos, axis=1)
    #linalg.norm is surprisingly slow
    x = pixel_tab - pos
    # D = np.sqrt(x[:, 0]**2 + x[:, 1]**2)
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
            # pxi = find_closest_pixel(photon, pixel_tab)
            x = pixel_tab - photon
            D2 = x[:, 0] ** 2 + x[:, 1] ** 2
            # pxi = D2.argmin()
            count[D2.argmin()] += 1

    # for photon in photon_pos_tab:
    #     pxi = find_closest_pixel(photon, pixel_tab)
    #     count[pxi] += photon[0]**2 + photon[1]**2 < d_max2

    return count


def write_camera_image(pix_hist, filename="data/camera_image.txt"):
    """
    Save camera image in a file
    :param pix_hist: array - pixel histogram
    :param filename: string
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
    :param pix_hist: pixel histogram
    :param lam: lambda for Poisson law
    :return: pixel histogram
    """
    if(lam>=0):
        signal += np.random.poisson(lam, signal.size)
    return signal


def shower_image_in_camera(telescope, photon_pos_tab, lam=0, result_filename=None):
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
        direction = np.array(shower[shower[:, 2].argmin()] - shower[shower[:, 2].argmax()])
    else:
        direction = shower_direction
    direction = direction/np.sqrt((direction**2).sum())
    visible = geo.mask_visible_particles(tel, shower, direction)
    shower_image = geo.image_shower_pfo(shower[visible], tel)
    shower_cam = geo.site_to_camera_cartesian(shower_image, tel)
    tel.signal_hist = shower_image_in_camera(tel, shower_cam[:, [0, 1]], noise)
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


