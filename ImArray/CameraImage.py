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


def find_closest_pixel(pos, pixel_tab):
    """
    Compute the closest pixel of a given position
    Parameters
    ----------
    pos : 3-floats array giving a position
    pixel_tab : [X,Y] array of pixels positions

    Returns
    -------
    Pixel index in the given pixel_tab, corresponding distance to the pixel
    """
    #D = np.linalg.norm(pixel_tab-pos, axis=1)
    #linalg.norm is surprisingly slow
    x = pixel_tab - pos
    D = np.sqrt(x[:,0]**2+x[:,1]**2)
    return D.argmin()


def photon_count(photon_pos_tab, pixel_tab):
    """
    Count the number of photons in a pixel
    :param photon_pos_tab: array
    :param pixel_tab: array [X,Y]
    :return: array
    """
    count = np.zeros(len(pixel_tab))
    d_max2 = pixel_tab[:,0]**2 + pixel_tab[:,1]**2
    for photon in photon_pos_tab:
        if photon[0]**2 + photon[1]**2 < d_max2:
            pxi = find_closest_pixel(photon, pixel_tab)
            count[pxi] += 1
    return np.column_stack((pixel_tab[:,0],pixel_tab[:,1],count))


def photons_to_signal(photon_pos_tab, pixel_tab):
    """
    Count the number of photons in each pixel of the camera
    :param photon_pos_tab: array
    :param pixel_tab: array [X,Y]
    :return: array
    """
    signal = np.zeros(len(pixel_tab))
    for photon in photon_pos_tab:
        pxi = find_closest_pixel(photon, pixel_tab)
        if pxi:
            signal[pxi] += 1
    return signal


def write_camera_image(pix_hist, filename="data/camera_image.txt"):
    """
    Save camera image in a file
    :param pix_hist: array - pixel histogram
    :param filename: string
    """
    np.savetxt(filename,pix_hist,fmt='%.5f')


def shower_image_in_camera_old(telescope, photon_pos_tab, pixel_pos_filename):
    """
    :param telescope: telescope class
    :param photon_pos_tab: array
    :param pixel_pos_filename: string
    :return: array - pixel histogram
    """
    pixel_tab = read_pixel_pos(pixel_pos_filename)
    pix_hist = photon_count(photon_pos_tab, pixel_tab)
    return pix_hist


def shower_image_in_camera(telescope, photon_pos_tab):
    """
    Compute the real image recorded by the camera from the image of the shower in the camera plane
    Parameters
    ----------
    telescope: telescope class
    photon_pos_tab: 2D numpy array of the photons position[[x1,y1],[x2,y2],...[xn,yn]]

    Returns
    -------
    1D array of the integrated signal in each pixel
    """
    signal = photons_to_signal(photon_pos_tab, telescope.pixel_tab)
    return signal


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


def camera_image(telescope, photon_pos_tab, lam=100, result_filename=None):
    """
    Compute the camera image
    :param telescope: telescope class
    :param photon_pos_tab: array
    :param result_filename: string
    :param lam: Poisson law lambda parameter to compute noise
    :return: pixel histogram
    """
    pixels_signal = shower_image_in_camera(telescope, photon_pos_tab)
    pixels_signal = add_noise_poisson(pixels_signal, lam)
    if result_filename:
        write_camera_image(pix_hist, result_filename)
    return pixels_signal


def threshold_pass(pix_hist, threshold):
    """
    Test if image signal pass a threshold
    :param pix_hist: pixel histogram
    :param threshold: float
    :return: Boolean
    """
    if pix_hist.sum()>threshold:
        return True
    else:
        return False

