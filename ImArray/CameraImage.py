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
    filename
    string - data file name
    Returns
    -------
    [X,Y] array with pixel positions
    """
    return np.loadtxt(filename)


def find_closest_pixel(pos, pixel_tab):
    """
    Compute the closest pixel of a given position
    Parameters
    ----------
    pos : 3-floats array giving a position
    pixel_tab : [X,Y] array of pixels positions

    Returns
    -------
    Pixel index in the given pixel_tab
    """
    X = np.array(pixel_tab[:,0])
    Y = np.array(pixel_tab[:,1])
    d = (X-pos[0])**2 + (Y-pos[1])**2
    if d.min() < ((pixel_tab[1:,0]-pixel_tab[0,0])**2 + (pixel_tab[1:,1]-pixel_tab[0,1])**2).min():
        return d.argmin()
    else:
        return None


def photon_count(photon_pos_tab, pixel_tab):
    """
    Count the number of photons in a pixel
    :param photon_pos_tab: array
    :param pixel_tab: array [X,Y]
    :return: array
    """
    count = np.zeros(len(pixel_tab))
    for photon in photon_pos_tab:
        pxi = find_closest_pixel(photon,pixel_tab)
        if pxi:
            count[pxi] += 1
    return np.column_stack((pixel_tab[:,0],pixel_tab[:,1],count))


def write_camera_image(pix_hist, filename="data/camera_image.txt"):
    """
    Save camera image in a file
    :param pix_hist: array - pixel histogram
    :param filename: string
    """
    np.savetxt(filename,pix_hist,fmt='%.5f')


def shower_image_in_camera(telescope, photon_pos_tab, pixel_pos_filename):
    """
    :param telescope: telescope class
    :param photon_pos_tab: array
    :param pixel_pos_filename: string
    :return: array - pixel histogram
    """
    pixel_tab = read_pixel_pos(pixel_pos_filename)
    pix_hist = photon_count(photon_pos_tab, pixel_tab)
    return pix_hist


def add_noise_poisson(pix_hist, lam=100):
    """
    Add Poisson noise to the image
    :param pix_hist: pixel histogram
    :param lam: lambda for Poisson law
    :return: pixel histogram
    """
    l = pix_hist[:,2].size
    if(lam>=0):
        pix_hist[:,2] += np.random.poisson(lam, l)
    return pix_hist


def camera_image(telescope, photon_pos_tab, result_filename="data/camera_image.txt", lam=100):
    """
    Compute the camera image
    :param telescope: telescope class
    :param photon_pos_tab: array
    :param result_filename: string
    :param lam: Poisson law lambda parameter to compute noise
    :return: pixel histogram
    """
    pix_hist = shower_image_in_camera(telescope, photon_pos_tab, telescope.pixpos_filename)
    pix_hist = add_noise_poisson(pix_hist, lam)
    write_camera_image(pix_hist, result_filename)
    return pix_hist


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

