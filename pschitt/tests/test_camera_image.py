from pschitt import camera_image as ci
import numpy as np


def test_photons_to_signal():
    pix = np.array([[-1, -1], [1, 1]])
    ph = np.array([[0, 0], [0, 1], [-1, 0], [0, -4], [1, 3]])

    signal = ci.photons_to_signal(ph, pix, 1.01)
    assert (signal == np.ones(2)).all()


def test_threshold_pass():
    """
    Test the threshold pass function
    """
    signal = np.array([0,3,1,5])
    assert(ci.threshold_pass(signal,8.7))


def test_shower_camera_image():
    """
    Test the image camera of a shower
    """
    import pschitt.geometry as geo
    import pschitt.sky_objects as sky
    shower = sky.shower()
    shower.number_of_particles = 3
    shower.particles = np.array([[0, 0, 20], [0, 0, 20], [0, 0, 20]])
    tel = geo.Telescope([10, 0, 0], [-1./3., 0, 2./3.])

    tel.pixel_tab = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    assert (ci.shower_camera_image(shower, tel) == np.array([3, 0, 0, 0])).all()

    tel.pixel_tab = np.array([[0, 0], [0, 1], [1, 0], [-1, 1]])
    tel.camera_type = 'not_std'
    assert (ci.shower_camera_image(shower, tel) == np.array([3, 0, 0, 0])).all()

