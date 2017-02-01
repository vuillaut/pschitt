import geometry as geo
import CameraImage as ci
import object as obj
import Hillas as hillas
import numpy as np

def test_image_point_pfi():
    """
    Test the projection of a point in the image focal plane
    """
    tel = geo.Telescope([0, 0, 0], [0,0,1], 0)
    tel.focale = 10
    point = np.array([5,0,20])
    assert (geo.image_point_pfi(point, tel) == np.array([-2.5, 0, -10])).all()


def test_image_point_pfo():
    """
    Test the projection of a point in the object focal plane
    """
    tel = geo.Telescope([0, 0, 0], [0, 0, 1], 0)
    tel.focale = 10
    point = np.array([5, 0, 20])
    assert (geo.image_point_pfo(point, tel) == np.array([-2.5, 0, 10])).all()


def test_image_shower_pfi():
    """
    Test the projection of a shower in the image focal plane
    """
    shower = np.array([[0, 0, 20], [10, 0, 20]])
    tel = geo.Telescope([10, 0, 0], [0, 0, 1], 0)
    tel.focale= 10
    assert (geo.image_shower_pfi(shower, tel) == np.array([[15, 0, -10], [10, 0, -10]])).all()


def test_image_shower_pfo():
    """
    Test the projection of a shower in the object focal plane
    """
    shower = np.array([[0, 0, 20], [10, 0, 20]])
    tel = geo.Telescope([10, 0, 0], [0, 0, 1], 0)
    tel.focale = 10
    assert (geo.image_shower_pfo(shower, tel) == np.array([[15, 0, 10], [10, 0, 10]])).all()


def test_threshold_pass():
    """
    Test the threshold pass function
    """
    signal = np.array([0,3,1,5])
    assert(ci.threshold_pass(signal,8.7))


def test_shower_camera_image():
    shower = np.array([[0, 0, 20], [0, 0, 20], [0, 0, 20]])
    tel = geo.Telescope([10, 0, 0], [-1./3., 0, 2./3.], 0)
    tel.pixel_tab = np.array([[0,0],[0,1],[1,0],[-1,1]])
    print(ci.shower_camera_image(shower, tel))
    assert (ci.shower_camera_image(shower, tel) == np.array([3,0,0,0])).all()


def test_photons_to_signal():
    photon_pos_tab = np.array([[0, 0], [0, 1], [1, 2], [-1, -1]])
    pixel_tab = np.array([[0, 0], [0, 0.5], [0, 2]])
    print(ci.photons_to_signal(photon_pos_tab, pixel_tab))


#test_photons_to_signal()

#test_shower_camera_image()