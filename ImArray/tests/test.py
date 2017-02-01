import geometry as geo
import CameraImage as ci
import object as obj
import Hillas as hillas
import numpy as np

def test_image_point_pfi():
    tel = geo.Telescope([0, 0, 0], [0,0,1], 0)
    tel.focale = 10
    point = np.array([5,0,20])
    assert (geo.image_point_pfi(point, tel) == np.array([-2.5, 0, -10])).all()


def test_image_point_pfo():
    tel = geo.Telescope([0, 0, 0], [0, 0, 1], 0)
    tel.focale = 10
    point = np.array([5, 0, 20])
    assert (geo.image_point_pfo(point, tel) == np.array([-2.5, 0, 10])).all()


def test_image_shower_pfi():
    shower = np.array([[0, 0, 20], [10, 0, 20]])
    tel = geo.Telescope([10, 0, 0], [0, 0, 1], 0)
    tel.focale= 10
    assert (geo.image_shower_pfi(shower, tel) == np.array([[15, 0, -10], [10, 0, -10]])).all()


def test_image_shower_pfo():
    shower = np.array([[0, 0, 20], [10, 0, 20]])
    tel = geo.Telescope([10, 0, 0], [0, 0, 1], 0)
    tel.focale = 10
    assert (geo.image_shower_pfo(shower, tel) == np.array([[15, 0, 10], [10, 0, 10]])).all()

