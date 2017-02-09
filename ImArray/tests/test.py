import geometry as geo
import CameraImage as ci
import object as obj
import Hillas as hillas
import numpy as np
import unittest

class TestGeometry(unittest.TestCase):

    def test_image_point_pfi(self):
        """
        Test the projection of a point in the image focal plane
        """
        tel = geo.Telescope([0, 0, 0], [0,0,1], 0)
        tel.focale = 10
        point = np.array([5,0,20])
        assert (geo.image_point_pfi(point, tel) == np.array([-2.5, 0, -10])).all()


    def test_image_point_pfo(self):
        """
        Test the projection of a point in the object focal plane
        """
        tel = geo.Telescope([0, 0, 0], [0, 0, 1], 0)
        tel.focale = 10
        point = np.array([5, 0, 20])
        assert (geo.image_point_pfo(point, tel) == np.array([-2.5, 0, 10])).all()


    def test_image_shower_pfi(self):
        """
        Test the projection of a shower in the image focal plane
        """
        shower = np.array([[0, 0, 20], [10, 0, 20]])
        tel = geo.Telescope([10, 0, 0], [0, 0, 1], 0)
        tel.focale= 10
        assert (geo.image_shower_pfi(shower, tel) == np.array([[15, 0, -10], [10, 0, -10]])).all()


    def test_image_shower_pfo(self):
        """
        Test the projection of a shower in the object focal plane
        """
        shower = np.array([[0, 0, 20], [10, 0, 20]])
        tel = geo.Telescope([10, 0, 0], [0, 0, 1], 0)
        tel.focale = 10
        assert (geo.image_shower_pfo(shower, tel) == np.array([[15, 0, 10], [10, 0, 10]])).all()


    def test_threshold_pass(self):
        """
        Test the threshold pass function
        """
        signal = np.array([0,3,1,5])
        assert(ci.threshold_pass(signal,8.7))


    def test_shower_camera_image(self):
        """
        Test the image camera of a shower
        """
        shower = np.array([[0, 0, 20], [0, 0, 20], [0, 0, 20]])
        tel = geo.Telescope([10, 0, 0], [-1./3., 0, 2./3.], 0)
        tel.pixel_tab = np.array([[0,0],[0,1],[1,0],[-1,1]])
        assert (ci.shower_camera_image(shower, tel) == np.array([3, 0, 0, 0])).all()


    def test_photons_to_signal(self):
        photon_pos_tab = np.array([[0, 0], [0, 1], [1, 2], [-1, -1]])
        pixel_tab = np.array([[0, 0], [0, 0.51], [0, 2]])
        assert (ci.photons_to_signal(photon_pos_tab, pixel_tab) == np.array([ 2,  1,  0])).all()


    def test_is_particle_visible(self):
        particle_position = np.array([10, 0, 20])
        telescope = geo.Telescope([0, 0, 0], [1, 0, 2], 0)
        particle_direction = np.array([-1, 0, -2])
        particle_energy = 1
        assert geo.is_particle_visible(particle_position, particle_direction, particle_energy, telescope)



if __name__ == '__main__':
    unittest.main()