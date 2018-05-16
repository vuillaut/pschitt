from pschitt import camera_image as ci
import numpy as np


def test_photons_to_signal():
    pix = np.array([[-1, -1], [1, 1]])
    ph = np.array([[0, 0], [0, 1], [-1, 0], [0, -4], [1, 3]])

    signal = ci.photons_to_signal(ph, pix, 1.01)
    assert (signal == np.ones(2)).all()