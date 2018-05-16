from pschitt import sky_objects as obj
import numpy as np


def test_shower_creation():
    assert obj.shower()

def test_shower_linear():
    shower = obj.shower()
    top = np.array([0,0,110])
    bot = np.array([0,0,20])
    shower.linear_segment(top, bot)
    assert (shower.particles[:,2] - np.linspace(110,20,10) < 1e-8).all()

def test_shower_rotation():
    shower = obj.shower()
    top = np.array([0,0,110])
    bot = np.array([0,0,20])
    shower.linear_segment(top, bot)
    r = obj.shower_array_rot(shower.particles, np.pi / 4., 0)
    assert (r[:,0] - r[:,2] < 1e-10).all()
