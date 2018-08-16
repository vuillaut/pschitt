import pschitt.emission as em
import numpy as np


def test_verify_normalisation_solid_angle():
    assert em.angular_profile.verify_normalisation_solid_angle(lambda x: 1. / np.pi ** 3)


def test_angular_profile_exp_peak():
    assert em.angular_profile.exp_peak(0, 0.1, 1) == 0
    assert em.angular_profile.exp_peak(0.1, 0.1, 1) == 1


def test_angular_profile_constant():
    assert em.angular_profile.constant(np.pi * np.random.rand()) == 1

def test_angular_profile_heaviside():
    assert em.angular_profile.heaviside(np.random.rand(), 1) == 1
    assert em.angular_profile.heaviside(np.random.rand() + 1, 1) == 0

def test_angular_profile_lgdt06():
    eta = np.random.rand()
    assert em.angular_profile.lgdt06(np.random.rand()*eta, eta) == 1
    assert em.angular_profile.lgdt06(eta, eta) == 1
    assert em.angular_profile.lgdt06(2 * eta, eta) == 0.5 * np.exp(-0.25)