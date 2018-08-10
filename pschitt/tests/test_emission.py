import pschitt.emission as em
import numpy as np


def test_angular_profile_exp_peak():
    assert em.angular_profile.verify_normalisation(em.angular_profile.exp_peak)

def test_angular_profile_constant():
    assert em.angular_profile.verify_normalisation(em.angular_profile.constant)

def test_angular_profile_heaviside():
    assert em.angular_profile.verify_normalisation(em.angular_profile.heaviside)

def test_angular_profile_lgdt06():
    assert em.angular_profile.verify_normalisation(em.angular_profile.lgdt06)