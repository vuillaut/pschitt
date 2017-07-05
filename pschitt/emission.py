# Licensed under a 3-clause BSD style license - see LICENSE.rst


def emission_coef(distance_impact):
    """
    Compute a coefficient between 0 and 1 taking into accounts all radiative transfer effects

    Parameters
    ----------
    distance_impact: float, distance to impact parameter


    Returns
    -------
    float between 0 and 1
    """
    ep = emission_profile(distance_impact)
    abs = 0
    return ep * (1-abs)


def emission_profile(dist):
    """
    Dummy function to select one emission profile
    Parameters
    ----------
    dist: float, distance to the impact parameter

    Returns
    -------
    float between 0 and 1
    """
    return constant_profile(dist)
    # return ground_profile_1(dist, shift=250)



def constant_profile(dist):
    """
    Given a distance to the impact point, give the probability to detect the shower
    Dummy function where the probability is always one

    Parameters
    ----------
    dist: float, distance to the impact point

    Returns
    -------
    float between 0 and 1
    """
    return 1


def power_law_profile(dist, alpha=2):
    """
    Given a distance to the impact point, give the probability to detect the shower
    Power-law decrease from impact point. By default the power index alpha = 2

    Parameters
    ----------
    dist: float, distance to the impact point

    Returns
    -------
    float between 0 and 1
    """
    return 1./(1. + dist)**2


def ground_profile_heaviside(dist, shift=120):
    """
    Given a distance to the impact point, give the probability to detect the shower
    Simple emission profile constant below 120m then with an decrease

    Parameters
    ----------
    dist: float, distance to the impact point

    Returns
    -------
    float between 0 and 1
    """
    return (dist<=shift) * 1. + (dist>shift) * 0.


def ground_profile_1(dist, shift=120):
    """
    Given a distance to the impact point, give the probability to detect the shower
    Simple emission profile constant below 120m then with an decrease

    Parameters
    ----------
    dist: float, distance to the impact point

    Returns
    -------
    float between 0 and 1
    """
    return (dist<=shift) * 1. + (dist>shift) * 1./(dist - shift)


# def model3D_profile(angle):
#     """
#     Emission profile used in model3D
#     Parameters
#     ----------
#     angle: float - solid angle to the shower axis
#
#     Returns
#     -------
#     Emission probability
#     """
#     return 150 *( (angle<=1) + (angle>1)