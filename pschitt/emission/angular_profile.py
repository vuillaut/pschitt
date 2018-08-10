import numpy as np


def constant(angle):
    """
    Return a constant emission profile for angles between 0 and pi
    The profile is normalized so that $ 2\pi \int_0^\pi x * I(x) dx = 1$

    Parameters
    ----------
    angle: `numpy.ndarray` or float

    Returns
    -------
    `numpy.ndarray` of the same shape as angle or float
    """
    K = 1. / np.pi ** 3
    if type(angle) == np.ndarray:
        assert (angle <= np.pi).all()
        return K * np.ones(len(angle))
    else:
        assert angle <= np.pi
        return K


def heaviside(angle, limit=0.1):
    """
    Return a normalized emission profile I(angle) following a heaviside profile
    (constant between 0 and limit and null above)
    The profile is normalized so that $ 2\pi \int_0^\pi x * I(x) dx = 1$

    Parameters
    ----------
    angle: `numpy.ndarray` or float
    limit: float

    Returns
    -------
    `numpy.ndarray` of the same shape as angle or float
    """
    assert limit > 0
    K = 1. / (np.pi * min(np.pi, limit) ** 2)
    if type(angle) == np.ndarray:
        assert (angle <= np.pi).all()
        return K * (angle <= limit)
    else:
        if angle <= limit:
            return K
        else:
            return 0


def lgdt06(angle, eta=0.001):
    """
    Return a normalized emission profile I(angle) following the law proposed by [Lemoine et al 06]
    The profile is normalized so that $ 2\pi \int_0^\pi x * I(x) dx = 1$
    The normalization has been updated to work for any eta < pi.

    Parameters
    ----------
    angle: `numpy.ndarray` or float
    eta: float

    Returns
    -------
    `numpy.ndarray` of the same shape as angle or float
    """

    assert eta > 0

    # K = 1./(9 * np.pi * eta**2) #approx by [Lemoine et al 06], valid if eta<<pi

    # Normalisation for any eta:
    K = 1. / (np.pi * eta ** 2 * (1 + 8 * (1 - np.exp(0.25 * (1 - np.pi / eta)))))

    if type(angle) == np.ndarray:
        assert (angle < np.pi).all()
        y = K * np.ones(len(angle))
        y[angle > eta] = K * eta / angle[angle>eta] * np.exp(- (angle[angle>eta] - eta) / (4 * eta))
        return y

    else:
        assert angle < np.pi
        if angle <= eta:
            return K
        else:
            return K * eta / angle * np.exp(- (angle - eta) / (4 * eta))


def verify_normalisation(profile, **kwargs):
    """
    Quick verification if a profile function I is normalised
    so that $ 2\pi \int_0^\pi x * I(x) dx = 1$
    Disclaimer: the precision might not be enough for certain profiles.

    Parameters
    ----------
    profile: profile function
    **kwargs: profile function arguments

    Returns
    -------
    bool
    """
    import scipy.integrate as integrate
    integral = integrate.quad(lambda x: 2*np.pi*x*profile(x, **kwargs), 0, np.pi)
    return np.isclose(integral[0], 1)



def exp_peak(angle, eta=0.1, alpha=1):
    """
    Return a normalized emission profile I(angle) with an exponential increase from 0 to eta
    followed by an exponential decrease from eta to 2*eta.
    The function is null between 2*eta and pi.
    The profile is normalized so that $ 2\pi \int_0^\pi x * I(x) dx = 1$

    Parameters
    ----------
    angle: `numpy.ndarray` or float
    eta: float, peak angle
    alpha: float, exponential rate

    Returns
    -------
    `numpy.ndarray` of the same shape as angle or float
    """

    assert eta > 0
    assert alpha > 0
    assert 2*eta < np.pi

    K = alpha / (4 * np.pi * (np.exp(alpha * eta) - alpha * eta - 1))

    if type(angle) == np.ndarray:
        assert (angle < np.pi).all()
        return K / angle * (np.exp(alpha * angle) - 1) * (angle < eta) \
               + K / angle * (np.exp(alpha * (2 * eta - angle)) - 1) * (angle >= eta) * (angle <= 2 * eta)

    else:
        assert angle < np.pi
        if angle <= eta:
            return K / angle * (np.exp(alpha * angle) - 1)
        elif angle < 2 * eta:
            return K / angle * (np.exp(alpha * (2 * eta - angle)) - 1)
        else:
            return 0
