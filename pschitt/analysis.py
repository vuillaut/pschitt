from . import geometry as geo



def triggered_telescopes(array, trigger_threshold):
    """
    Return the list of telescopes that triggered in the array

    Parameters
    ----------
    array: list of telescope Class
    trigger_threshold: float

    Returns
    -------
    list of telescope Class
    """
    trig_tel = [tel for tel in array if tel.signal_hist.sum() > trigger_threshold]

    return trig_tel


def multiplicity(array, trigger_threshold):
    """
    Return the number of telescopes that triggered

    Parameters
    ----------
    array: list of telescope Class
    trigger_threshold: float

    Returns
    -------
    `int`
    """
    return len(triggered_telescopes(array, trigger_threshold))


