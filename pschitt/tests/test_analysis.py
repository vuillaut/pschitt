from pschitt import analysis as ana
import numpy as np




def test_triggered_telescopes():
    from pschitt import geometry as geo

    tel1 = geo.Telescope([10, 0, 0], [-1. / 3., 0, 2. / 3.])
    tel2 = geo.Telescope([0, 10, 0], [-1. / 3., 0, 2. / 3.])
    tel1.signal_hist = np.ones(len(tel1.pixel_tab))
    tel2.signal_hist = np.ones(len(tel2.pixel_tab)) * 3
    threshold = 2 * len(tel1.pixel_tab)

    assert ana.triggered_telescopes([tel1, tel2], threshold) == [tel2]


def test_multiplicity():
    from pschitt import geometry as geo

    tel1 = geo.Telescope([10, 0, 0], [-1. / 3., 0, 2. / 3.])
    tel2 = geo.Telescope([0, 10, 0], [-1. / 3., 0, 2. / 3.])
    tel1.signal_hist = np.ones(len(tel1.pixel_tab))
    tel2.signal_hist = np.ones(len(tel2.pixel_tab)) * 3
    threshold = 2 * len(tel1.pixel_tab)

    assert ana.multiplicity([tel1, tel2], threshold) == 1

