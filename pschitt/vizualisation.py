# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d
from scipy import stats
from . import geometry as geo


def plot_shower3d(shower, alltel, **options):
    """
    Display the sky object (shower) and the telescope in a 3D representation
    Parameters
    ----------
    shower: array of points (arrays [x,y,z])
    alltel: array of telescopes (telescope class)
    options:
        - density_color = True: use density for particles color. False by default.
        - display = True: show the plot. False by default
        - outfile = "file.eps" : save the plot as `file.eps`. False by default.
    """
    if options.get("figsize"):
        figsize = options.get("figsize")
    else:
        figsize=(12,12)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    for tel in alltel:
        p = plt.Circle((tel.mirror_center[0],tel.mirror_center[1]), 30, color='black')
        ax.add_patch(p)
        art3d.pathpatch_2d_to_3d(p, z=tel.mirror_center[2], zdir='z')

    ip = plt.Circle((shower.impact_point[0], shower.impact_point[1]), 15, color='red')
    ax.add_patch(ip)
    art3d.pathpatch_2d_to_3d(ip, z=tel.mirror_center[2], zdir='z')

    values = shower.particles.T

    if options.get("density_color") == True:
        kde = stats.gaussian_kde(values)
        density = kde(values)
        ax.scatter(values[0] , values[1], values[2], marker='o', c=density)
    else:
        ax.scatter(values[0] , values[1], values[2], marker='o')

    plt.axis([-1000, 1000, -1000, 1000])
    ax.set_xlabel("[m]")
    ax.set_zlabel("altitude [m]")
    if options.get("display") == True:
        plt.show()
    if options.get("outfile"):
        outfile = options.get("outfile")
        assert isinstance(outfile, str), "The given outfile option should be a string"
        plt.savefig(outfile + '.eps', format='eps', dpi=200)


def display_camera_image(telescope, ax=None, **kwargs):
    """
    display an image of the camera of the telescope
    Parameters
    ----------
    telescope : telescope class
    histogram : histogram of the signal in each pixel
    """

    ax = plt.gca() if ax is None else ax

    if not 's' in kwargs:
        kwargs['s'] = 27
    if not 'c' in kwargs:
        kwargs['c'] = telescope.signal_hist

    fig = ax.scatter(telescope.pixel_tab[:, 0], telescope.pixel_tab[:, 1], **kwargs)
    ax.axis('equal')
    plt.colorbar(fig, ax=ax, label='counts')

    return ax


def display_stacked_cameras(telescope_array, ax=None, **kwargs):
    """
    Display stacked camera images. This only works if all the telescope camera are the same type.
    Parameters
    ----------
    telescope_array: list of telescopes classes
    """

    ax = plt.gca() if ax is None else ax

    tel0 = telescope_array[0]
    l0 = len(tel0.signal_hist)
    assert np.all([len(tel.signal_hist) == l0 for tel in telescope_array]), \
        "Impossible to stack cameras with different shapes"

    stacked_hist = np.zeros(l0)

    for tel in telescope_array:
        stacked_hist += tel.signal_hist

    if not 'c' in kwargs:
        kwargs['c'] = stacked_hist

    fig = ax.scatter(tel0.pixel_tab[:, 0], tel0.pixel_tab[:, 1], **kwargs)
    ax.axis('equal')
    plt.colorbar(fig, label='counts', ax=ax)

    return ax



def display_pointing_tel(tel, show=True):
    """
    Display the pointing direction of a telescope as seen from above
    Parameters
    ----------
    tel: telescope class
    show: Boolean, to display or not
    """
    ax = plt.axes()
    ax.arrow(tel.camera_center[0], tel.camera_center[1], tel.normal[0], tel.normal[1], head_width=5, head_length=10, fc='k', ec='k')
    #plt.xlim([np.min([start[0], end[0]]), np.max([start[0], end[0]])])
    #plt.ylim([np.min([start[1], end[1]]), np.max([start[1], end[1]])])
    if show:
        plt.show()


def display_pointing_array(alltel):
    """
    Display the pointing direction of each telescope in the array as seen from above

    Parameters
    ----------
    alltel: list of telescope classes
    """
    centers = np.array([tel.mirror_center for tel in alltel])
    xmin = centers[:,0].min() - 50
    xmax = centers[:,0].max() + 50
    ymin = centers[:,1].min() - 50
    ymax = centers[:,1].max() + 50

    for tel in alltel:
        display_pointing_tel(tel, show=False)

    plt.axis('equal')
    plt.axis([xmin, xmax, ymin, ymax])
    plt.show()


def plot_array(telescope_array, ax=None, **kwargs):
    """
    Plot a map of the telescopes array

    Parameters
    ----------
    telescope_array: list of telescopes classes
    """

    ax = plt.gca() if ax is None else ax

    for tel in telescope_array:
        ax.scatter(tel.mirror_center[0], tel.mirror_center[1])
        ax.annotate(str(tel.id), (tel.mirror_center[0] + 20, tel.mirror_center[1] + 20))
    ax.axis('equal')
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    return ax


def plot_array_reconstructed(triggered_telescopes, hillas_parameters, impact_point, ax=None, **kwargs):


    ax = plt.gca() if ax is None else ax

    ax = plot_array(triggered_telescopes, ax=ax)

    x = np.linspace(-300, 300)

    for (tel, hp) in zip(triggered_telescopes, hillas_parameters):
        alt, az = geo.normal_to_altaz(tel.normal)
        psi_g = geo.direction_ground(hp[7] + np.pi/2., alt, az)
        ax.plot(x * np.cos(psi_g) + tel.mirror_center[0], x * np.sin(psi_g) + tel.mirror_center[1], **kwargs)

    ax.scatter(impact_point[0], impact_point[1], color='black', label='Impact point', marker='X', s=80)

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.axis('equal')
    ax.legend()

    return ax


def plot_angular_emission_profile(emission_profile, *em_args, **plot_kwargs):
    """
    Plot an angular emission profile.

    Parameters
    ----------
    emission_profile: `emission.angular_profile`
    em_args: args for the emission profile
    plot_kwargs:
        - args for `matplotlib.pyplot.plot`
        - ax: `matplotlib.pyplot.axes`. default = None
        - angle_max: `float` - max angle for the plot. default = 2

    Returns
    -------
    `matplot.pyplot.axes`
    """

    amax = plot_kwargs.pop('angle_max') if 'angle_max' in plot_kwargs else np.pi

    angles = np.linspace(0, amax, 100)
    emission = emission_profile(angles, *em_args)

    ax = plt.gca() if not 'ax' in plot_kwargs else plot_kwargs.pop('ax')

    ax.plot(angles, emission, **plot_kwargs)

    return ax