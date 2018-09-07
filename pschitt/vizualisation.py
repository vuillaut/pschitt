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

    colors = ['#8dd3c7', '#fb8072', '#ffffb3', '#bebada', '#80b1d3', '#fdb462', '#b3de69']
    if 'c' not in kwargs and 'color' not in kwargs:
        change_color = True
    else:
        change_color = False

    for tel in telescope_array:
        if change_color:
            kwargs['color'] = colors[int(tel.camera_type)]
        ax.scatter(tel.mirror_center[0], tel.mirror_center[1], **kwargs)
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


def plot_hillas_ground_direction(telescopes, hillas_parameters, ax=None, **kwargs):
    """
    Plot the reconstructed directions on the ground from Hillas parameters

    Parameters
    ----------
    telescopes: list of Telescope class
    hillas_parameters: list of Hillas parameters for the given telescopes
    ax: `matplotlib.pyplot.axes`
    kwargs: args for `matplotlib.pyplot.plot`

    Returns
    -------
    `matplotlib.pyplot.axes`
    """
    ax = plt.gca() if ax is None else ax

    for (tel, hp) in zip(telescopes, hillas_parameters):
        alt, az = geo.normal_to_altaz(tel.normal)
        psi_g = geo.direction_ground(hp[7] + np.pi / 2., alt, az)
        ax.plot(x * np.cos(psi_g) + tel.mirror_center[0], x * np.sin(psi_g) + tel.mirror_center[1], **kwargs)

    return ax


def ground_intensity(position, shower):
    """
    Compute the ground intensity at a given position as the sum of each particle intensity given by the
    `shower.particles_angular_emission_profile`.

    Parameters
    ----------
    position: `numpy.ndarray` of shape (3,)
    shower: shower class

    Returns
    -------
    float
    """
    angles = geo.angles_to_particles(position, shower)
    return shower.particles_angular_emission_profile(angles, **shower.particles_angular_emission_profile_kwargs).sum()


def intensity_map(shower, x=np.linspace(-2000, 2000), y=np.linspace(-2000, 2000)):
    """
    Intensity map of the shower Cherenkov light on the ground

    Parameters
    ----------
    shower: shower class
    x: `numpy.ndarray` of shape (n,)
    y: `numpy.ndarray` of shape (n,)

    Returns
    -------
    `numpy.ndarray` of shape (n,n)
    """
    intensity_map = np.empty((x.size, y.size))

    for i in range(x.size):
        for j in range(y.size):
            position = np.array([x[i], y[j], 0])
            intensity_map[i, j] = ground_intensity(position, shower)
    return intensity_map


def plot_intensity_map(x_grid, y_grid, intensity_map, ax=None, **kwargs):
    """
    Plot an intensity map

    Parameters
    ----------
    x_grid: `numpy.ndarray` of shape (n,m)
    y_grid: `numpy.ndarray` of shape (n,m)
    intensity_map: `numpy.ndarray` of shape (m,n)
    ax: `matplotlib.pyplot.axes`
    kwargs: kwargs for `matplotlib.pyplot.contourf`

    Returns
    -------
    `matplotlib.pyplot.axes`
    """
    ax = plt.gca() if ax is None else ax
    if 'cmap' not in kwargs:
        kwargs['cmap'] = plt.get_cmap('Blues')

    ax.contourf(x_grid, y_grid, intensity_map, **kwargs)

    return ax


def plot_shower_ground_intensity_map(shower, x=np.linspace(-2000, 2000), y=np.linspace(-2000, 2000), ax=None, **kwargs):
    """
    Plot the intensity map of the shower Cherenkov light on the ground

    Parameters
    ----------
    shower: shower class
    x: `numpy.ndarray`
    y: `numpy.ndarray`
    ax: `matplotlib.pyplot.axes`
    kwargs: kwargs for `plot_intensity_map`

    Returns
    -------
    `matplotlib.pyplot.axes`
    """

    ax = plt.gca() if ax is None else ax
    x_grid, y_grid = np.meshgrid(x, y)
    intensity_map = intensity_map(shower, x, y)
    ax = plot_intensity_map(x_grid, y_grid, intensity_map, ax=ax, **kwargs)

    return ax
