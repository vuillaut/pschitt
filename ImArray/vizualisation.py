import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d
from mpl_toolkits.mplot3d import Axes3D


def plot_shower3d(shower,alltel):
    """
    Display the sky object (shower) and the telescope in a 3D representation
    Parameters
    ----------
    shower: array of points (arrays [x,y,z])
    alltel: array of telescopes (telescope class)
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(shower[:,0] , shower[:,1], shower[:,2], c='r', marker='o')
    for tel in alltel:
        p = plt.Circle((tel.center[0],tel.center[1]), 30)
        ax.add_patch(p)
        art3d.pathpatch_2d_to_3d(p, z=tel.center[2], zdir='z')
    plt.axis([-1000, 1000, -1000, 1000])
    plt.show()


def display_camera_image(telescope, histogram):
    """
    display an image of the camera of the telescope
    Parameters
    ----------
    telescope : telescope class
    histogram : histogram of the signal in each pixel
    """
    X, Y = np.loadtxt(telescope.pixpos_filename, unpack = True)
    plt.scatter(X,Y,c=histogram)
    plt.axis('equal')
    plt.show()
    return X,Y,histogram


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
    centers = np.array([tel.center for tel in alltel])
    xmin = centers[:,0].min() - 50
    xmax = centers[:,0].max() + 50
    ymin = centers[:,1].min() - 50
    ymax = centers[:,1].max() + 50

    for tel in alltel:
        display_pointing_tel(tel, show=False)

    plt.axis('equal')
    plt.axis([xmin,xmax,ymin,ymax])
    plt.show()


def plot_cherenkov_cone():
    """
    plot a 3d cherenkov cone given a direction
    Returns
    -------
    """