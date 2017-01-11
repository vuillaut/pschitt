import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d



def plot_shower3d(shower,alltel):
    """
    Display the sky object (shower) and the telescope in a 3D representation
    Parameters
    ----------
    shower : array of points [x,y,z] (array)
    alltel : array of telescopes (telescope class)
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(shower[:,0] , shower[:,1], shower[:,2], c='r', marker='o')
    for tel in alltel:
        p = plt.Circle((tel.center[0],tel.center[1]), 30)
        ax.add_patch(p)
        art3d.pathpatch_2d_to_3d(p, z=tel.center[2], zdir="z")
    plt.axis([-1000, 1000, -1000, 1000])
    plt.show()
