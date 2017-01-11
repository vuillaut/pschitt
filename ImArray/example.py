"""
ImArray is a software to image an object in the sky by an array of ground Telescopes
Copyright (C) 2016  Thomas Vuillaume

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>

The author may be contacted @
thomas.vuillaume@lapp.in2p3.fr
"""


import geometry as geo
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle, PathPatch
# register Axes3D class with matplotlib by importing Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
import time
import math
import object as sh
import CameraImage as ci
import Hillas as hillas
import os

MEAN = np.empty(0)


#If you want some plots to be done
BoolPlot = True
noise = 0


'''
Shower parameters
'''
impact_point = np.array([0,250,0])

salt = math.radians(74)
saz = math.radians(0)
talt = math.radians(74)
taz = math.radians(0)

s_top = [0,0,15000]
s_bot = [0,0,1000]
s_center = [0,0,8000]
slength = 12000
swidth = 200

npoints = 500

#shower = np.array(sh.linear_segment(s_top,s_bot,40))
#shower = np.array(sh.random_ellipsoide_alongz(s_center, slength, swidth, npoints))
#shower = np.array(sh.shifted_ellipsoide_v1(s_center, slength, swidth, npoints, impact_point, s_top[2]))
#shower = sh.random_ellipsoide(15000, slength, swidth, salt, saz, impact_point, npoints)
#shower2 = sh.random_ellipsoide(15000, slength, swidth, math.pi/2., 0, [0,0,0], npoints)
#shower3 = sh.random_ellipsoide(15000, slength, swidth, salt, math.pi/2., [0,0,0], npoints)


'''
Load a telescope configuration
'''
#tels = geo.load_telescopes("telescopes.dat")
#alltel = tels

#tel_normal = [0.0,0.0,1.]
tel_normal = geo.altaz_to_normal(talt,taz)
'''
tel1 = geo.Telescope([500,100,-100],tel_normal,0)
tel2 = geo.Telescope([-10,300,100],tel_normal,0)
tel3 = geo.Telescope([-500,400,10],tel_normal,0)
tel4 = geo.Telescope([-350,-500,0],tel_normal,0)
tel5 = geo.Telescope([600,-100,150],tel_normal,0)
tel6 = geo.Telescope([300,-300,50],tel_normal,0)
tel10 = geo.Telescope([80,0,0],tel_normal,0)
tel11 = geo.Telescope([-80,0,0],tel_normal,0)
tel12 = geo.Telescope([0,-80,0],tel_normal,0)
tel13 = geo.Telescope([0,80,0],tel_normal,0)

alltel = [tel1,tel2,tel3,tel4,tel5,tel6]
'''


#[0,0.001,0.005,0.01,0.02]
#[  3.72336726, 7.27775374, 21.79562501, 54.48460046, 22.48430215]

def plot_shower3d(shower,alltel):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(shower[:,0] , shower[:,1], shower[:,2], c='r', marker='o')
    for tel in alltel:
        p = Circle((tel.center[0],tel.center[1]), 30)
        ax.add_patch(p)
        art3d.pathpatch_2d_to_3d(p, z=tel.center[2], zdir="z")
    plt.axis([-1000, 1000, -1000, 1000])
    plt.show()

#for dev in [0.02,0.03,0.05]:
for dev in [0.01]:
    ERR = np.empty(0)
    for n in range(1):
        shower = sh.random_ellipsoide(15000, slength, swidth, salt, saz, impact_point, npoints)

    #dev = 0.01

        tel1 = geo.Telescope([200,100,0],geo.altaz_to_normal(talt,taz),0)
        tel2 = geo.Telescope([-100,0,0],geo.altaz_to_normal(talt,taz+dev),0)
        tel3 = geo.Telescope([0,-100,0],geo.altaz_to_normal(talt+dev,taz),0)
        tel4 = geo.Telescope([100,0,0],geo.altaz_to_normal(talt,taz-dev),0)
        tel5 = geo.Telescope([0,100,0],geo.altaz_to_normal(talt-dev,taz),0)
        tel6 = geo.Telescope([300,-300,0], geo.altaz_to_normal(talt,taz),0)


        #alltel = [tel1,tel2,tel3,tel4,tel5,tel6]
        #alltel = geo.load_telescopes("3HB1-NG-tels.txt")
        #alltel = geo.load_telescopes("data/tel_pos.dat")
        #alltel = [tel2, tel6]
        alltel = [tel1,tel2,tel5, tel6]
        #alltel = geo.load_telescopes_flatfloor("data/UniquePositionTelescope.txt", tel_normal)
        #alltel = [tel1, tel2, tel3, tel4,tel5,tel6,tel10,tel11,tel12,tel13]
        for tel1 in alltel:
            for tel2 in alltel:
                if((tel1.center==tel2.center).all() and tel1.id!=tel2.id):
                    print(tel1.id, tel2.id, tel1.center, tel2.center)

        if BoolPlot:
            plot_shower3d(shower,alltel)


        #sys.exit()

        IM = []
        coord = []
        vis = []

        plt.xkcd()

        start = time.time()

        if not os.path.isdir('results'):
            os.mkdir('results')
        f = open('results/teldata.xml','w')

        f.write("\n")
        f.write("<subarray>\n")

        #SIZE SHOULD CHANGE WITH CAMERA TYPE
        allhist = np.zeros(1855)
        HillasParameters = []

        for tel in alltel:
            X = []
            Y = []
            for point in shower:
                im = geo.image_point_pfo(point, tel)
                IM.append(im)
                center_camera = geo.camera_center(tel)
                if geo.is_point_in_camera_plane(im, tel):
                    vis.append(geo.is_point_visible(im,tel))
                im_cam = geo.site_to_camera_cartesian([im[0],im[1],im[2]], tel)
                X.append(im_cam[0])
                Y.append(im_cam[1])
            #hist = ci.camera_image(tel, np.column_stack((X,Y)), filename="results/{}.txt".format(tel.id))
            hist = ci.camera_image(tel, np.column_stack((X,Y)), "results/{}.txt".format(tel.id), noise)
            print(hist)
            hp = hillas.hillas_parameters_2(hist[:,0], hist[:,1], hist[:,2])
            hp.append(tel.id)
            #[size, cen_x, cen_y, length, width, r, phi, psi, miss] =   hp

            HillasParameters.append(hp)

            allhist += hist[:,2]
            alt,az = geo.normale_to_altaz(tel.normale)
            f.write('<telescope telId="%d" position="%f,%f,%f" dirAlt="%f" dirAz="%f" focal="%f">\n' % (tel.id,tel.center[0],tel.center[1],tel.center[2],alt,az,tel.focale))
            for xyph in hist:
                f.write('%f,%f,%f;\n' % (xyph[0],xyph[1],xyph[2]))
            f.write('</telescope>\n')

            if BoolPlot:
                plt.plot(X,Y,'o', label = tel.center,markersize=3)


        pa = hillas.impact_parameter_average(alltel, HillasParameters, alt, az)
        p = hillas.impact_parameter_ponderated(alltel, HillasParameters, alt, az)
        print("Real impact parameter : ", impact_point)
        print("Reco with simple average = %s \tError = %.2fm" % (pa, math.sqrt(((impact_point-pa)**2).sum())))
        print("Reco with ponderation and cut = %s \tError = %.2fm" % (p, math.sqrt(((impact_point-p)**2).sum())))
        ERR = np.append(ERR, math.sqrt(((impact_point-p)**2).sum()))
        #print(ERR)
        np.savetxt('results/hillas.txt',HillasParameters,fmt='%.5f',
            header="size\t<x>\t<y>\tl\tw\tr\tphi\tpsi\tmiss\tTelId")

        hist[:,2] = allhist
        np.savetxt('results/all.txt',hist,fmt='%.5f')


        #print('\nElapsed (ms):', (time.time() - start) * 1000)

        f.write('\n</subarray>')
        f.close()


        if BoolPlot:
            plt.legend(loc='upper center', fancybox=True, ncol=3, bbox_to_anchor=(0.5,1.1))
            plt.show()
    print(ERR)
    MEAN = np.append(MEAN, ERR.mean())
print(MEAN)
#plt.plot([0,0.001,0.005,0.01,0.02], MEAN)
plt.show()
