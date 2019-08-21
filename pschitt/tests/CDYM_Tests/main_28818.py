"""
This script is for running the M-P simulation.
"""
## Import Required Packages ##
import pschitt.geometry as geo
import numpy as np
import matplotlib.pyplot as plt
import pschitt.sky_objects as sky
import pschitt.camera_image as ci
import pschitt.hillas as hillas
import pschitt.vizualisation as viz
#from importlib import reload
import math

"""
------------------------
SHOWER SETUP: Edit the variables to change the shape of the air shower.
------------------------

"""

## Creating a Shower ##
# Activate the shower class - don't change this #
shower = sky.shower()

# X, Y, Z coordinates of impact - VARIABLE [meters] (x & y range: -1000 to 1000) #
shower.impact_point = np.array([80,60,0])

# Energy of Primary Photon [eV] - VARLIBLE [Electron Volts] #
shower.energy_primary = 300e9

# Defining shower direction - VARIABLE [Altitude, Azimuth] #
shower.alt = math.radians(90)
shower.az = math.radians(0)

# Defining height of first interaction - VARIABLE [meters] #
shower.height_of_first_interaction = 25000

# Defining resolution of the shower - VARIABLE [g cm^-2] #
shower.scale = 10

# Defining longitudinal shower shape - don't change this #
shower.Greisen_Profile()

"""
------------------------
END OF SHOWER SETUP
------------------------
"""

"""
------------------------
TELESCOPE ARRAY SETUP - Change the number of telescopes, their types, their locations (coordinates in meters from +1000 to -1000) and the array pointing direction.
------------------------
"""

## Creating the telescope array ##
# Pointing direction - VARIABLE [Altituze, Azimuth] #
talt = math.radians(90)
taz = math.radians(0)

# Telescope setup - VARIABLE - Types you can choose: default, gct, dc, astri, sct, flash, nectar and lst_cam #
tel_normal = geo.altaz_to_normal(talt, taz)
tel1 = geo.Telescope([200,200,0], tel_normal, camera_type='0')
tel2 = geo.Telescope([-200,200,0], tel_normal, camera_type='0')
tel3 = geo.Telescope([-200,-200,0], tel_normal, camera_type='0')
tel4 = geo.Telescope([200,-200,0], tel_normal, camera_type='0')
tel5 = geo.Telescope([0,0,0], tel_normal, camera_type='0')
tel6 = geo.Telescope([1000,-500,0], tel_normal, camera_type='0')

# Creating a list of the telescopes - VARIABLE - You need to add all the above telescope IDs to this list (or only the telescopes you want to simulate) #
alltel = [tel1, tel2, tel3, tel4, tel5, tel6]


"""
------------------------
END OF TELESCOPE ARRAY SETUP
------------------------

Beyond this point, you should only need to change the test number in the output file names. The file names will automatically correspond to the options you selected above for the energy, initial height, scale and number of telescopes.


The quickest way to change the output file name for each test run is to 'find and replace, replace all' "T1" with "T2" etc (test 1, test 2 etc...).
"""


## Imaging the shower with the array
# For naming #
Eig = int(shower.energy_primary / 1e9)
Eit = int(shower.energy_primary / 1e12)
hfi = int(shower.height_of_first_interaction / 1000)
sca = int(shower.scale)
ntel = int(len(alltel))

if Eig >= 1000:
	ShEn = Eit
else:
	ShEn = Eig

# Plotting the system arrangement with the shower
plt.figure(figsize=(9,6))
viz.plot_shower3d(shower, alltel, density_color=True)
if Eig >= 1000:
	plt.title("3d Shower Plot (%i TeV, %i km, Scale: %i, NumTel: %i)" %(Eit, hfi, sca, ntel))
	plt.savefig('shower3D_%i_TeV_%i_km_scale_%i_num_%i_T1.png' %(Eit, hfi, sca, ntel), fmt='png', dpi=300)
else:
	plt.title("3d Shower Plot (%i GeV, %i km, Scale: %i, NumTel: %i)" % (Eig, hfi, sca, ntel))
	plt.savefig('shower3D_%i_GeV_%i_km_scale_%i_num_%i_T1.png' %(Eig, hfi, sca, ntel), fmt='png', dpi=300)
#plt.savefig('shower3D_%i_GeV_%i_km_scale_%i_num_%i_T1.png' %(Eig, hfi, sca, ntel), fmt='png', dpi=300) # Normal file type = eps
plt.show()
plt.clf() # This clears the plot environment to allow for further plotting

# Projection
plt.scatter(shower.particles[:,1], shower.particles[:,2], marker=".")
plt.axis('equal')
if Eig >= 1000:
	plt.title("Shower Projection (%i TeV, %i km, Scale: %i, NumTel: %i)" %(Eit, hfi, sca, ntel))
	plt.savefig('projection_%i_TeV_%i_km_scale_%i_num_%i_T1.eps' %(Eit, hfi, sca, ntel), fmt='eps')
else:
	plt.title("Shower Projection (%i GeV, %i km, Scale: %i, NumTel: %i)" %(Eig, hfi, sca, ntel))
	plt.savefig('projection_%i_GeV_%i_km_scale_%i_num_%i_T1.eps' %(Eig, hfi, sca, ntel), fmt='eps')
#plt.savefig('projection_%i_GeV_%i_km_scale_%i_num_%i_T1.eps' %(Eig, hfi, sca, ntel), fmt='eps')
plt.show()
plt.clf()

## Photosphere distribution along each axis
# X axis
plt.hist(shower.particles[:,0], bins=50);
plt.xlabel("X [m]")
if Eig >= 1000:
	plt.title("Particle Distribution in x (%i TeV, %i km, Scale: %i, NumTel: %i)" %(Eit, hfi, sca, ntel))
	plt.savefig('x_distrib_%i_TeV_%i_km_scale_%i_num_%i_T1.png' %(Eit, hfi, sca, ntel), fmt='png')
else:
	plt.title("Particle Distribution in x (%i GeV, %i km, Scale: %i, NumTel: %i)" %(Eig, hfi, sca, ntel))
	plt.savefig('x_distrib_%i_GeV_%i_km_scale_%i_num_%i_T1.png' %(Eig, hfi, sca, ntel), fmt='png')
#plt.savefig('x_distrib_%i_GeV_%i_km_scale_%i_num_%i_T1.png' %(Eig, hfi, sca, ntel), fmt='png')
plt.show()
plt.clf()

# Y axis
plt.hist(shower.particles[:,1], bins=50);
plt.xlabel("Y [m]")
if Eig >= 1000:
	plt.title("Particle Distribution in y (%i TeV, %i km, Scale: %i, NumTel: %i)" %(Eit, hfi, sca, ntel))
	plt.savefig('y_distrib_%i_TeV_%i_km_scale_%i_num_%i_T1.png' %(Eit, hfi, sca, ntel), fmt='png')
else:
	plt.title("Particle Distribution in y (%i GeV, %i km, Scale: %i, NumTel: %i)" %(Eig, hfi, sca, ntel))
	plt.savefig('y_distrib_%i_GeV_%i_km_scale_%i_num_%i_T1.png' %(Eig, hfi, sca, ntel), fmt='png')
#plt.savefig('y_distrib_%i_GeV_%i_km_scale_%i_num_%i_T1.png' %(Eig, hfi, sca, ntel), fmt='png')
plt.show()
plt.clf()

# Z axis

plt.hist(shower.particles[:,2], bins=50);
plt.xlabel("Z [m]")
if Eig >= 1000:
	plt.title("Particle Distribution in z (%i TeV, %i km, Scale: %i, NumTel: %i)" %(Eit, hfi, sca, ntel))
	plt.savefig('z_distrib_%i_TeV_%i_km_scale_%i_num_%i_T1.png' %(Eit, hfi, sca, ntel), fmt='png')
else:
	plt.title("Particle Distribution in z (%i GeV, %i km, Scale: %i, NumTel: %i)" %(Eig, hfi, sca, ntel))
	plt.savefig('z_distrib_%i_GeV_%i_km_scale_%i_num_%i_T1.png' %(Eig, hfi, sca, ntel), fmt='png')
#plt.savefig('z_distrib_%i_GeV_%i_km_scale_%i_num_%i_T1.png' %(Eig, hfi, sca, ntel), fmt='png')
plt.show()
plt.clf()

"""
------------------------
BEGINNING OF IMAGE RECONSTRUCTION
------------------------
"""

## Making a map of the site
plt.figure() # Wrapping this in the plot environment allows for the plot to be saved
ax = viz.plot_array(alltel)
ax.scatter(shower.impact_point[0], shower.impact_point[1], color='black', label='Impact Point', marker='+')
ax.legend()
if Eig >= 1000:
	plt.title("Map of Array Site (%i TeV, %i km, Scale: %i, NumTel: %i)" %(Eit, hfi, sca, ntel))
	plt.savefig('rep_Array_map_%i_TeV_%i_km_scale_%i_num_%i_T1.png' %(Eit, hfi, sca, ntel), fmt='png')
else:
	plt.title("Map of Array Site (%i GeV, %i km, Scale: %i, NumTel: %i)" %(Eig, hfi, sca, ntel))
	plt.savefig('rep_Array_map_%i_GeV_%i_km_scale_%i_num_%i_T1.png' %(Eig, hfi, sca, ntel), fmt='png')
#plt.savefig('rep_Array_map_%i_GeV_%i_km_scale_%i_num_%i_T1.png' %(Eig, hfi, sca, ntel), fmt='png')
plt.clf()

## Producing Images in Telescopes
trigger_intensity = 20.
noise = 0
ci.array_shower_imaging(shower, alltel, noise)
fig, axes = plt.subplots(1, len(alltel), figsize=(20,3))
plt.tight_layout()
for tel, ax in zip(alltel, axes):
    ax = viz.display_camera_image(tel, ax=ax, s=6)
    ax.set_title("Tel {0}\nSignal sum = {1}".format(tel.id, tel.signal_hist.sum()))
if Eig >= 1000:
	plt.savefig('scopes_%i_TeV_%i_km_scale_%i_num_%i_T1.png' %(Eit, hfi, sca, ntel), fmt='png')
else:
	plt.savefig('scopes_%i_GeV_%i_km_scale_%i_num_%i_T1.png' %(Eig, hfi, sca, ntel), fmt='png')
#plt.savefig('scopes_%i_GeV_%i_km_scale_%i_num_%i_T1.png' %(Eig, hfi, sca, ntel), fmt='png') #normal format = eps

## Hillas Reconstruction
HP, triggered_telescopes = hillas.array_hillas_parameters(alltel, trigger_intensity)
print("Number of triggered telescopes =", len(triggered_telescopes))

if len(triggered_telescopes)>1:
    pa = hillas.impact_parameter_average(triggered_telescopes, HP)
    p = hillas.impact_parameter_ponderated(triggered_telescopes, HP)

plt.figure()
viz.display_stacked_cameras(alltel, s=14)
if Eig >= 1000:
	plt.title("Stack of Camera Images (%i TeV, %i km, Scale: %i, NumTel: %i)" %(Eit, hfi, sca, ntel))
	plt.savefig('Stacked_cameras_%i_TeV_%i_km_scale_%i_num_%i_T1.eps' %(Eit, hfi, sca, ntel), fmt='eps')
else:
	plt.title("Stack of Camera Images (%i GeV, %i km, Scale: %i, NumTel: %i)" %(Eig, hfi, sca, ntel))
	plt.savefig('Stacked_cameras_%i_GeV_%i_km_scale_%i_num_%i_T1.eps' %(Eig, hfi, sca, ntel), fmt='eps')
#plt.savefig('Stacked_cameras_%i_GeV_%i_km_scale_%i_num_%i_T1.eps' %(Eig, hfi, sca, ntel), fmt='eps') #normal format = eps

## Superimposed Hillas directions
plt.figure(figsize=(7,7))
ax = viz.display_stacked_cameras(alltel, s=27)
x = np.linspace(-0.5,0.5)
for tel, hp in zip(triggered_telescopes, HP):
    ax.plot(hp[1] + x*np.cos(hp[7] + math.pi/2.), hp[2] + x*np.sin(hp[7] + math.pi/2.), color="white")
if Eig >= 1000:
	plt.title("Hillas Reconstruction (%i TeV, %i km, Scale: %i, NumTel: %i)" %(Eit, hfi, sca, ntel))
	plt.savefig('Hillas_%i_TeV_%i_km_scale_%i_num_%i_T1.png' %(Eit, hfi, sca, ntel), fmt='png', dpi=300)
else:
	plt.title("Hillas Reconstruction (%i GeV, %i km, Scale: %i, NumTel: %i)" %(Eig, hfi, sca, ntel))
	plt.savefig('Hillas_%i_GeV_%i_km_scale_%i_num_%i_T1.png' %(Eig, hfi, sca, ntel), fmt='png', dpi=300)
#plt.savefig('Hillas_%i_GeV_%i_km_scale_%i_num_%i_T1.png' %(Eig, hfi, sca, ntel), fmt='png', dpi=300) #normal format = eps

## On Site Directions
# Plotting a figure of the impact direction on the array map
plt.figure()
viz.plot_array_reconstructed(triggered_telescopes, HP, shower.impact_point)
if Eig >= 1000:
	plt.title("Reconstructed Array Directions (%i TeV, %i km, Scale: %i, NumTel: %i)" %(Eit, hfi, sca, ntel))
	plt.savefig('Array_direct_%i_TeV_%i_km_scale_%i_num_%i_T1.eps' %(Eit, hfi, sca, ntel), fmt='eps')
else:
	plt.title("Reconstructed Array Directions (%i GeV, %i km, Scale: %i, NumTel: %i)" %(Eig, hfi, sca, ntel))
	plt.savefig('Array_direct_%i_GeV_%i_km_scale_%i_num_%i_T1.eps' %(Eig, hfi, sca, ntel), fmt='eps')
#plt.savefig('Array_direct_%i_GeV_%i_km_scale_%i_num_%i_T1.eps' %(Eig, hfi, sca, ntel), fmt='eps')

# Calculating the location of the impact point using the reconstructed information
if len(triggered_telescopes)>1:
    pa = hillas.impact_parameter_average(triggered_telescopes, HP)
    p = hillas.impact_parameter_ponderated(triggered_telescopes, HP)
if len(triggered_telescopes)>1:
    print("Real impact parameter :", shower.impact_point)
    print("Reconstructed with simple average = %s \tError = %.2fm" % (pa, math.sqrt(((shower.impact_point-pa)**2).sum())))
    print("Reconstructed with ponderation and cut = %s \tError = %.2fm" % (p, math.sqrt(((shower.impact_point-p)**2).sum())))

# Plotting Lightpool
plt.clf()

plt.figure()
viz.plot_shower_ground_intensity_map(shower, x=np.linspace(-2000, 2000), y=np.linspace(-2000, 2000), ax=None)
plt.show()
plt.clf()


