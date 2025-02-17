#%% Imports 
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import arim
import os
from scipy.spatial.transform import Rotation
import time
from scipy.io import loadmat

import arim.geometry as g
import sys
sys.path.append('functions_scanning')
import arim.ray, arim.io, arim.signal, arim.im
import arim.models.block_in_immersion as bim
import arim.plot as aplt
from Functions_PWI import fn_PlaneWaveIntersections

def fn_SurfaceToWall(surfaces,wall_points_per_mm,names=['Frontwall']):
    Walls = {}
    for ss,s in enumerate(surfaces):
        numpoints = []
        #Calc numpoints for each line in surface
        for ii in range(s.shape[0]-1):
            dist = np.sqrt(np.square(s[ii+1,0]-s[ii,0]) + np.square(s[ii+1,1]-s[ii,1]))
            numpoints.append( int(np.ceil(dist*1e3*wall_points_per_mm)) )
        new_wall = arim.geometry.make_contiguous_geometry(s, int(np.ceil(dist*1e3*wall_points_per_mm)))
        new_wall = arim.geometry.combine_oriented_points(list(new_wall.values()), name=names[ss])
        Walls[names[ss]] = new_wall
    return Walls

#%% Inspection inputs

## Probe
conf_name = 'conf.yaml'
conf = arim.io.load_conf_file(conf_name)
array_width = conf['probe']['pitch_x'] * (conf['probe']['numx']-1)
array_rot = np.array([0,0,0])

## Imaging mesh
imaging_res = 0.2e-3
imaging_lims = np.array([[-array_width/2-15e-3,0],[array_width/2+15e-3,50e-3]])
wall_points_per_mm = 10#8

## Sample
standoff = 15e-3
thickness = 30e-3

#%% Surfaces
surface_point_n = 3
curvature_height = 0

x1 = np.linspace(-35e-3,35e-3,surface_point_n)

z1 =  curvature_height*np.sin((x1-x1.min())/(x1.max()-x1.min())*np.pi)+standoff - curvature_height*0.5

x2 = x1.copy()
z2 =  np.ones([len(x2)])*(standoff+thickness)
s1 = np.stack([x1,z1],1)
s2 = np.stack([x2,z2],1)

#%% Objects

#Probe
Probe = arim.io.probe_from_conf(conf)
#Match with new syntax:
"""array_rot_orig = np.copy(array_rot)
array_rot[2] = array_rot[2] + 180
array_rot[1] = array_rot[1] + 90
array_rot[0] = array_rot[0] + 90
Probe.flip_probe_around_axis_Oz()"""

rot_mat = Rotation.from_euler('xyz', array_rot, degrees=True).as_matrix()
Probe.rotate(rot_mat,[0,0,0])
n_els = Probe.numelements

    


##Walls 
wall_names = ['Frontwall','Backwall']
Walls = fn_SurfaceToWall([s1,s2],wall_points_per_mm,wall_names)

##Examination
Examination = arim.core.BlockInImmersion(arim.io.material_from_conf(conf['block_material']), arim.io.material_from_conf(conf['couplant_material']), Walls, [0,1])
     
##Grid
Grid = arim.geometry.Grid(
    xmin = x1[0]-5e-3,
    xmax = x1[-1]+5e-3,
    ymin = 0.0,
    ymax = 0.0,
    zmin = 0,
    zmax = z2[0],
    pixel_size = imaging_res,
)
    

#Paths
interface_dict = bim.make_interfaces(Examination.couplant_material,
    Probe.to_oriented_points(),
    Walls['Frontwall'],
    Grid.to_oriented_points())
paths = bim.make_paths(Examination.block_material, Examination.couplant_material, interface_dict, max_number_of_reflection=0)


    
#%% Transmission delay
Nt = 2
theta_diff_d = 10
couplant_angles = np.linspace(0,(Nt-1)*theta_diff_d,Nt)
couplant_angles -= couplant_angles.mean()


focal_law = {}

#Transmit angles
Probe = arim.io.probe_from_conf(conf)

#couplant_angles =  np.arcsin( np.sin(block_angles) *conf['couplant_material']['longitudinal_vel'] / conf['block_material']['longitudinal_vel'])
block_angles =  np.arcsin( np.sin(np.deg2rad(couplant_angles)) *conf['block_material']['longitudinal_vel'] / conf['couplant_material']['longitudinal_vel'])
focal_law['transmit'] = {'block_angles':block_angles,
                         'couplant_angles':couplant_angles}

#Delay times
focal_law['transmit']['delay_times'] = np.zeros([Nt,conf['probe']['numx']])
for angle,n in zip(couplant_angles,range(Nt)):
    t_x_diff = Probe.locations.x-Probe.locations.x.mean();
    delay_vec = t_x_diff * np.sin(np.deg2rad(angle)) / conf['couplant_material']['longitudinal_vel']
    delay_vec -= delay_vec.min()
    focal_law['transmit']['delay_times'][n,:] = delay_vec




#%% PWI transmit focal law

N_rays = 10
c1 = Examination.couplant_material.longitudinal_vel
c2 = Examination.block_material.longitudinal_vel
            
transmit_rays = fn_PlaneWaveIntersections(couplant_angles,N_rays,c1,c2,Walls['Frontwall'],Probe,Grid)


#%% plot
ax = plt.subplot()
aplt.plot_interfaces(
    [
        Probe.to_oriented_points(),
        *Examination.walls.values(),
        Grid.to_oriented_points(),
    ],
    ax = ax,
    show_last=False,
    markers=[".", "-", "-", "d", ".k"],
)

colors = ['k','r']
for ii,intersections in enumerate(transmit_rays):
    c = colors[ii]
    ax.plot(intersections[0].points.x,intersections[0].points.z,':'+c,label=str(couplant_angles[ii]) + '° in couplant')
ax.legend(bbox_to_anchor=[1,1])