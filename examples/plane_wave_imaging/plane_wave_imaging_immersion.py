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
from Functions_PWI import is_point_in_polygon_grid,plane_wave_intersections,plane_wave_intersection_for_one_ray,make_views_pwi
from collections import OrderedDict

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
wall_points_per_mm = 10#8

## Sample
standoff = 15e-3
thickness = 30e-3

## Plane waves
Nt = 2 #number fired
theta_diff_d = 10 #angle between them in degrees
#%% Surfaces
surface_point_n = 1000
curvature_height = -5e-3

x1 = np.linspace(-35e-3,35e-3,surface_point_n)

z1 =  curvature_height*np.sin((x1-x1.min())/(x1.max()-x1.min())*np.pi)+standoff - curvature_height*0.5

x2 = x1.copy()
z2 =  np.ones([len(x2)])*(standoff+thickness)
s1 = np.stack([x1,z1],1)
s2 = np.stack([x2,z2],1)

#%% Objects

#Probe
Probe = arim.io.probe_from_conf(conf)
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
    xmin = x1[0],
    xmax = x1[-1],
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
Paths = bim.make_paths(Examination.block_material, Examination.couplant_material, interface_dict, max_number_of_reflection=0)

#Frame
Frame = arim.io.frame_from_conf(conf)

#%% Transmission delay
couplant_angles = np.linspace(0,(Nt-1)*theta_diff_d,Nt)
couplant_angles -= couplant_angles.mean()


focal_law = {}

#Transmit angles
Probe = arim.io.probe_from_conf(conf)

#couplant_angles =  np.arcsin( np.sin(block_angles) *conf['couplant_material']['longitudinal_vel'] / conf['block_material']['longitudinal_vel'])
block_angles =  np.arcsin( np.sin(np.deg2rad(couplant_angles)) *conf['block_material']['longitudinal_vel'] / conf['couplant_material']['longitudinal_vel'])
focal_law['transmit'] = {'block_angles':block_angles,
                         'couplant_angles':couplant_angles}

#Delay law
focal_law['transmit']['delay_times'] = np.zeros([Nt,conf['probe']['numx']])
for angle,n in zip(couplant_angles,range(Nt)):
    t_x_diff = Probe.locations.x-Probe.locations.x.mean();
    delay_vec = t_x_diff * np.sin(np.deg2rad(angle)) / conf['couplant_material']['longitudinal_vel']
    delay_vec -= delay_vec.min()
    focal_law['transmit']['delay_times'][n,:] = delay_vec




#%% PWI transmit focal law

N_rays = 2*Probe.numelements - 1 # (2*numelements - 1) in Rachev, Rosen K., et al. "Plane wave imaging techniques for immersion testing of components with nonplanar surfaces."
c1 = Examination.couplant_material.longitudinal_vel
c2 = Examination.block_material.longitudinal_vel
            
transmit_rays = plane_wave_intersections(couplant_angles,N_rays,c1,c2,Walls['Frontwall'],Probe,Grid)

#%% Views
imaging_walls = []
views = make_views_pwi(
    Examination,
    Probe.to_oriented_points(),
    Grid.to_oriented_points(),
    walls_for_imaging=imaging_walls,
)
views = OrderedDict({'L - L':views['L - L']}) #Debugging

#%% Iterate through imaging grid regions split by rays
fw = Walls['Frontwall']

fw_coords = np.stack([fw[0].x,fw[0].z],1)
fw_orientations = np.arctan2(fw[1].z[:,2] , fw[1].x[:,2])
extent = [Grid.xmin,Grid.xmax,Grid.zmax,Grid.zmin]
probe_angle = np.arctan2(Probe.to_oriented_points().orientations.z[:,2] ,Probe.to_oriented_points().orientations.x[:,2])[0]

bound = np.array([[Grid.xmin,Grid.zmin],
                    [Grid.xmax,Grid.zmin],
                    [Grid.xmax,Grid.zmax],
                    [Grid.xmin,Grid.zmax],
                    [Grid.xmin,Grid.zmin]])

probe_centre = Probe.locations.coords.mean(0)
times = np.zeros([Nt,Grid.size])
interior_indices = np.zeros([1,Nt,Grid.size],dtype=int)
pw_points = g.Points([[np.sin(np.deg2rad(x)),0,np.cos(np.deg2rad(x))] for x in couplant_angles],name="PW angles")
fermat_path = arim.ray.FermatPath((pw_points, c1, Walls['Frontwall'][0], c2, Grid.to_1d_points()))

for n in range(Nt):
    rays_current = transmit_rays[n]
    n_rays = len(rays_current)
    theta1 = np.deg2rad(couplant_angles[n])
    theta1 += probe_angle
    for r in range(n_rays-1):
        r1 = rays_current[r][0].coords
        r2 = rays_current[r+1][0].coords
    
        if np.sum(np.isnan(r1)+np.isnan(r2)) == 0: #check below crit angle and both rays reach front wall
            ray_bounds_block = [[r1[1,0],r1[1,2]],
                                [r1[2,0],r1[2,2]],
                                [r2[2,0],r2[2,2]],
                                [r2[1,0],r2[1,2]]]
    
            in_beam = is_point_in_polygon_grid(Grid, ray_bounds_block)
            in_beam = in_beam[:,np.newaxis,:]
            
            #Imaging of plane wave section assuming locally flat surface
            A = [(r1[0,0] + r2[0,0])/2, 0, (r1[0,2] + r2[0,2])/2]
            S1,S2,theta1,theta2 = plane_wave_intersection_for_one_ray(A,couplant_angles[n],probe_angle,fw,c1,c2,bound)
            couplant_dist = np.sqrt(np.sum(np.square(np.array(A)-np.array(S1))))
            
            dx = Grid.x*in_beam - S1[0]
            dz = Grid.z*in_beam - S1[2]
            block_dist = dx * np.cos(theta2) + dz * np.sin(theta2)
            
            dx_adj = A[0] - probe_centre[0]
            dz_adj = A[2] - probe_centre[2]
            beam_adust_time = (dx_adj * np.cos(theta1) + dz_adj * np.sin(theta1)) / c1
            
            travel_time = block_dist / c2 + couplant_dist / c1 + beam_adust_time
            
            #Add to time_array - NOTE: ATM OVERWRITING WHERE A PIXEL IS SEEN BY MORE THAN ONE BEAM
            n_overwrite = np.sum(times[n,in_beam.flatten()]>0)
            if n_overwrite>0:
                print(f"Overwriting {n_overwrite} grid points")
            times[n,in_beam.flatten()] = travel_time[in_beam].flatten()
            
            #Interior indeces - these don't make too much sense for plane waves so only include if needs be
            """idx = np.argmin(np.sum(np.square(fw.points.coords-S1),1))
            interior_indices[0,n,:] = idx1
            """
            
    
            
    #Plot for debugging
    plt.figure(figsize=[8,4])
    ax = plt.subplot()
    plotting = times[n,:].reshape(Grid.shape)[:,0,:].T
    plt.imshow(plotting*1e6,extent=extent)
    plt.xlim([extent[0],extent[1]])
    plt.ylim([extent[2],extent[3]])
    cbar = plt.colorbar()
    cbar.set_label('Travel time (µs)', rotation=270, labelpad=9)
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

    for rr in range(n_rays):
        r = rays_current[rr][0].coords
        if np.sum(np.isnan(r)) == 0:
            ax.plot(r[:,0],r[:,2],'r',alpha=0.2)
    plt.title(f"{couplant_angles[n]} degrees in couplane")
            
#Make into rays
pwi_rays = arim.ray.Rays(times, interior_indices, fermat_path)

#%% TFM on reception

paths_set = set(v.rx_path for v in list(views.values()))

from arim.ray import ray_tracing_for_paths
ray_tracing_for_paths(
    list(paths_set),
    walls=Walls,
    turn_off_invalid_rays=False,
    convert_to_fortran_order=False,
)

#%% Overwrite tx_path with PWI
views['L - L'].tx_path.rays = pwi_rays

print(views['L - L'].tx_path.rays.times.shape)
print(views['L - L'].rx_path.rays.times.shape)
#%% PWI

PWIs = dict()
from arim.im.tfm import tfm_for_view

for viewname, view in views.items():
    with arim.helpers.timeit(f"TFM {view.name}"):
        PWIs[viewname] = tfm_for_view(
            Frame, Grid, view, fillvalue=0.0, interpolation="nearest"
        )

