#%% Imports 
import numpy as np
from matplotlib import pyplot as plt
import arim
from scipy.spatial.transform import Rotation
import time

import arim.geometry as g
import sys
sys.path.append('functions_scanning')
import arim.ray, arim.io, arim.signal, arim.im
import arim.models.block_in_immersion as bim
import arim.plot as aplt
from Functions_PWI import is_point_in_polygon_grid,plane_wave_intersection_for_one_ray,make_views_pwi,shift_time_domain_signals,fn_PWI_ray_tracing,find_intersections
from collections import OrderedDict

def fn_SurfaceToWall(surfaces,wall_points_per_mm,names=['Frontwall']):
    Walls = {}
    for ss,s in enumerate(surfaces):
        numpoints = []
        #Calc numpoints for each line in surface
        for ii in range(s.shape[0]-1):
            dist = np.sqrt(np.square(s[ii+1,0]-s[ii,0]) + np.square(s[ii+1,2]-s[ii,2]))
            numpoints.append( int(np.ceil(dist*1e3*wall_points_per_mm)) )
        new_wall = arim.geometry.make_contiguous_geometry(s, int(np.ceil(dist*1e3*wall_points_per_mm)))
        new_wall = arim.geometry.combine_oriented_points(list(new_wall.values()), name=names[ss])
        Walls[names[ss]] = new_wall
    return Walls

#%% Inspection inputs
timetrace_flatten_order = 'F' #flattening done in matlab -> Fortran style

## Conf
conf_name = 'conf.yaml'
conf = arim.io.load_conf_file(conf_name)
array_width = conf['probe']['pitch_x'] * (conf['probe']['numx']-1)
array_rot = np.array([0,0,0])

## Imaging mesh
imaging_res = 0.1e-3
wall_points_per_mm = 10#8

## Sample
standoff = 0.0201
thickness = 0.01

## Plane waves
couplant_angles = [-19.2414] #Angle of plane waves fired
plane_waves = {}
for x,theta in enumerate(couplant_angles):
    plane_waves[f'PW {x}'] = theta
#%% Surfaces
surface_point_n = 1000
curvature_height = 0

x1 = np.linspace(-25e-3,25e-3,surface_point_n)

z1 =  curvature_height*np.sin((x1-x1.min())/(x1.max()-x1.min())*np.pi)+standoff - curvature_height*0.5

x2 = x1.copy()
z2 =  np.ones([len(x2)])*(standoff+thickness)
s1 = np.stack([x1,np.zeros_like(x1),z1],1)
s2 = np.stack([x2,np.zeros_like(z1),z2],1)

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
    xmin = -22e-3,#x1[0]-1e-3,
    xmax = -12e-3,#x1[-1]+1e-3,
    ymin = 0.0,
    ymax = 0.0,
    zmin = standoff,
    zmax = standoff+thickness,
    pixel_size = imaging_res,
)
N_rays = 2*Probe.numelements - 1 # (2*numelements - 1) in Rachev, Rosen K., et al. "Plane wave imaging techniques for immersion testing of components with nonplanar surfaces."

    

#Paths
interface_dict = bim.make_interfaces(Examination.couplant_material,
    Probe.to_oriented_points(),
    Walls['Frontwall'],
    Grid.to_oriented_points())
Paths = bim.make_paths(Examination.block_material, Examination.couplant_material, interface_dict, max_number_of_reflection=0)

#Frame
Frame = arim.io.frame_from_conf(conf)
#import scipy
#Frame.timetraces = scipy.io.loadmat('L5_A0_sim.mat')['exp_data']['timetraces'][0][0][0].T

#Probe
Probe = arim.io.probe_from_conf(conf)

#%% Transmission delay law (calc used in transmission, not in imaging, only here for fullness)
Nt = len(couplant_angles) #number of fired plane waves

block_angles =  np.arcsin( np.sin(np.deg2rad(couplant_angles)) *conf['block_material']['longitudinal_vel'] / conf['couplant_material']['longitudinal_vel'])
ref_elements = []
for c in couplant_angles:
    if c < 0:
        ref_elements.append([-1]) #Element with time=0 delay
    else:
        ref_elements.append([0])
transmission = {'block_angles':block_angles,
                'couplant_angles':couplant_angles,
                'reference_elements':ref_elements}

#Delay law
transmission['delay_times'] = np.zeros([Nt,conf['probe']['numx']])
for angle,n in zip(couplant_angles,range(Nt)):
    t_x_diff = Probe.locations.x-Probe.locations.x[ref_elements[n]]
    delay_vec = t_x_diff * np.sin(np.deg2rad(angle)) / conf['couplant_material']['longitudinal_vel']

    transmission['delay_times'][n,:] = delay_vec
transmission['delay_times'] = transmission['delay_times'].flatten(order=timetrace_flatten_order)
#%% Apply delay to timetraces so at 0 time the wavefront is at the reference element

transmission['reference_element_delay'] = np.zeros([Nt,conf['probe']['numx']])
for angle,n in zip(couplant_angles,range(Nt)):
    t_x_diff_ref = Probe.locations.x.mean()-Probe.locations.x[ref_elements[n]]
    delay_vec_ref = t_x_diff_ref * np.sin(np.deg2rad(angle)) / conf['couplant_material']['longitudinal_vel']
    transmission['reference_element_delay'][n,:] = -abs(delay_vec_ref)

transmission['reference_element_delay'] = transmission['reference_element_delay'].flatten(order=timetrace_flatten_order)

Frame = shift_time_domain_signals(Frame,transmission['reference_element_delay'])

#
plt.figure()
plt.imshow(abs(Frame.timetraces))
#%% Views
imaging_walls = ['Backwall']
views = make_views_pwi(
    Examination,
    Probe.to_oriented_points(),
    Grid.to_oriented_points(),
    walls_for_imaging=imaging_walls,
)

viewname_used = 'T Backwall T - T'
views = OrderedDict({viewname_used:views[viewname_used]})

#%% PWI focal law (i. propogation time calcs)

c1 = Examination.couplant_material.longitudinal_vel
c2 = Examination.block_material.transverse_vel
            
#transmit_rays = plane_wave_intersections(couplant_angles,N_rays,c1,c2,Walls['Frontwall'],Probe,Grid)

rays = fn_PWI_ray_tracing(views,N_rays,plane_waves,intersect_tol=1e-9,plot_on=True)

#%% Use rays to split up grid into pw 'beams' by using pairs of rays

grid_bound_corners = np.array([[Grid.xmin,0,Grid.zmin],
                                [Grid.xmax,0,Grid.zmin],
                                [Grid.xmax,0,Grid.zmax],
                                [Grid.xmin,0,Grid.zmax],
                                [Grid.xmin,0,Grid.zmin]])
grid_bound_pts = g.Points(grid_bound_corners,name='Grid bound')
grid_bound_pts = g.default_oriented_points(grid_bound_pts)
probe_centre = Probe.locations.coords.mean(0)
for viewname, view in views.items():
    interfaces = view.tx_path.interfaces
    travel_times = np.zeros([Nt,Grid.size])
    numlegs = view.tx_path.numlegs
    assert view.tx_path.interfaces[-1].points.name == Grid.name
    
    for nn,(wavename, couplant_angle) in enumerate(plane_waves.items()):
        rays_last_interface = rays[viewname][wavename][-1]
        rays_first_interface = rays[viewname][wavename][0]
        n_rays = rays_last_interface.points.shape[0]
        in_plane_wave = np.zeros(Grid.shape,dtype=bool)

        #Find beams in imaging grid
        grid_bound_intersections, _ , _ = find_intersections(rays_last_interface,grid_bound_pts, intersect_tol=1e-9, closest=False)
        
        
        #Travel time up to last interface
        interface_time = np.zeros([n_rays-1])
        interface_dist = np.zeros([n_rays-1])
        for ii in range(numlegs-1):
            vel = view.tx_path.velocities[ii]
            for rr in range(n_rays-1):
                #Approximate beam travel time as mean of side rays
                p1 = rays[viewname][wavename][ii].points[rr:rr+2].mean(0) 
                p2 = rays[viewname][wavename][ii+1].points[rr:rr+2].mean(0)
                dist = np.sqrt(np.sum(np.square(p2-p1)))
                interface_dist[rr] += dist
                interface_time[rr] += dist / vel
  
        #Travel time from last interface to grid
        vel_final = view.tx_path.velocities[-1]
        vel_initial = view.tx_path.velocities[0]
        for rr in range(n_rays-1):    
            r = rays_last_interface.points[rr:rr+2]
            r_mean = r.mean(0)
            b = grid_bound_intersections[rr:rr+2]
            
            #sort left to right
            r = r[r[:, 0].argsort()]
            b = b[b[:, 0].argsort()]
            
            beam_bounds = [[r[0,0],r[0,2]],
                            [r[1,0],r[1,2]],
                            [b[1,0],b[1,2]],
                            [b[0,0],b[0,2]]]
    
            in_beam = is_point_in_polygon_grid(Grid, beam_bounds)
            in_beam = in_beam[:,np.newaxis,:]
            in_plane_wave[in_beam] = True
            in_beam = in_beam.flatten() 
            
            
            dx = Grid.x - r_mean[0]
            dy = Grid.y - r_mean[1]
            dz = Grid.z - r_mean[2]
            interface_to_grid_vec = np.array([dx.flatten(),dy.flatten(),dz.flatten()]).T
            ray_ori = rays_last_interface.orientations
            ray_ori_mean = np.array([ray_ori.x[rr:rr+2, 2].mean(0), 0, ray_ori.z[rr:rr+2, 2].mean(0)])
            ray_ori_mean /= np.linalg.norm(ray_ori_mean,2)
            dist_grid = np.dot(interface_to_grid_vec,ray_ori_mean)
            grid_time = dist_grid / vel_final

            #Adjust for offset of beam origin from centre of array
            ray_origin = rays_first_interface.points[rr:rr+2].mean(0)
            ray_origin_ori = rays_first_interface.orientations
            ray_origin_ori_mean = np.array([ray_origin_ori.x[rr:rr+2, 2].mean(0), 0, ray_origin_ori.z[rr:rr+2, 2].mean(0)])
            ray_origin_ori_mean /= np.linalg.norm(ray_origin_ori_mean,2)
            dx_adj = ray_origin[0] - probe_centre[0]
            dy_adj = ray_origin[1] - probe_centre[1]
            dz_adj = ray_origin[2] - probe_centre[2]
            origin_to_probe_centre_vec = np.array([dx_adj,dy_adj,dz_adj])
            dist_adj = np.dot(origin_to_probe_centre_vec,ray_origin_ori_mean)
            adj_time = dist_adj / vel_initial
            
            #Add to time_array - NOTE: ATM OVERWRITING WHERE A PIXEL IS SEEN BY MORE THAN ONE BEAM
            n_overwrite = np.sum(travel_times[nn,in_beam.flatten()]>0)
            if n_overwrite>0:
                print(f"Overwriting {n_overwrite} grid points")
            travel_times[nn,in_beam] = interface_time[rr] + grid_time[in_beam] + adj_time
        travel_times[nn,~in_plane_wave.flatten()] = np.inf
        
#%%  TEMP RAYS OBJECT TO CARRY TRAVEL TIMES
interior_indices = np.zeros([1,Nt,Grid.size],dtype=int)
pw_points = g.Points([[np.sin(np.deg2rad(x)),0,np.cos(np.deg2rad(x))] for x in couplant_angles],name="PW angles")
fermat_path = arim.ray.FermatPath((pw_points, c1, Walls['Frontwall'][0], c2, Grid.to_1d_points()))
pwi_rays = arim.ray.Rays(travel_times, interior_indices, fermat_path)
#%% Iterate through imaging grid regions split by rays
"""
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
    in_plane_wave = np.zeros(Grid.shape,dtype=bool)
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
            
            dx = Grid.x - S1[0]
            dz = Grid.z- S1[2]
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
            #idx = np.argmin(np.sum(np.square(fw.points.coords-S1),1))
            #interior_indices[0,n,:] = idx1
            
            
            in_plane_wave[in_beam] = True
    times[n,~in_plane_wave.flatten()] = np.inf
            
    
            

            
#Make into rays
pwi_rays = arim.ray.Rays(times, interior_indices, fermat_path)"""

#%% TFM on reception

paths_set = set([v.rx_path for v in list(views.values())]) #set(v.tx_path for v in list(views.values()))|

from arim.ray import ray_tracing_for_paths


ray_tracing_for_paths(
    list(paths_set),
    walls=Walls,
    turn_off_invalid_rays=False,
    convert_to_fortran_order=False
)

#%% Overwrite tx_path with PWI
views[viewname_used].tx_path.rays = pwi_rays

#Mask recieve where transmit invalid
lookup_times_tx = views[viewname_used].tx_path.rays.times
lookup_times_rx = views[viewname_used].rx_path.rays.times
print(lookup_times_tx.shape)
print(lookup_times_rx.shape)
lookup_times_rx[:,np.isinf(lookup_times_tx.sum(0))] = np.inf
#%% Plot for debugging
recieve_el = 20
plane_wave_n = 0

plotting = views[viewname_used].tx_path.rays.times[plane_wave_n].reshape(Grid.shape)[:,0,:].T
#plotting += views[viewname_used].rx_path.rays.times[recieve_el].reshape(Grid.shape)[:,0,:].T

extent = [Grid.xmin,Grid.xmax,Grid.zmax,Grid.zmin]
plt.figure(figsize=[8,4])
ax = plt.subplot()
plt.imshow(plotting,extent=extent)
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

"""for rr in range(n_rays):
    r = rays_current[rr][0].coords
    if np.sum(np.isnan(r)) == 0:
        ax.plot(r[0:2,0],r[0:2,2],'r',alpha=0.2)"""
plt.title(f"{couplant_angles[n]} degrees in couplant")
#%% PWI

from arim.im.tfm import tfm_for_view
pwis = dict()
for viewname, view in views.items():
    with arim.helpers.timeit(f"TFM {view.name}"):
        
        pwis[viewname] = tfm_for_view(
            Frame, Grid, view, fillvalue=0.0, interpolation="nearest"
        )

# %% Plot PWI
clim = -40
for i, (viewname, pwi) in enumerate(pwis.items()):
    assert pwi.grid is Grid
    fig=plt.figure(figsize=[10,6])
    ax1 = fig.add_subplot(121)
    
    aplt.plot_tfm(
        pwis[viewname],ax=ax1,
        clim=clim,
        scale="db",
        title=f"PWI {viewname}",
        savefig=False,
        draw_cbar=True,
        interpolation="none",
        cmap='jet'
    )
    ax1.set_adjustable("box")
    ax1.axis([Grid.xmin, Grid.xmax, Grid.zmax, 0])
    aplt.plot_interfaces(
        [
            Probe.to_oriented_points(),
            *Examination.walls.values(),
            Grid.to_oriented_points(),
        ],
        ax = ax1,
        show_last=False,
        markers=[".", "-", "-", "d", ".k"],
    )
    # Block script until windows are closed.
    
    ax2=fig.add_subplot(122)
    ax2, _ = aplt.plot_tfm(
        pwis[viewname],ax=ax2,
        clim=clim,
        scale="db",
        title=f"PWI {viewname}",
        savefig=False,
        draw_cbar=True,
        interpolation="none",
        cmap='jet'
    )
    plt.show()














#%%




"""fw = Walls['Frontwall']

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
"""

