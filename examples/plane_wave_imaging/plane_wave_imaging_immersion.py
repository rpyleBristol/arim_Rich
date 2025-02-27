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
from Functions_PWI import make_views_pwi,shift_time_domain_signals,PWI_find_all_intersections,ray_tracing_for_views_PWI
from collections import OrderedDict

def SurfaceToWall(surfaces,wall_points_per_mm,names=['Frontwall']):
    """
    Convert surfaces to walls with a given number of points per millimeter.

    Parameters
    ----------
    surfaces
        List of surfaces, each represented as an array of points.
    wall_points_per_mm
        Number of wall points per millimeter.
    names
        List of names for the walls (default is ['Frontwall']).

    Returns
    -------
    Walls
        Dictionary of walls with names as keys and wall geometries as values.

    """
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
wall_points_per_mm = 10

## Sample
standoff = 0.0195
thickness = 0.0099

## Plane waves
couplant_angles = [-19.2414] #Angle of plane waves fired
plane_waves = {}
for x,theta in enumerate(couplant_angles):
    plane_waves[f'PW {x}'] = theta

#%% Surfaces
surface_point_n = 1000
curvature_height = 0#-5e-3

x1 = np.linspace(-25e-3,25e-3,surface_point_n)
z1 =  curvature_height*np.sin((x1-x1.min())/(x1.max()-x1.min())*np.pi)+standoff - curvature_height*0.5

x2 = x1.copy()
z2 =  np.ones([len(x2)])*(standoff+thickness)
s1 = np.stack([x1,np.zeros_like(x1),z1],1)
s2 = np.stack([x2,np.zeros_like(z1),z2],1)

#%% Objects
#Probe
Probe = arim.io.probe_from_conf(conf)
N_rays = 2*Probe.numelements - 1 # (2*numelements - 1) in Rachev, Rosen K., et al. "Plane wave imaging techniques for immersion testing of components with nonplanar surfaces."
 
##Walls 
wall_names = ['Frontwall','Backwall']
Walls = SurfaceToWall([s1,s2],wall_points_per_mm,wall_names)

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

    
#Paths
interface_dict = bim.make_interfaces(Examination.couplant_material,
    Probe.to_oriented_points(),
    Walls['Frontwall'],
    Grid.to_oriented_points())
Paths = bim.make_paths(Examination.block_material, Examination.couplant_material, interface_dict, max_number_of_reflection=0)

#Frame
Frame = arim.io.frame_from_conf(conf)
Frame.tx = np.zeros_like(Frame.rx)
#import scipy
#Frame.timetraces = scipy.io.loadmat('L5_A0_sim.mat')['exp_data']['timetraces'][0][0][0].T


#%% Transmission delay law (calc used in transmission, not in imaging, only here for fullness)
Nt = len(couplant_angles) #number of fired plane waves

block_angles =  np.arcsin( np.sin(np.deg2rad(couplant_angles)) *conf['block_material']['transverse_vel'] / conf['couplant_material']['longitudinal_vel'])
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

#%% Apply delay to timetraces so at 0 time the wavefront is at the central element

transmission['reference_element_delay'] = np.zeros([Nt,conf['probe']['numx']])
for angle,n in zip(couplant_angles,range(Nt)):
    t_x_diff_ref = Probe.locations.x.mean()-Probe.locations.x[ref_elements[n]]
    delay_vec_ref = t_x_diff_ref * np.sin(np.deg2rad(angle)) / conf['couplant_material']['longitudinal_vel']
    transmission['reference_element_delay'][n,:] = -abs(delay_vec_ref)

transmission['reference_element_delay'] = transmission['reference_element_delay'].flatten(order=timetrace_flatten_order)

Frame = shift_time_domain_signals(Frame,transmission['reference_element_delay'])

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

viewname_used = ['T - T','T Backwall T - T','T Backwall T - L']#,'T - T Backwall T','T - T Backwall L']
views = OrderedDict({v:views[v] for v in viewname_used})


#%% TFM on reception

paths_set = set([v.rx_path for v in list(views.values())]) #set(v.tx_path for v in list(views.values()))|

from arim.ray import ray_tracing_for_paths

ray_tracing_for_paths(
    list(paths_set),
    walls=Walls,
    turn_off_invalid_rays=False,
    convert_to_fortran_order=False
)
#%% PWI focal law (i. propagation time calcs)
intersect_tol = 1e-9
rays = PWI_find_all_intersections(views,N_rays,plane_waves,intersect_tol=intersect_tol,plot_on=False)
ray_tracing_for_views_PWI(Grid,Probe,views,plane_waves,rays,intersect_tol=intersect_tol)


#%% Travel time plot for debugging
recieve_el = 20
plane_wave_n = 0
extent = [Grid.xmin,Grid.xmax,Grid.zmax,Grid.zmin]

plt.figure(figsize=[len(views)*3,3])
for ii,(viewname,view) in enumerate(views.items()):
    
    plotting = views[viewname].tx_path.rays.times[plane_wave_n].reshape(Grid.shape)[:,0,:].T.copy()
    plotting += views[viewname].rx_path.rays.times[recieve_el].reshape(Grid.shape)[:,0,:].T
    
    ax = plt.subplot(1,len(views),ii+1)
    plt.imshow(plotting*1e6)
    plt.xticks([])
    plt.yticks([])
    cbar = plt.colorbar()
    plt.title(f'PW {plane_wave_n},\n' + viewname)
cbar.set_label('Travel time (µs)', rotation=270, labelpad=9)
#%% Delay and sum

from arim.im.tfm import tfm_for_view
pwis = dict()
for viewname, view in views.items():
    with arim.helpers.timeit(f"TFM {view.name}"):
        
        pwis[viewname] = tfm_for_view(
            Frame, Grid, view, fillvalue=0.0, interpolation="nearest"
        )

# %% Plot PWI
clim = -40
wavename = list(plane_waves.keys())[0]

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
    aplt.plot_interfaces(
        rays[viewname][wavename],
        ax=ax1,
        show_probe=True,
        show_last=True,
        show_orientations=True,
        n_arrows=10,markers=[""]*len(rays[viewname][wavename]),)
    ax1.set_adjustable("box")
    ax1.axis([Grid.xmin, Grid.xmax, Grid.zmax, 0])
    aplt.plot_interfaces(
        [
            *Examination.walls.values(),
            Grid.to_oriented_points(),
        ],
        ax = ax1,
        show_last=False,
        markers=["-", "-", "d", ".k"],
    )
    
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


