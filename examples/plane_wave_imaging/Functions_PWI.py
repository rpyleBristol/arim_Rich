import math
import numpy as np
import arim.geometry as g
from matplotlib.path import Path
from collections import OrderedDict
from arim.models.block_in_immersion import *




def is_point_in_polygon_grid(Grid, polygon):
    """
    For each point in the Grid, check if it lies within the given four-sided polygon.

    Parameters
    ----------
    Grid
        Object containing imaging points.
        Attributes:
        - x: Shape (n, 1, m)
        - z: Shape (n, 1, m)
    polygon
        List of shape (4, 2) defining the corner points of the polygon.

    Returns
    -------
    mask
        Numpy array of shape (n, m). Points within the polygon are marked as True.

    Notes
    -----
    Uses matplotlib's Path.contains_points to check if points lie within the polygon.

    """
    X = Grid.x[:,0,:]
    Z = Grid.z[:,0,:]
    if X.shape != Z.shape:
        raise ValueError("X and Y must have the same shape")
    
    path = Path(polygon)
    points = np.vstack((X.flatten(), Z.flatten())).T
    
    mask = path.contains_points(points)
    
    return mask.reshape(X.shape)

def make_paths_pwi(
    block_material,
    couplant_material,
    interface_dict,
    max_number_of_reflection=1,
):
    """
    Creates all iterations of paths up with max_number_of_reflection. Path
    names are defined as the wave modes of the ray in transmit convention,
    separated by the wall which is skipped from. If 1 reflection is allowed
    from a wall "backwall", then the paths returned will be named "L", "T",
    "L backwall L", "L backwall T", "T backwall L", "T backwall T".

    Paths are returned in transmit convention: for the path XY, X is the mode
    before reflection against the backwall and Y is the mode after reflection.
    The path XY in transmit convention is the path YX in receive convention.

    All paths are given the postscript '-pw' to indicate that their path is 
    described by a plane wave.
    
    Parameters
    ----------
    block_material : Material
        The material of the block.
    couplant_material : Material
        The couplant material.
    interface_dict : dict[Interface]
        Dictionary of interfaces with keys "probe", "frontwall_trans", "grid", and wall names.
    max_number_of_reflection : int, optional
        The maximum number of reflections allowed. Default is 1.

    Returns
    -------
    paths : OrderedDict
        Ordered dictionary of paths.

    """
    paths = OrderedDict()

    if max_number_of_reflection > 2:
        raise NotImplementedError
    if max_number_of_reflection < 0:
        raise ValueError

    probe = interface_dict["probe"]
    frontwall = interface_dict["frontwall_trans"]
    grid = interface_dict["grid"]
    
    wall_dict = OrderedDict(
        (key, val)
        for key, val in interface_dict.items()
        if key not in ["probe", "grid", "frontwall_trans"]
    )
    wall_names = list(wall_dict.keys())

    if (max_number_of_reflection > 0 and len(wall_names) == 0) or (
        max_number_of_reflection > 1 and len(wall_names) < 2
    ):
        raise ValueError("Not enough walls to reflect from.")

    modes = (c.Mode.longitudinal, c.Mode.transverse)
    for no_reflections in range(max_number_of_reflection + 1):
        # For this number of reflections, make all the combinations of paths.
        path_indices_up_to_refl = list(product(range(2), repeat=no_reflections + 1))
        for path_indices in path_indices_up_to_refl:
            # For each path with this number of reflections.
            path_modes = [c.Mode.longitudinal]
            path_interfaces = [probe, frontwall]
            path_materials = [couplant_material]

            for i, mode in enumerate(path_indices):
                if i != 0:
                    # Settled on naming convention which includes wall name with
                    # spaces in between (MC 20/12/24)
                    path_interfaces.append(wall_dict[wall_names[i - 1]])
                path_modes.append(modes[mode])
                path_materials.append(block_material)
            path_interfaces.append(grid)

            path = c.Path(
                interfaces=path_interfaces,
                materials=path_materials,
                modes=path_modes,
            )
            path.name = path.name + ' -pw'
            paths[path.name] = path
    return paths

def make_views_pwi(
    examination_object,
    probe_oriented_points,
    scatterers_oriented_points,
    walls_for_imaging=None,
):
    """
    Make views for the measurement model of a block in immersion, transmission 
    by plane, focussed in reception.

    Parameters
    ----------
    examination_object : arim.core.BlockInImmersion
    probe_oriented_points : OrientedPoints
    scatterers_oriented_points : OrientedPoints
    walls_for_imaging : list[str]
        Keys of the walls in examination_object.walls which will be used to reflected
        from when imaging. Must be provided in the order that they are reflected
        from on the transmit path. The front wall transmission should not be provided
        in this list, it is assumed to exist in the examination object because this
        is an immersion configuration. Subsequent reflections from the front wall
        may be included. The length of this list will be used as the max number of
        reflections. The default is None, i.e. no reflections.

    Returns
    -------
    views: OrderedDict[Views]

    """
    try:
        couplant = examination_object.couplant_material
        block = examination_object.block_material
        frontwall = None
        walls = OrderedDict()
        if walls_for_imaging is None:
            walls_for_imaging = []
        frontwall = examination_object.walls["Frontwall"]
        for name in walls_for_imaging:
            walls[name] = examination_object.walls[name]
        max_number_of_reflection = len(walls_for_imaging)
    except AttributeError as e:
        raise ValueError("Examination object should be a BlockInImmersion") from e

    interfaces = make_interfaces(
        couplant,
        probe_oriented_points,
        frontwall,
        scatterers_oriented_points,
        walls,
    )
    paths_tx = make_paths_pwi(block, couplant, interfaces, max_number_of_reflection)
    paths_rx = make_paths(block, couplant, interfaces, max_number_of_reflection)
    
    return make_views_from_paths_pwi(paths_tx,paths_rx)

def make_views_from_paths_pwi(paths_dict_tx, paths_dict_rx):
    """
    Returns 'View' objects for the case of a block in immersion assuming plane
    wave propogation.

    Construct all possible views that can be constructed with the paths given 
    as argument.

    Parameters
    ----------
    paths_dict_tx : Dict[Path]
        Key: transmit path names
        (example: 'L', 'LT', 'L -pw'). Values: :class:`Path`
        
    paths_dict_rx : Dict[Path]
        Key: recieve path names 
        (example: 'L', 'LT', 'L -pw'). Values: :class:`Path`

    Returns
    -------
    views: OrderedDict[Views]

    """
    views = OrderedDict()
    for tx_path, rx_path in product(paths_dict_tx.values(), paths_dict_rx.values()):
        view_name = f"{tx_path.longname} - {rx_path.reverse_longname}"
            
        views[view_name] = c.View(tx_path, rx_path, view_name)

    return views

def shift_time_domain_signals(Frame, delays):
    """
    Shifts the time domain signals in Frame by the specified delays.

    Parameters
    ----------
    Frame : object
        Object containing time domain signals.
        Attributes:
        - timetraces: Array of shape (num_signals, num_time_points).
        - time.samples: Array of time points.
    delays : array_like
        Array of delays for each signal.

    Returns
    -------
    Frame : object
        The Frame object with shifted time domain signals.

    Notes
    -----
    - Shifts the signals in Frame.timetraces by the corresponding delays.
    - If the delay is longer than the time vector, the signal is set to zero.
    - Positive delays shift the signal to the right, negative delays shift to the left.

    """
    data = Frame.timetraces
    time_points = Frame.time.samples
    num_signals, num_time_points = data.shape
    shifted_data = np.zeros_like(data)
    
    for i in range(num_signals):
        delay = delays[i]

        delay_samples = int(np.round(delay / (time_points[1] - time_points[0])))
        if delay_samples == 0:
            shifted_data[i, :] = data[i, :]
        elif delay_samples < num_time_points and delay_samples>0:
            shifted_data[i, delay_samples:] = data[i, :-delay_samples]
        elif delay_samples < num_time_points and delay_samples<0:
            shifted_data[i, :-abs(delay_samples)] = data[i, abs(delay_samples):]
        else:
            shifted_data[i, :] = 0
            print("Error: delay longer than time vector")
    Frame.timetraces = shifted_data
    return Frame




def find_intersections(ray_current, interface_current, intersect_tol=1e-9, closest=True):
    """
    Finds the intersection points between rays and a surface, along with angles 
    of incidence and surface angles.

    Parameters
    ----------
    ray_current : OrientedPoints containing location and ray direction of ray 
    origin
    interface_current : Interface object describing surface
        Attributes:
        - points.coords: Array of surface coordinates.
    intersect_tol : float, optional
        Tolerance for ray/surface intersection check. Default is 1e-9.
    closest : bool, optional
        If True, find the closest intersection point. If False, find the 
        farthest. Default is True.

    Returns
    -------
    intersection_pts : ndarray
        Array of intersection points.
    angles_of_incidence : ndarray
        Array of angles of incidence - RELATIVE TO SURFACE NORMAL.
    surface_angles : ndarray
        Array of surface angles.
        

    """
    
    pts_ray, ori_ray = ray_current
    coords = interface_current.points.coords


    intersection_pts = np.zeros((pts_ray.shape[0], 3))
    angles_of_incidence = np.zeros(pts_ray.shape[0])
    surface_angles = np.zeros(pts_ray.shape[0])

    for idx in range(pts_ray.shape[0]):
        ray_origin = pts_ray[idx]
        ray_direction = np.array([ori_ray.x[idx, 2], 0, ori_ray.z[idx, 2]])
        #ray_direction = ray_direction / np.linalg.norm(ray_direction)

        returned_intersection = None
        if closest:
            distance_check = float('inf')
        else:
            distance_check = 0
        for i in range(len(coords) - 1):
            # By convention ensure pair of surface points are ordered by x-axis position
            if coords[i,0] > coords[i+1,0]:
                p2 = coords[i]
                p1 = coords[i + 1]
            else:
                p1 = coords[i]
                p2 = coords[i + 1]

            # Define the segment direction
            segment_direction = p2 - p1
            segment_direction = segment_direction / np.linalg.norm(segment_direction)

            # Calculate intersection point
            A1 = ray_direction[2]
            B1 = -ray_direction[0]
            C1 = A1 * ray_origin[0] + B1 * ray_origin[2]

            A2 = segment_direction[2]
            B2 = -segment_direction[0]
            C2 = A2 * p1[0] + B2 * p1[2]

            det = A1 * B2 - A2 * B1

            if det != 0:
                x_intersect = (B2 * C1 - B1 * C2) / det
                z_intersect = (A1 * C2 - A2 * C1) / det
                intersection_point = np.array([x_intersect, 0, z_intersect])

                # Check if the intersection point is within the segment and in the direction of the ray
                if ((min(p1[0], p2[0])-intersect_tol) <= x_intersect <= (max(p1[0], p2[0])+intersect_tol) and
                    (min(p1[2], p2[2])-intersect_tol) <= z_intersect <= (max(p1[2], p2[2])+intersect_tol) and
                    np.dot(intersection_point - ray_origin, ray_direction) > 0):
                    distance = np.linalg.norm(intersection_point - ray_origin)

                    if closest:
                        test = distance < distance_check
                    else:
                        test = distance > distance_check
                    if test:
                        distance_check = distance
                        returned_intersection = intersection_point

                        # Calculate angle of incidence
                        normal = np.array([-segment_direction[2], 0, segment_direction[0]])  # Normal vector to the segment
                        dot_product = np.dot(ray_direction, normal)
                        cross_product = np.cross(ray_direction, normal)
                        angle_of_incidence = np.arccos(np.clip(dot_product, -1.0, 1.0))
                        if cross_product[1] > 0:
                            angle_of_incidence = -angle_of_incidence

                        # Calculate the angle of the surface at the intersection point
                        surface_angle = np.arctan2(segment_direction[2], segment_direction[0])

        if returned_intersection is not None:
            intersection_pts[idx] = returned_intersection
            angles_of_incidence[idx] = angle_of_incidence
            surface_angles[idx] = surface_angle
        else:
            intersection_pts[idx] = np.nan
            angles_of_incidence[idx] = np.nan
            surface_angles[idx] = np.nan

    return intersection_pts, angles_of_incidence, surface_angles

def make_orient_pts_from_intersect(coords,angles_of_incidence,name='Plane wave intersections'):
    """
    Create oriented points from intersection coordinates and angles of incidence.

    Parameters
    ----------
    coords : ndarray
        Array of intersection coordinates.
    angles_of_incidence : ndarray
        Array of angles of incidence corresponding to the intersection points.
    name : str, optional
        Name for the oriented points. Default is 'Plane wave intersections'.

    Returns
    -------
    or_pts : OrientedPoints
        Combined OrientedPoints created from the intersection coordinates and angles of incidence.

    """
    or_pts = []
    for b in range(coords.shape[0]):
        points = g.Points(coords[b:b+1])
        orientations = g.default_orientations(points)
        #rot_mat = Rotation.from_euler('xyz', [0,angles_of_incidence[b],0], degrees=False).as_matrix()
        rot_mat = g.rotation_matrix_y(angles_of_incidence[b])
        orientations = orientations.rotate(rot_mat)
        or_pts.append(g.OrientedPoints(points,orientations))
    or_pts = g.combine_oriented_points(or_pts,name=name)
    return or_pts

import arim.plot as aplt
import matplotlib.pyplot as plt
def PWI_find_all_intersections(views,N_rays,plane_waves,intersect_tol=1e-9,plot_on=False):
    """
    Find all intersections of plane wave with interface, splitting plane wave
    into rays.

    Parameters
    ----------
    views : dict
        Dictionary of views to draw paths from.
    N_rays : int
        Number of rays to use.
    plane_waves : dict
        Dictionary of plane waves with their corresponding angles.
    intersect_tol : float, optional
        Tolerance for intersection checks. Default is 1e-9.
    plot_on : bool, optional
        If True, plot the interfaces and rays. Default is False.

    Returns
    -------
    rays : dict
        Dictionary of rays for each view and wave.

    """
    rays = {}
    for viewname, view in views.items():
        path = view.tx_path
        rays[viewname] = {}
        interfaces = path.interfaces[:-1] #skip grid
        for wavename, couplant_angle in plane_waves.items():
            
            #Initial ray positions
            probe_coords = interfaces[0].points
            origin_coords = np.stack([np.linspace(probe_coords.x.min(),probe_coords.x.max(),N_rays,endpoint=True),
                                         np.zeros([N_rays,]),
                                         np.linspace(probe_coords.z.max(),probe_coords.z.max(),N_rays,endpoint=True)],1)
            origin_angles = np.radians(np.ones(N_rays)*couplant_angle)
            
            rays[viewname][wavename] = [ make_orient_pts_from_intersect(origin_coords,origin_angles,name=wavename+f', {interfaces[0].points.name}') ]
            
            for ii in range(len(interfaces)-1):
                
                #Intersections of rays into next surface
                interface_current = interfaces[ii+1]
                rays_previous = rays[viewname][wavename][ii]
                m1 = path.modes[ii]
                c1 = path.materials[ii].velocity(m1)


                m2 = path.modes[ii+1]
                c2 = path.materials[ii+1].velocity(m2)
   
                intersection_pts, angles_of_incidence, surface_angles = find_intersections(rays_previous,interface_current, intersect_tol=intersect_tol)
                
                #calculate refraction and return to global angular frame
                angles_of_transmission = np.arcsin(np.sin(angles_of_incidence) * c2 / c1) - surface_angles
                if ii > 0:
                    #ASSUMING REFLECTIONS PAST FIRST TRANSMISSION
                    angles_of_transmission = np.pi - angles_of_transmission
                    
                
                new_ray = make_orient_pts_from_intersect(intersection_pts,angles_of_transmission,name=wavename+f', {interface_current.points.name}')
                rays[viewname][wavename].append(new_ray)

        
            if plot_on:
                #Plot 
                plt.figure()
                ax = plt.subplot()
                aplt.plot_interfaces(
                    rays[viewname][wavename],
                    ax=ax,
                    show_probe=True,
                    show_last=True,
                    show_orientations=True,
                    n_arrows=10,markers=["o"]*len(interfaces),)
                aplt.plot_interfaces(
                    interfaces,
                    ax = ax,
                    show_last=True,
                    markers=["-"]*len(interfaces),
                )

    return rays

from arim.ray import FermatPath, Rays
def ray_tracing_for_views_PWI(Grid,Probe,views,plane_waves,rays,intersect_tol=1e-9):
    """
    Perform calculates travel times of plane wave from Probe to imaging grid 
    via paths described in views.

    Parameters
    ----------
    Grid : object
        Object representing the imaging grid.
        Attributes:
        - xmin, xmax, zmin, zmax: Boundaries of the grid.
        - x, y, z: Coordinates of the grid points.
        - size: Total number of grid points.
    Probe : object
        Object representing the probe.
        Attributes:
        - locations.coords: Coordinates of the probe locations.
    views : dict
        Dictionary of views containing transmission paths.
    plane_waves : dict
        Dictionary of plane waves with their corresponding angles.
    rays : dict
        Dictionary of rays for each view and wave.
    intersect_tol : float, optional
        Tolerance for intersection checks. Default is 1e-9.

    Returns
    -------
    None. 
    Calculated travel times stored in the `tx_path` attribute of each view as 
    a Rays object.

    Notes
    -----
    - Based on the plane wave adapted in postprocessing (PWAPP) described in:
      Rachev, Rosen K., et al. "Plane wave imaging techniques for immersion 
      testing of components with nonplanar surfaces."
      IEEE Transactions on Ultrasonics, Ferroelectrics, and 
      Frequency Control 67.7 (2020): 1303-1316.

    """
    
    grid_bound_corners = np.array([[Grid.xmin,0,Grid.zmin],
                                    [Grid.xmax,0,Grid.zmin],
                                    [Grid.xmax,0,Grid.zmax],
                                    [Grid.xmin,0,Grid.zmax],
                                    [Grid.xmin,0,Grid.zmin]])
    grid_bound_pts = g.Points(grid_bound_corners,name='Grid bound')
    grid_bound_pts = g.default_oriented_points(grid_bound_pts)
    probe_centre = Probe.locations.coords.mean(0)
    Nt = len(plane_waves.items())
    for viewname, view in views.items():
        travel_times = np.zeros([Nt,Grid.size])
        numlegs = view.tx_path.numlegs
        assert view.tx_path.interfaces[-1].points.name == Grid.name
        in_plane_wave = np.zeros(Grid.shape,dtype=bool)
        for nn,(wavename, couplant_angle) in enumerate(plane_waves.items()):
            rays_last_interface = rays[viewname][wavename][-1]
            rays_first_interface = rays[viewname][wavename][0]
            n_rays = rays_last_interface.points.shape[0]
            
    
            #Find beams in imaging grid
            grid_bound_intersections, _ , _ = find_intersections(rays_last_interface,grid_bound_pts, intersect_tol=intersect_tol, closest=False)
            
            
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
                dist_grid = np.dot(interface_to_grid_vec,ray_ori_mean)#np.linalg.norm(interface_to_grid_vec,2,1)
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
                travel_times[nn,in_beam] = grid_time[in_beam] +interface_time[rr] +  + adj_time
        
        travel_times[nn,~in_plane_wave.flatten()] = np.inf
            
        # Ray object - NOTE ONLY TRAVEL TIMES REALLY FOR PWI
        fermat_path = FermatPath.from_path(views[viewname].tx_path)
        fermat_path[0].coords = fermat_path[0].coords[np.newaxis,:,:] #work around to get by assert in Rays
        interior_indices = np.zeros([fermat_path.num_points_sets-2,Nt,Grid.size],dtype=int) #not relevant at all to PWI transmit path
        pwi_rays = Rays(travel_times, interior_indices, fermat_path)
            
        # Overwrite tx_path with PWI
        views[viewname].tx_path.rays = pwi_rays
    
        #Mask recieve focal law where transmit invalid
        #lookup_times_tx = views[viewname].tx_path.rays.times
        #views[viewname].rx_path.rays.times[:,np.isinf(lookup_times_tx.sum(0))] = np.inf
        