import math
import numpy as np
import arim.geometry as g
from matplotlib.path import Path
from collections import OrderedDict
from arim.models.block_in_immersion import *

def line_intersection_with_segments(A, theta1, points, orientations=[]):
    """
    Find the intersections of a line defined by origin A and angle theta1 with a set of line segments.
    
    Parameters:
    A (list): Origin of the line [x, 0, z]
    theta1 (float): Angle of the line in radians
    curve (np.ndarray): n_points by 2 numpy array of coordinates
    
    Returns:
    list: Closest intersection point [x, 0, z] to the origin A
    """
    intersections = []
    if len(orientations)==0:
        orientations = np.zeros([points.shape[0],])
    
    # Line direction vector
    dx = math.cos(theta1)
    dz = math.sin(theta1)
    
    num_segments = points.shape[0] - 1
    
    for i in range(num_segments):
        x1, z1 = points[i]
        x2, z2 = points[i + 1]
        
        # Line segment direction vector
        dx_seg = x2 - x1
        dz_seg = z2 - z1
        
        # Determinant
        det = -dx * dz_seg + dz * dx_seg
        
        if det == 0:
            continue  # Lines are parallel
        
        # Solve for t and u (parametric equations)
        t = ((A[0] - x1) * dz_seg - (A[2] - z1) * dx_seg) / det
        u = ((A[0] - x1) * dz - (A[2] - z1) * dx) / det
        
        # Check if intersection is within the line segment
        if 0 <= u <= 1 and t > 0:
            intersection_x = A[0] + t * dx
            intersection_z = A[2] + t * dz
            intersections.append(([intersection_x, 0, intersection_z], orientations[i]))
    
    if not intersections:
        return [],None
    
    closest_intersection, associated_orientation = min(intersections, key=lambda point: (point[0][0] - A[0])**2 + (point[0][1] - A[-2])**2)
    
    return closest_intersection, associated_orientation


def plane_wave_intersection_for_one_ray(A,couplant_angle,probe_angle,fw,c1,c2,bound):
    theta1 = np.deg2rad(couplant_angle)
    theta1 += probe_angle
    
    #Probe to frontwall
    points = np.stack([fw[0].x,fw[0].z],1)
    orientations = np.arctan2(fw[1].z[:,2] , fw[1].x[:,2])
    S1,wall_angle1 = line_intersection_with_segments(A, theta1, points, orientations) 
    if len(S1)==0: #Check intersection with fw
        S1 = [np.nan,np.nan,np.nan]
        S2 = [np.nan,np.nan,np.nan]
    else:
        #Frontwall to backwall
        
        alpha = np.sin(theta1-wall_angle1) * c2 / c1
        if abs(alpha)>1: #Check critical angle
            S2 = [np.nan,np.nan,np.nan]
            theta2 = np.nan
        else:
            theta2 = np.arcsin(alpha)+wall_angle1
            S2,_ = line_intersection_with_segments(S1, theta2, bound) 

    return S1,S2,theta1,theta2

def plane_wave_intersections(couplant_angles,N_rays,c1,c2,fw,Probe,Grid):
    """
    Find the plane wave rays through couplant into a sample.
    
    Parameters:
    couplant_angles (list): Angles in degrees of transmitted plane waves
    N_rays (float): Number of rays to split plane wave into. Produces N_rays-1 imaging regions for PWI.
    c1, c2 (floats): Sound speeds in m/s in couplant and sample respectively
    fw (OrientedPoints): Frontwall description
    Probe (Object): Description of probe
    Grid: (Object): Description of imaging grid
    
    Returns:
    list: Points where rays intersect with probe, frontwall and imaging grid
    """
    
    probe_angle = np.arctan2(Probe.to_oriented_points().orientations.z[:,2] ,Probe.to_oriented_points().orientations.x[:,2])[0]
    origin_positions = np.stack([np.linspace(Probe.locations.x[0],Probe.locations.x[-1],N_rays,endpoint=True),
                                 np.linspace(Probe.locations.z[0],Probe.locations.z[-1],N_rays,endpoint=True)],1)
    Nt = len(couplant_angles)
    
    
    
    bound = np.array([[Grid.xmin,Grid.zmin],
                        [Grid.xmax,Grid.zmin],
                        [Grid.xmax,Grid.zmax],
                        [Grid.xmin,Grid.zmax],
                        [Grid.xmin,Grid.zmin]])
    
    transmit_rays = []
    for N_ii in range(Nt):
        rays = []
        for ray_ii in range(N_rays):
        

            A = [origin_positions[ray_ii,0],0, origin_positions[ray_ii,1]] #element position
            S1,S2,theta1,theta2 = plane_wave_intersection_for_one_ray(A,couplant_angles[N_ii],probe_angle,fw,c1,c2,bound)

            coords = np.stack([A,
                               S1,
                               S2],0)
            rays.append( g.default_oriented_points(g.Points(coords,name=f'ray {ray_ii}, pw {couplant_angles[N_ii]}°')))
            
        #rays = g.combine_oriented_points(rays)
        transmit_rays.append( rays )
    return transmit_rays


def is_point_in_polygon_grid(Grid, polygon):
    """
    Find the points in the grid that are within the four sided polygon
    
    Parameters:
    Grid (object): Grid of imaging points
    N_rays (list): 4x2 grid defining the corner points of the polygon
    
    Returns:
    mask: Numpy array of points within the polygon
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

    Parameters
    ----------
    block_material : Material
    couplant_material : Material
    interface_dict : dict[Interface]
    max_number_of_reflection : int
        Default: 1.


    Returns
    -------
    paths : OrderedDict

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
            path.name = path.name + '_p'
            paths[path.name] = path
    return paths

def make_views_pwi(
    examination_object,
    probe_oriented_points,
    scatterers_oriented_points,
    walls_for_imaging=None,
):
    """
    Make views for the measurement model of a block in immersion (scatterers response
    only).

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
    tfm_unique_only : bool
        Default False. If True, returns only the views that give *different* imaging
        results with TFM (AB-CD and DC-BA give the same imaging result).

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
    Returns 'View' objects for the case of a block in immersion.

    Consut all possible views that can be constructed with the paths given as argument.

    If unique only ``unique_only`` is false,

    Parameters
    ----------
    paths_dict : Dict[Path]
        Key: path names (exemple: 'L', 'LT'). Values: :class:`Path`
    tfm_unique_only : bool
        Default: False. If True, returns only the views that give *different* imaging
        results with TFM (AB-CD and DC-BA give the same imaging result).

    Returns
    -------
    views: OrderedDict[Views]

    """
    views = OrderedDict()
    for tx_path, rx_path in product(paths_dict_tx.values(), paths_dict_rx.values()):
        view_name = f"{tx_path.longname} - {rx_path.reverse_longname}"
            
        views[view_name] = c.View(tx_path, rx_path, view_name)

    return views


