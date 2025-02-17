import math
import numpy as np
import arim.geometry as g

def line_intersection_with_segments(A, theta1, points, orientations=[]):
    """
    Find the intersections of a line defined by origin A and angle theta1 with a set of line segments.
    
    Parameters:
    A (list): Origin of the line [x, y]
    theta1 (float): Angle of the line in radians
    curve (np.ndarray): n_points by 2 numpy array of coordinates
    
    Returns:
    list: Closest intersection point [x, y] to the origin A
    """
    intersections = []
    if len(orientations)==0:
        orientations = np.zeros([points.shape[0],])
    
    # Line direction vector
    dx = math.cos(theta1)
    dy = math.sin(theta1)
    
    num_segments = points.shape[0] - 1
    
    for i in range(num_segments):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        
        # Line segment direction vector
        dx_seg = x2 - x1
        dy_seg = y2 - y1
        
        # Determinant
        det = -dx * dy_seg + dy * dx_seg
        
        if det == 0:
            continue  # Lines are parallel
        
        # Solve for t and u (parametric equations)
        t = ((A[0] - x1) * dy_seg - (A[1] - y1) * dx_seg) / det
        u = ((A[0] - x1) * dy - (A[1] - y1) * dx) / det
        
        # Check if intersection is within the line segment
        if 0 <= u <= 1 and t > 0:
            intersection_x = A[0] + t * dx
            intersection_y = A[1] + t * dy
            intersections.append(([intersection_x, intersection_y], orientations[i]))
    
    if not intersections:
        return [],None
    
    closest_intersection, associated_orientation = min(intersections, key=lambda point: (point[0][0] - A[0])**2 + (point[0][1] - A[1])**2)
    
    return closest_intersection, associated_orientation



def fn_PlaneWaveIntersections(couplant_angles,N_rays,c1,c2,fw,Probe,Grid):
    """
    Find the plane wave rays through couplant into a sample.
    
    Parameters:
    couplant_angles (list): Angles in degrees of transmitted plane waves
    N_rays: Number of rays to split plane wave into. Produces N_rays-1 imaging regions for PWI.
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
        

            A = [origin_positions[ray_ii,0],origin_positions[ray_ii,1]] #element position
            theta1 = np.deg2rad(couplant_angles[N_ii])
            theta1 += probe_angle
            
            #Probe to frontwall
            points = np.stack([fw[0].x,fw[0].z],1)
            orientations = np.arctan2(fw[1].z[:,2] , fw[1].x[:,2])
            S1,wall_angle1 = line_intersection_with_segments(A, theta1, points, orientations) 
            
            if len(S1)==0: #Check intersection with fw
                S1 = [np.nan,np.nan]
                S2 = [np.nan,np.nan]
            else:
                #Frontwall to backwall

                theta2 = np.arcsin(np.sin(theta1-wall_angle1) * c2 / c1)+wall_angle1
            
            if np.isnan(theta2): #Check critical angle
                S2 = [np.nan,np.nan]
            else:
                S2,_ = line_intersection_with_segments(S1, theta2, bound) 

            coords = np.stack([[A[0],0,A[1]],
                                     [S1[0],0,S1[1]],
                                     [S2[0],0,S2[1]]],0)
            rays.append( g.default_oriented_points(g.Points(coords)))
            
        rays = g.combine_oriented_points(rays)
        transmit_rays.append( [rays] )
    return transmit_rays