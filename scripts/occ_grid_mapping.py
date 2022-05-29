"""
@ Author: Mohamed Hassanin Mohamed
@ Brief: occupancy grid mapping algorithm
@ data: 24/5/2022
"""

import numpy as np
from skimage.draw import line


# reference: slide 26 at http://ais.informatik.uni-freiburg.de/teaching/ws12/mapping/pdf/slam11-gridmaps.pdf
def inv_sensor_model(map, laser_eps, pf, free_lo, occ_lo):
    """
    Description: Inverse Sensor Model for
    Sonars Range Sensors

    Input:
        - laser_eps: the laser scan endpoints
        - pf: perceptual field
        - free_lo: free cell logodds 
        - occ_lo: occupied cell logodds 

    Output:
        - map: map with logodds.
    """
    result_map = np.zeros(np.shape(map))
    
    pf_x, pf_y = pf
    eps_x, eps_y = laser_eps

    # subsitute the rasters with free logodds on the result_map
    result_map[pf_x, pf_y] = free_lo
    # subsitute the laser_eps with occupied logodds
    result_map[eps_x, eps_y] = occ_lo

    return result_map

def get_laser_eps(map, pixel_size, pose, scan):
    """
    Description: Get the endpoints of the laser scan at a specific
    pose.
      
    Input:
      - map: grid map
      - pixel_size: pixel size to map from the world to the map
      - pose: pose (x, y, theta)
      - scan: LaserMsg
    Output:
      - eps: endpoints of the laser scan at the pose x
    """
    
    x, y, theta = pose
    x = int(x / pixel_size) + 300
    y = int(y / pixel_size) + 300
    ranges = np.array(scan.ranges) / pixel_size

    # theta = int(np.degrees(-theta))

    start_angle = 0
    end_angle = 360
    
    angles = np.arange(start_angle, end_angle, 1)
    cos_vec = np.cos(np.radians(angles))
    sin_vec = np.sin(np.radians(angles))

    rot = int(round(np.degrees(-theta)))
    X = ranges * np.roll(cos_vec.reshape(-1), rot)
    Y = ranges * np.roll(sin_vec.reshape(-1), rot)

    eps_x = x + X
    eps_y = y + Y
    eps_x = eps_x.astype(np.int)
    eps_y = eps_y.astype(np.int)
    
    eps_x[eps_x < 0] = 0
    eps_x[eps_x >= map.shape[1]] = map.shape[1]-1
    eps_y[eps_y < 0] = 0
    eps_y[eps_y >= map.shape[0]] = map.shape[0]-1

    return eps_x, eps_y

def get_perceptual_field(pixel_size, pose, laser_eps):
    """
    Description: get the perceptual field of a scan
    
    Input:
        - pose: the robot pose (x, y, theta)
        - laser_eps: laser endpoints
    
    Output:
        - X: the x coordiantes of the field cells
        - Y: the y coordinates of the field cells
    """ 
    X = []
    Y = []

    x, y, _ = pose
    x = int(x / pixel_size) + 300
    y = int(y / pixel_size) + 300

    # get the rasters from the robot_pose to the laser_eps
    eps_x, eps_y = laser_eps
    for x2, y2 in zip(eps_x, eps_y):
        raster_x, raster_y = line(x, y, x2, y2)
        X.extend(raster_x)
        Y.extend(raster_y)

    return X, Y
    

def occupancy_grid_update(map, pixel_size, pose, scan, prior, free_lo=.5, occ_lo=.9):
    """
    Description: update the map based on the occupancy grid 
    mapping algorithm.

    reference: slide 24 at http://ais.informatik.uni-freiburg.de/teaching/ws12/mapping/pdf/slam11-gridmaps.pdf 
    
    Input: 
        - map: the previous map
        - pixel_size: pixel size to map from the world to the map
        - pose: the robot pose (x, y, theta)
        - scan: LaserMsg
        - prior: scaler representing the prior of the map cells
        - free_lo: free cell logodds 
        - occ_lo: occupied cell logodds 

    Output:
        - map: the updated map
    """
    laser_eps = get_laser_eps(map, pixel_size, pose, scan)
    pf = get_perceptual_field(pixel_size, pose, laser_eps)
    invmod = inv_sensor_model(map, laser_eps, pf, free_lo, occ_lo)
    map[pf] = map[pf] + invmod[pf] - prior
    
    return map
