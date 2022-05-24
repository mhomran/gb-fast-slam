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

    # subsitute the rasters with free logodds on the result_map
    result_map[pf] = free_lo
    # subsitute the laser_eps with occupied logodds
    result_map[laser_eps] = occ_lo

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

    measurements = scan.ranges / pixel_size
    x, y, theta = pose
    x //= pixel_size
    y //= pixel_size

    angle_min = scan.angle_min 
    angle_max = scan.angle_max
    angle_accuracy = scan.angle_increment
    h, w = map.shape

    start_angle = theta + angle_min
    end_angle = theta + angle_max

    scan_thetas = np.arange(start_angle, end_angle+1, angle_accuracy)
    scan_thetas = np.radians(scan_thetas)
    
    eps_x = x + (np.cos(scan_thetas) * measurements).astype(np.int) 
    eps_y = y + (np.sin(scan_thetas) * measurements).astype(np.int) 
    
    # out of the map borders
    x_outbound = np.logical_or(eps_x < 0, eps_x >= w)
    y_outbound = np.logical_or(eps_y < 0, eps_y >= h)
    eps_y[y_outbound] = 0
    eps_x[x_outbound] = 0

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
    x //= pixel_size
    y //= pixel_size

    # get the rasters from the robot_pose to the laser_eps
    for i in range(len(laser_eps[0])):
        raster_x, raster_y = line(x, y, laser_eps[0][i], laser_eps[1][i])
        X.append(raster_x)
        Y.append(raster_y)

    return X, Y
    

def occupancy_grid_mapping(map, pixel_size, pose, scan, prior, free_lo, occ_lo):
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
    map[pf] = map[pf] + inv_sensor_model(laser_eps, pf, free_lo, occ_lo) - prior
    
    return map
