import numpy as np
import random
from math import pi, sin, cos
from ray_caster import RayCaster
import cv2 as cv
from skimage.draw import line


class Particle(object):
    """Particle for tracking a robot with a particle filter.

    The particle consists of:
    - a robot pose
    - a weight
    - a map consisting of landmarks
    """

    def __init__(self, num_particles, noise, pixel_size=.005):
        """Creates the particle and initializes location/orientation"""
        self.noise = noise

        # initialize robot pose at origin
        # self.pose = np.vstack([300, 300, 0])
        self.pose = np.vstack([0, 0, 0])


        # initialize weights uniformly
        self.weight = 1.0 / float(num_particles)

        # Trajectory of the particle
        self.trajectory = []

        # initialize the grid map
        self.prior = 0.5
        self.free_lo = .5
        self.occ_lo = .9

        self.map = np.ones((600, 600)) * self.prior
        self.trajectory_map = np.zeros((600, 600), dtype=np.uint8) 

        # sensor model
        self.ray_caster = RayCaster(self.map, pixel_size=pixel_size)
        self.pixel_size = pixel_size
        self.laser_eps = []



    def motion_update(self, odom):
        """Predict the new pose of the robot"""

        # odom.tr /= self.pixel_size
        # odom.r1 = np.degrees(odom.r1)
        # odom.r2 = np.degrees(odom.r2)

        # append the old position
        self.trajectory.append(self.pose)

        # noise sigma for delta_rot1
        delta_rot1_noisy = random.gauss(odom.r1, self.noise[0])

        # noise sigma for translation
        translation_noisy = random.gauss(odom.tr, self.noise[1])

        # noise sigma for delta_rot2
        delta_rot2_noisy = random.gauss(odom.r2, self.noise[2])

        
        # Estimate of the new position of the Particle
        x_new = self.pose[0] + odom.tr * cos(self.pose[2] + odom.r1)
        #print(sin(self.pose[2] + odom.r1))
        y_new = self.pose[1] + odom.tr * sin(self.pose[2] + odom.r1)
        theta_new = normalize_angle(self.pose[2] + odom.r1 + odom.r2)

        self.pose = np.array([x_new, y_new, theta_new])
        
        y_new = int(y_new/self.pixel_size + 300)
        x_new = int(x_new/self.pixel_size + 300)

        # self.trajectory_map[y_new, x_new] = 255
        # cv.imshow("trajectory", self.trajectory_map)
        # print(round(self.pose[0], 3), round(self.pose[1], 3), round(np.degrees(self.pose[2]), 3))



    def sensor_update(self, scan):
        """
        Description: Weight the particles according to the current map
         of the particle and the scan observations z.
        
        Input:
            - scan : LaserMsg
        """

        est, mask_collided, laser_eps = self.ray_caster.cast(self.pose, scan,
        show_rays=True)
        ranges = np.array(scan.ranges) / self.pixel_size
        
        error = np.sum(np.abs(est[mask_collided] - ranges[mask_collided]))
        
        self.weight = 1 / (1 + error)

        self.laser_eps = laser_eps

    def _inv_sensor_model(self, pf):
        """
        Description: Inverse Sensor Model for
        Sonars Range Sensors

        Input:
            - pf: perceptual field

        Output:
            - map: map with logodds.
        """
        result_map = np.zeros(np.shape(self.map))
        
        pf_x, pf_y = pf
        eps_x, eps_y = self.laser_eps

        # subsitute the rasters with free logodds on the result_map
        result_map[pf_y, pf_x] = self.free_lo
        # subsitute the laser_eps with occupied logodds
        result_map[eps_y, eps_x] = self.occ_lo

        return result_map

    def _get_perceptual_field(self, pose):
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
        x = int(x / self.pixel_size) + 300
        y = int(y / self.pixel_size) + 300

        # get the rasters from the robot_pose to the laser_eps
        eps_x, eps_y = self.laser_eps
        for x2, y2 in zip(eps_x, eps_y):
            raster_x, raster_y = line(x, y, x2, y2)
            X.extend(raster_x)
            Y.extend(raster_y)

        return X, Y
        

    def map_update(self):
        """
        Description: update the map based on the occupancy grid 
        mapping algorithm.

        reference: slide 24 at http://ais.informatik.uni-freiburg.de/teaching/ws12/mapping/pdf/slam11-gridmaps.pdf 
        
        Input: 
            - laser_eps: laser end points

        Output:
            - map: the updated map
        """
        pf = self._get_perceptual_field(self.pose)
        invmod = self._inv_sensor_model(pf)
        pf_x, pf_y = pf
        self.map[pf_y, pf_x] = self.map[pf_y, pf_x] + invmod[pf_y, pf_x] - self.prior


def normalize_angle(angle):
    """Normalize the angle between -pi and pi"""

    while angle > pi:
        angle = angle - 2 * pi

    while angle < -pi:
        angle = angle + 2 * pi

    return angle