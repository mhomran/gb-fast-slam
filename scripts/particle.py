import numpy as np
import random
from math import pi, sin, cos

class Particle(object):
    """Particle for tracking a robot with a particle filter.

    The particle consists of:
    - a robot pose
    - a weight
    - a map consisting of landmarks
    """

    class LandmarkEKF(object):
        """EKF representing a landmark"""
        def __init__(self):
            self.observed = False
            self.mu = np.vstack([0, 0])  # landmark position as vector of length 2
            self.sigma = np.zeros((2, 2))  # covariance as 2x2 matrix

        def __str__(self):
            return "LandmarkEKF(observed  = {0}, mu = {1}, sigma = {2})".format(self.observed, self.mu, self.sigma)
    
    def __init__(self, num_particles, num_landmarks, noise):
        """Creates the particle and initializes location/orientation"""
        self.noise = noise

        # initialize robot pose at origin
        self.pose = np.vstack([0., 0., 0.])

        # initialize weights uniformly
        self.weight = 1.0 / float(num_particles)

        # Trajectory of the particle
        self.trajectory = []

        # initialize the landmarks aka the map
        self.landmarks = [self.LandmarkEKF() for _ in range(num_landmarks)]


    def motion_update(self, odom):
        """Predict the new pose of the robot"""

        # append the old position
        self.trajectory.append(self.pose)

        # noise sigma for delta_rot1
        delta_rot1_noisy = random.gauss(odom.r1, self.noise[0])

        # noise sigma for translation
        translation_noisy = random.gauss(odom.t, self.noise[1])

        # noise sigma for delta_rot2
        delta_rot2_noisy = random.gauss(odom.r2, self.noise[2])

        # Estimate of the new position of the Particle
        x_new = self.pose[0] + translation_noisy * cos(self.pose[2] + delta_rot1_noisy)
        y_new = self.pose[1] + translation_noisy * sin(self.pose[2] + delta_rot1_noisy)
        theta_new = normalize_angle(self.pose[2] + delta_rot1_noisy + delta_rot2_noisy)

        self.pose = np.vstack([x_new, y_new, theta_new])


    def sensor_update(self, sensor_measurements):
        """Weight the particles according to the current map of the particle and the scan observations z.
        - sensor_measurements                : list of sensor measurements for the current timestep
        """

        robot_pose = self.pose

        # process each sensor measurement
        for measurement in sensor_measurements:
            self.weight = self.weight

def normalize_angle(angle):
    """Normalize the angle between -pi and pi"""

    while angle > pi:
        angle = angle - 2. * pi

    while angle < -pi:
        angle = angle + 2. * pi

    return angle