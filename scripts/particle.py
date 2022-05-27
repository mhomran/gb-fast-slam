import numpy as np
import random
from math import pi, sin, cos
from ray_caster import RayCaster

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

        # initialize the landmarks aka the map
        self.prior = 0.5
        self.map = np.ones((600, 600)) * self.prior

        # sensor model
        self.ray_caster = RayCaster(self.map, pixel_size=pixel_size)

        self.pixel_size = pixel_size



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
        print(sin(self.pose[2] + odom.r1))
        y_new = self.pose[1] + odom.tr * sin(self.pose[2] + odom.r1)
        theta_new = normalize_angle(self.pose[2] + odom.r1 + odom.r2)

        self.pose = np.array([x_new, y_new, theta_new])
        
        print(round(self.pose[0], 3), round(self.pose[1], 3), round(np.degrees(self.pose[2]), 3))



    def sensor_update(self, scan):
        """
        Description: Weight the particles according to the current map
         of the particle and the scan observations z.
        
        Input:
            - scan : LaserMsg
        """

        self.ray_caster.cast(self.pose, scan,
        show_rays=True)


def normalize_angle(angle):
    """Normalize the angle between -pi and pi"""

    while angle > pi:
        angle = angle - 2 * pi

    while angle < -pi:
        angle = angle + 2 * pi

    return angle