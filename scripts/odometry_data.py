import numpy as np

class OdometryData(object):
    """Represents an odometry command.

    - r1: initial rotation in radians counterclockwise
    - tr: translation in meters
    - r2: final rotation in radians counterclockwise
    """
    def __init__(self, r1, tr, r2):
        self.r1 = self.normalize_angle(r1)
        self.tr = tr
        self.r2 = self.normalize_angle(r2)

    def normalize_angle(self, angle):
        """Normalize the angle between -pi and pi"""

        while angle > np.pi:
            angle = angle - 2 * np.pi

        while angle < -np.pi:
            angle = angle + 2 * np.pi

        return angle

    def __str__(self):
        # return "Odometry(r1 = {0} rad,\ntr = {1} m\n, r2 = {2} rad\n)".format(self.r1, self.tr, self.r2)
        return "Odometry(r1 = {0} deg,\ntr = {1} m\n, r2 = {2} deg\n)".format(np.degrees(self.r1), self.tr, np.degrees(self.r2))
