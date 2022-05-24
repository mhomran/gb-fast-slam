class OdometryData(object):
    """Represents an odometry command.

    - r1: initial rotation in radians counterclockwise
    - tr: translation in meters
    - r2: final rotation in radians counterclockwise
    """
    def __init__(self, r1, tr, r2):
        self.r1 = r1
        self.tr = tr
        self.r2 = r2

    def __str__(self):
        return "Odometry(r1 = {0} rad, tr = {1} m, r2 = {2} rad)".format(self.r1, self.tr, self.r2)
