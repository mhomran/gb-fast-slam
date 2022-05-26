"""
@file   : RayCaster.py
@brief  : a modified solution for the problem 4.1 in Assignment 1
@author : Mohamed Hassanin Mohamed
@data   : 25/03/2022
"""

import cv2 as cv
import numpy as np

BGR_RED_COLOR = (0, 0, 255)
BGR_BLUE_COLOR = (255, 0, 0)
BGR_GREEN_COLOR = (0, 255, 0)

GRAYSCALE_BLACK_COLOR = 0
GRAYSCALE_WHITE_COLOR = 255

CM_IN_METER = 100

class RayCaster:
  def __init__(self, map, angle_range=250, angle_accuracy=2, 
  length_range=12, pixel_size=.05, occ_th=.9):
    """
    Description: constructor for the ray caster.

    Input:
      - map: 2d numpy array of the map where the ray will be casted.
      - angle_range: the angle range to be scanned in degrees 
      from (1 to 360). Half of this range is for the right
      scan and the second half is for the left part.
      - angle_accuracy: An integer representing the step to
      increment the angle with in degrees.
      - length_range: the maximum length for the ray to be scanned
      in meters.
      - pixel_size: the pixel size in real world in centimeters.
      - occ_th: the threshold for the cell to be considered occupied.
    """
    self.angle_range = angle_range
    self.angle_accuracy = angle_accuracy
    self.length_range = length_range
    self.pixel_size = pixel_size
    self.map = map
    self.occ_th = occ_th

    # reuse
    self.angles = None
    self.cos_vec = None
    self.sin_vec = None
    self.dst_vec = None
    self.X_org = None
    self.Y_org = None
    self.measurements = None

  def _is_collided(self, p):
    """
    Description: check if (a) certain point(s) is/are collided with
    an obstacle in the map.

    Input:
      - p: the onject to be checked (x, y).

    Output: 
      - True if collided, False otherwise
    """
    X, Y = p
    
    return self.map[Y, X] > self.occ_th

  def _calculate_dist(self, p1, p2):
    """
    Description: calculate the euclidean distance.

    Input:
      - p1
      - p2

    output:
      - distance
    """
    dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    return dist

  def cast(self, pose, scan, show_rays=False):
    """
    Description: Cast a ray.
      
    Input:
      - pose: robot pose (x, y, theta).
      - scan: LaserScan msg
      - show_rays: if True, show the image with the rays shown.
      shown.
    
    Output:
      - measurements: numpy array for the distances in pixels.
      The ray that's not collided has a distance of -1
    """
    x, y, theta = pose
    x = int(x)
    y = int(y)
    theta = int(theta)

    if self.X_org is None:
      angle_min = scan.angle_min
      angle_max = scan.angle_max
      angle_increment = int(np.degrees(scan.angle_increment))

      start_angle = theta + int(np.degrees(angle_min))
      end_angle = theta + int(np.degrees(angle_max))
      start_len = int(scan.range_min / self.pixel_size) 
      end_len = int(scan.range_max / self.pixel_size) 
      
      self.measurements = np.ones((end_angle-start_angle)//angle_increment) * -1

      self.angles = np.arange(start_angle, end_angle, angle_increment)
      self.cos_vec = np.cos(np.radians(self.angles)).reshape(-1, 1)
      self.sin_vec = np.sin(np.radians(self.angles)).reshape(-1, 1)
      self.dst_vec = np.arange(start_len, end_len, 1).reshape(1, -1)
      self.X_org = np.matmul(self.cos_vec, self.dst_vec).astype(np.int)
      self.Y_org = np.matmul(self.sin_vec, self.dst_vec).astype(np.int)

    X = x + self.X_org
    Y = y + self.Y_org
    X_Y = (X, Y)

    X[X < 0] = 0
    X[X >= self.map.shape[1]] = self.map.shape[1]-1
    Y[Y < 0] = 0
    Y[Y >= self.map.shape[0]] = self.map.shape[0]-1

    collision = self._is_collided(X_Y) 
    dst_idx = np.argmax(collision, axis=-1)
    mask_collided = np.any(collision, axis=-1)
    dst = self.dst_vec.flatten()[dst_idx]
    measurements = self.measurements.copy()
    measurements[mask_collided] = dst[mask_collided]

    if show_rays:
      img = self.map.copy()
      cv.normalize(img, img, 0, 255, cv.NORM_MINMAX)
      img = cv.cvtColor(img.astype(np.uint8),cv.COLOR_GRAY2RGB)
    
      img[Y, X] = BGR_RED_COLOR
      cv.imshow("Ray Casting", img)

    return measurements    

if __name__ == '__main__':
  map = np.zeros((600, 600))
  for i in range(599):
    map[i][i] = 1
    map[i][i+1] = 1
    map[i][599-i] = .5
    map[i][598-i] = .5

  ray_caster = RayCaster(map)

  measurements = ray_caster.cast(x=350, y=80, theta=30,
  angle_max=90, angle_min=0,
  show_rays=True)
