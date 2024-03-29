"""
@file   : RayCaster.py
@brief  : a modified solution for the problem 4.1 in Assignment 1
@author : Mohamed Hassanin Mohamed
@data   : 25/03/2022
"""

from math import degrees
import cv2 as cv
import numpy as np

BGR_RED_COLOR = (0, 0, 255)
BGR_BLUE_COLOR = (255, 0, 0)
BGR_GREEN_COLOR = (0, 255, 0)

GRAYSCALE_BLACK_COLOR = 0
GRAYSCALE_WHITE_COLOR = 255

CM_IN_METER = 100

class ScanObj:
  def __init__(self, m, n, i, rn, rm):
    self.angle_max = m
    self.angle_min = n
    self.angle_increment = i
    self.range_min = rn
    self.range_max = rm

class RayCaster:
  def __init__(self, map, angle_range=250, angle_accuracy=2, 
  length_range=12, pixel_size=.05, occ_th=.9, offset_x=300, offset_y=300):
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
    self.offset_x = offset_x
    self.offset_y = offset_y

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
    x = int(x / self.pixel_size) + self.offset_x
    y = int(y / self.pixel_size) + self.offset_y
    ranges = np.array(scan.ranges) / self.pixel_size

    if self.X_org is None:
      angle_min = scan.angle_min
      angle_max = scan.angle_max
      angle_increment = int(np.degrees(scan.angle_increment))

      start_angle = int(np.degrees(angle_min))
      end_angle = int(np.degrees(angle_max))
      start_len = int(scan.range_min / self.pixel_size) 
      end_len = int(scan.range_max / self.pixel_size) 
      
      self.measurements = np.ones((end_angle-start_angle)//angle_increment) * -1

      self.angles = np.arange(start_angle, end_angle, angle_increment)
      self.cos_vec = np.cos(np.radians(self.angles)).reshape(-1, 1)
      self.sin_vec = np.sin(np.radians(self.angles)).reshape(-1, 1)
      self.dst_vec = np.arange(start_len, end_len, 1).reshape(1, -1)
      self.X_org = np.matmul(self.cos_vec, self.dst_vec).astype(np.int)
      self.Y_org = np.matmul(self.sin_vec, self.dst_vec).astype(np.int)


    X = x + self.X_org * np.cos(theta) - self.Y_org * np.sin(theta)
    Y = y + self.X_org * np.sin(theta) + self.Y_org * np.cos(theta)
    X = X.astype(np.int)
    Y = Y.astype(np.int)
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
      img[img < self.occ_th] = 0
      img[img > self.occ_th] = 255

      img = cv.cvtColor(img.astype(np.uint8),cv.COLOR_GRAY2RGB)
      # img[Y, X] = BGR_RED_COLOR
      n = np.arange(X.shape[0])
      X = X[n, dst_idx]
      Y = Y[n, dst_idx]
      # img[Y, X] = BGR_BLUE_COLOR
      X = X - 1
      Y = Y - 1
      # img[Y, X] = BGR_BLUE_COLOR

      # show robot
      cv.circle(img, (x, y), 15, BGR_GREEN_COLOR, 2)
      pt1 = (x, y)
      pt2 = (x + 15 * np.cos(theta), y + 15 * np.sin(theta))
      cv.line(img, pt1, pt2, BGR_GREEN_COLOR, 2)
      
      rot = int(round(np.degrees(-theta)))
      scan_epsx = ranges * np.roll(self.cos_vec.reshape(-1), rot)
      scan_epsy = ranges * np.roll(self.sin_vec.reshape(-1), rot)

      scan_epsx = x + scan_epsx
      scan_epsy = y + scan_epsy
      scan_epsx = scan_epsx.astype(np.int)
      scan_epsy = scan_epsy.astype(np.int)

      scan_epsx[scan_epsx < 0] = 0
      scan_epsx[scan_epsx >= self.map.shape[1]] = self.map.shape[1]-1
      scan_epsy[scan_epsy < 0] = 0
      scan_epsy[scan_epsy >= self.map.shape[0]] = self.map.shape[0]-1
    
      # for i, j in zip(scan_epsx, scan_epsy):
      #   cv.circle(img, (i, j), 2, BGR_GREEN_COLOR, 5)

      img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
      img = cv.flip(img, 0)
      cv.imshow("Ray Casting", img)

    return measurements, mask_collided, (scan_epsx, scan_epsy)
   

if __name__ == '__main__':
  map = np.zeros((600, 600))
  for i in range(599):
    map[i][i] = 1
    map[i][i+1] = 1
    map[i][599-i] = .5
    map[i][598-i] = .5

  ray_caster = RayCaster(map= map, pixel_size= 1)

  measurements = ray_caster.cast(pose= np.array([50, 300, 0]), 
    scan= ScanObj(np.radians(360), 0, np.radians(1), 0, 160), show_rays=True)

  cv.waitKey(0)
