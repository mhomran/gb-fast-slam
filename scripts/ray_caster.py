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

  def _is_collided(self, p):
    """
    Description: check if a certain point is collided with
    an obstacle in the map.

    Input:
      - p: the point to be checked (x, y).

    Output: 
      - is_collided: True if collided, False otherwise
    """
    is_collided = False

    if self.map[p[1]][p[0]] > self.occ_th:
      is_collided = True
    
    return is_collided

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

  def cast(self, pose, scan, show_rays=False, show_collided=False):
    """
    Description: Cast a ray.
      
    Input:
      - pose: robot pose (x, y, theta).
      - scan: LaserScan msg
      - show_rays: if True, show the image with the rays shown.
      - show_collided: if True, show the image with the collided points
      shown.
    
    Output:
      - measurements: numpy array for the distances in pixels.
      The ray that's not collided has a distance of -1
    """
    x, y, theta = pose

    angle_min = scan.angle_min
    angle_max = scan.angle_max
    angle_increment = int(np.degrees(scan.angle_increment))

    start_angle = theta + int(np.degrees(angle_min))
    end_angle = theta + int(np.degrees(angle_max))
    start_len = int(scan.range_min / self.pixel_size) 
    end_len = int(scan.range_max / self.pixel_size) 

    if show_rays or show_collided:
      img = self.map.copy()
      cv.normalize(img, img, 0, 255, cv.NORM_MINMAX)
      img = cv.cvtColor(img.astype(np.uint8),cv.COLOR_GRAY2RGB)
  
    
    # +1 because the measurement at the original pose
    measurements = np.ones((end_angle-start_angle)//angle_increment + 1) * -1

    for i, angle in enumerate(range(start_angle, end_angle, angle_increment)):
      for dst in range(start_len, end_len + 1):
        tobe_checked_x = x + int(dst * np.cos(np.radians(angle)))
        if not (0 <= tobe_checked_x < self.map.shape[1]):
          break 
        
        tobe_checked_y = y + int(dst * np.sin(np.radians(angle)))
        if not (0 <= tobe_checked_y < self.map.shape[0]):
          break 
        
        tobe_checked = (tobe_checked_x, tobe_checked_y)
        
        if show_rays:
          cv.circle(img, tobe_checked, 1, BGR_RED_COLOR, -1)
        
        if self._is_collided(tobe_checked): 
          measurements[i] = dst  
          
          if show_collided:
            cv.circle(img, tobe_checked, 2, BGR_BLUE_COLOR, -1)

          # no need to progress with that ray since it's collided
          break 
      

    if show_rays or show_collided:
      cv.circle(img, (x, y), 2, BGR_GREEN_COLOR, 5)
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
  show_rays=True, show_collided=True)
