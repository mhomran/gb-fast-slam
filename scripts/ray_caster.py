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
  length_range=12, pixel_size=4, occ_th=.9):
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

  def cast(self, x, y, theta, angle_min, angle_max, show_rays=False, 
  show_collided=False, save_results=False):
    """
    Description: Cast a ray.
      
    Input:
      - x: the x coordinate on the map.
      - y: the y coordinate on the map.
      - theta: the heading direction on the map in degrees (1-360).
      0 degree makes the robot look horizontally right. The angle of the
      robot is incremented in the clock wise direction (i.e. 90 degree will
      make the robot look vertically down). (MAP FRAME)
      - angle_min: the minimum angle that the scan starts from.
      - angle_max: the maximum angle that the scan ends at.
      - show_rays: if True, show the image with the rays shown.
      - show_collided: if True, show the image with the collided points
      shown.
      - save_results: If True, save the measurements in a text file. 
    
    Output:
      - measurements: numpy array for the distances in pixels.
      The ray that's not collided has a distance of -1
    """
    
    if show_rays or show_collided:
      img = cv.cvtColor((self.map * 255).astype(np.uint8),cv.COLOR_GRAY2RGB)

    if save_results:
      file = open(f'Measurements_theta_{theta}_x_{x}_y_{y}_angle_accuracy_{self.angle_accuracy}.txt', 'w')
    
    start_angle = theta + angle_min
    end_angle = theta + angle_max
    start_len = 1
    # from metric unit to the number of pixels 
    end_len = (self.length_range * CM_IN_METER) // self.pixel_size 
    
    # +1 because the measurement at the original pose
    measurements = np.ones((end_angle-start_angle)//self.angle_accuracy + 1) * -1

    for i, angle in enumerate(range(start_angle, end_angle + 1, self.angle_accuracy)):
      for len in range(start_len, end_len + 1):
        tobe_checked_x = x + int(len * np.cos(np.radians(angle)))
        if not (0 <= tobe_checked_x < self.map.shape[1]):
          break 
        
        tobe_checked_y = y + int(len * np.sin(np.radians(angle)))
        if not (0 <= tobe_checked_y < self.map.shape[0]):
          break 
        
        tobe_checked = (tobe_checked_x, tobe_checked_y)
        
        if show_rays:
          cv.circle(img, tobe_checked, 1, BGR_RED_COLOR, -1)
        
        if self._is_collided(tobe_checked): 
          measurements[i] = len  
          
          if show_collided:
            cv.circle(img, tobe_checked, 2, BGR_BLUE_COLOR, -1)
          
          if save_results:
            file.write(f'theta:{angle}_x:{tobe_checked_x}_y:{tobe_checked_y}_distance:{measurements[i]}\n')

          # no need to progress with that ray since it's collided
          break 
      
      # save it even it's not collided
      if save_results and measurements[i] < 0:
        file.write(f'theta:{angle}_x:{tobe_checked_x}_y:{tobe_checked_y}_distance:{measurements[i]}\n')

    if show_rays or show_collided:
      cv.circle(img, (x, y), 2, BGR_GREEN_COLOR, 5)

      if save_results:
        cv.imwrite(f'Measurements_theta_{theta}_x_{x}_y_{y}_angle_accuracy_{self.angle_accuracy}.png', img)

      cv.imshow("Ray Casting", img)
      cv.waitKey(0)

    if save_results:
      
      file.close()
      
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
