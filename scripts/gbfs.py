from glob import glob
import rospy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import numpy as np
from tf.transformations import euler_from_quaternion
from particle import Particle
from gbfs_algorithm import FastSlam
from odometry_data import OdometryData
import cv2 as cv

ms_to_sec = lambda x: x * (10**-3)
ms_to_ns = lambda x: x * (10**6)
s_to_ns = lambda x: x * (10**9)

INTERVAL_IN_MS = 10000
INTERVAL_IN_S = ms_to_sec(INTERVAL_IN_MS)
INTERVAL_IN_NS = ms_to_ns(INTERVAL_IN_MS)

FREQ = 1/INTERVAL_IN_S


t1 = 0 # t
pt1 = np.zeros(3) # odom at t-1
pt2 = np.zeros(3) # odom at t
r1 = 0 # initial rotation in radians counterclockwise
tr = 0 # translation
r2 = 0 # initial rotation in radians counterclockwise

gscan = False
godom = False

read_first_odom = False

def lidar_callback(scan):
    global gscan
    gscan = scan


def odom_callback(odom):
    global t1
    global pt1 
    global pt2 
    global r1 
    global tr
    global r2
    global godom
    global gscan
    global read_first_odom
    global fs
    
    t2 = int(str(odom.header.stamp)) 
    
    if t2 - t1 >= INTERVAL_IN_NS:
        # print(odom.pose)
        # advance time
        t1 = t2

        # advance readings
        pt1[:] = pt2[:]
            
        # get new readings  
        pt2[0] = odom.pose.pose.position.x
        pt2[1] = odom.pose.pose.position.y
        orientation_q = odom.pose.pose.orientation
        # new theta
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (_, _, theta) = euler_from_quaternion(orientation_list)
        pt2[2] = theta
        #print('New odometry:', pt2)
        #print('theta, theta (deg):', theta, np.degrees(theta))
        # only executed when there is a previous reading
        # not in the first callback
        if read_first_odom:
            # calculate s1, t, s2
            # translation
            tr = np.sqrt((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)
            ydiff = np.around(pt2[1]-pt1[1], 2)
            xdiff = np.around(pt2[0]-pt1[0], 2)
            # initial rotation
            r1 = np.arctan2(ydiff, xdiff) - pt1[2] + np.pi
            # final rotation
            r2 = pt2[2] - pt1[2] - r1
            godom = OdometryData(r1, tr, r2)
            #print(godom)
            #print('\n')

            #print(godom)
        if godom and gscan:
            fs.fast_slam(godom, gscan)
            godom = False # None
            gscan = False # None

        cv.waitKey(1)
        
        read_first_odom = True

def main():
    '''Main function of the program.

    This script calls all the required functions in the correct order.
    You can change the number of steps the filter runs for to ease the
    debugging. You should however not change the order or calls of any
    of the other lines, as it might break the framework.

    If you are unsure about the input and return values of functions you
    should read their documentation which tells you the expected dimensions.
    '''
    global gscan
    global godom
    global fs

    rospy.init_node('slam', anonymous=True)


    # initial timestamp
    t1 = int(str(rospy.Time.now())) # t

    # sensors and map subscribtion
    rospy.Subscriber("/scan", LaserScan, lidar_callback)
    rospy.Subscriber("/odom", Odometry, odom_callback)

    # how many particles
    num_particles = 1
    noise = [0.000, 0.000, 0.000]
    particles = [Particle(num_particles, noise) for _ in range(num_particles)]

    # set the axis dimensions
    fs = FastSlam(particles)

    rate = rospy.Rate(FREQ) # rate of the loop
    while not rospy.is_shutdown():
        # if godom and gscan:
        #     fs.fast_slam(godom, gscan)
        #     godom = False
        #     gscan = False

        # cv.waitKey(1)
        rate.sleep()



if __name__ == '__main__':
    main()