import rospy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import numpy as np
from tf.transformations import euler_from_quaternion
from particle import Particle
from gbfs_algorithm import FastSlam
from odometry_data import OdometryData

ms_to_sec = lambda x: x * (10**-3)
ms_to_ns = lambda x: x * (10**6)
s_to_ns = lambda x: x * (10**9)

INTERVAL_IN_MS = 200
INTERVAL_IN_S = ms_to_sec(INTERVAL_IN_MS)
INTERVAL_IN_NS = ms_to_ns(INTERVAL_IN_MS)

FREQ = 1/INTERVAL_IN_S


t1 = 0 # t
pt1 = np.zeros(3) # odom at t-1
pt2 = np.zeros(3) # odom at t
r1 = 0 # initial rotation in radians counterclockwise
tr = 0 # translation
r2 = 0 # initial rotation in radians counterclockwise


def lidar_callback(data):
    rospy.loginfo(data.ranges)


def odom_callback(odom):
    global t1

    t2 = int(str(odom.header.stamp)) 

    if t2 - t1 >= INTERVAL_IN_NS:
        t1 = t2

        pt1[:] = pt2[:]
        pt2[0] = odom.pose.pose.position.x
        pt2[1] = odom.pose.pose.position.y
        orientation_q = odom.pose.pose.orientation

        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (_, _, yaw) = euler_from_quaternion(orientation_list)

        pt2[2] = yaw

        r1 = np.arctan2(pt2[1]-pt1[1], pt2[0]-pt1[0]) - pt1[2]
        tr = np.sqrt((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)
        r2 = pt2[2] - pt1[2] - r1

        print("r1: " + str(r1) + " tr:" + str(tr) + " r2:" + str(r2))

def main():
    '''Main function of the program.

    This script calls all the required functions in the correct order.
    You can change the number of steps the filter runs for to ease the
    debugging. You should however not change the order or calls of any
    of the other lines, as it might break the framework.

    If you are unsure about the input and return values of functions you
    should read their documentation which tells you the expected dimensions.
    '''

    rospy.init_node('slam', anonymous=True)


    # initial timestamp
    t1 = int(str(rospy.Time.now())) # t

    # sensors and map subscribtion
    # rospy.Subscriber("/scan", LaserScan, lidar_callback)
    rospy.Subscriber("/odom", Odometry, odom_callback)
    # map_pub = rospy.Publisher('/map', PointCloud2, queue_size=10)

    # how many particles
    num_particles = 100

    noise = [0.005, 0.01, 0.005]

    # initialize the particles array
    num_landmarks = 0
    particles = [Particle(num_particles, num_landmarks, noise) for _ in range(num_particles)]

    # set the axis dimensions
    fs = FastSlam(particles)

    rate = rospy.Rate(FREQ) # rate of the loop
    while not rospy.is_shutdown():
        odom = OdometryData(r1, tr, r2)
        sensor = [1, 2, 3]
        fs.fast_slam(odom, sensor)

        rate.sleep()



if __name__ == '__main__':
    main()