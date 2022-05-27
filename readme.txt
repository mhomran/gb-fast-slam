intialization
1- create a workspace
cd

2- create new workspace 
mkdir -p ~/slam_ws/src
cd ~/slam_ws/
catkin_make

3- clone turtlebot3
cd ~/slam_ws/src/
git clone -b melodic-devel https://github.com/ROBOTIS-GIT/turtlebot3_simulations.git
git clone -b melodic-devel https://github.com/ROBOTIS-GIT/turtlebot3
source devel/setup.bash
cd ~/slam_ws && catkin_make

launch and running
1- setup workspace 
source devel/setup.bash
export TURTLEBOT3_MODEL=burger

2- start ROS
roslaunch turtlebot3_gazebo turtlebot3_empty_world.launch

3- for motion commands
roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch
 
