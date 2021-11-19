cmake_minimum_required(VERSION 3.0.2)
project(oriented_point_sampling)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  sensor_msgs
  std_msgs
  pcl_conversions
)

find_package(PCL REQUIRED)

include_directories(
include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

catkin_package(
 INCLUDE_DIRS include
#  LIBRARIES oriented_point_sampling
 CATKIN_DEPENDS roscpp rospy sensor_msgs std_msgs pcl_conversions
 DEPENDS PCL
)

add_executable(PointCloudToPCLXYZ, src/pcltopcl2.cpp)
target_link_libraries(PointCloudToPCLXYZ, ${catkin_LIBRARIES} ${PCL_LIBRARIES})


