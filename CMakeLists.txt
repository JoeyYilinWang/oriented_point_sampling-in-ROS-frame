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
find_package(OpenCV REQUIRED)



catkin_package(
)


include_directories(
  include/oriented_point_sampling
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

# add_library(needed SHARED ${PROJECT_SOURCE_DIR}/src/lib.cpp)
# add_library(pcltopcl2 ${PROJECT_SOURCE_DIR}/src/pcltopcl2.cpp)

link_directories(/home/joey/workspace/catkin_ws/devel/lib)
# add_executable(cccccccconversion src/pcltopcl2.cpp) 
# target_link_libraries(cccccccconversion ${catkin_LIBRARIES} ${PCL_LIBRARIES} )

add_executable(opsfrommsg src/opsFromMsg.cpp)
target_link_libraries(opsfrommsg libneeded.so libpcltopcl2.so ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS})