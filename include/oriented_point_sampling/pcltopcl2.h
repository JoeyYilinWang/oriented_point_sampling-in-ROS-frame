#include <ros/ros.h>
#include <iostream>
#include <fstream>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <sensor_msgs/point_cloud_conversion.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>

bool convertPointCloudToPointCloud2(const sensor_msgs::PointCloudConstPtr &cloudin, sensor_msgs::PointCloud2Ptr &PointCloud2);
void convertPointCloudToPointCloud2(const sensor_msgs::PointCloudConstPtr &cloudin);
bool convertPointCloud2ToPCLXYZ(const sensor_msgs::PointCloud2ConstPtr &cloudin, pcl::PointCloud<pcl::PointXYZ>::Ptr &pclPointcloud);
void convertPointCloudToPCLXYZ(const sensor_msgs::PointCloudConstPtr &cloudin);
