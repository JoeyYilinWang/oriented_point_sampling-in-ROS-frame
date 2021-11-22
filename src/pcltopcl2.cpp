#include "pcltopcl2.h"

using namespace std;

bool convertPointCloudToPointCloud2(const sensor_msgs::PointCloudConstPtr &cloudin, sensor_msgs::PointCloud2Ptr &PointCloud2)
{
    if (sensor_msgs::convertPointCloudToPointCloud2(*cloudin, *PointCloud2))
        return true;
    return false;
}

void  convertPointCloudToPointCloud2(const sensor_msgs::PointCloudConstPtr &cloudin)
{
    sensor_msgs::PointCloud2Ptr PointCloud2(new sensor_msgs::PointCloud2);
    if (sensor_msgs::convertPointCloudToPointCloud2(*cloudin, *PointCloud2))
        ROS_INFO("PointCloud are successfully converted to PointCloud2");
}

bool convertPointCloud2ToPCLXYZ(const sensor_msgs::PointCloud2ConstPtr &cloudin, pcl::PointCloud<pcl::PointXYZ>::Ptr &pclPointcloud)
{
  pcl::fromROSMsg(*cloudin, *pclPointcloud);
  return true;
}

void convertPointCloudToPCLXYZ(const sensor_msgs::PointCloudConstPtr &cloudin)
{   
    sensor_msgs::PointCloud2Ptr PointCloud2(new sensor_msgs::PointCloud2);
    convertPointCloudToPointCloud2(cloudin, PointCloud2);
    pcl::PointCloud<pcl::PointXYZ>::Ptr pclPointcloud(new pcl::PointCloud<pcl::PointXYZ>);
    convertPointCloud2ToPCLXYZ(PointCloud2, pclPointcloud);
    ROS_INFO("pclPointCloud converted successfully");
    return;
}
