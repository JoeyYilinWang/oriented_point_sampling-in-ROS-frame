#include <ros/ros.h>
#include <iostream>
#include <fstream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/filter.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <algorithm>
#include <vector>
#include <string>
#include <numeric>
#include <cmath>
#include <time.h>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>

clock_t t[100];

pcl::PointCloud<pcl::PointXYZ>::Ptr ReadPCD(char* filePath);
pcl::PointCloud<pcl::PointXYZ>::Ptr DeleteNAN(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
void PointsKNN(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, int pointNum, int maxK, int** indexKNN, double** distKNN);
cv::Mat NormalEstimation(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, int pointNum, int maxK, int k, int** indexKNN, double** distKNN, double randRate, std::vector<int> &randPoints);
cv::Mat OnePointRANSAC(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_hori, cv::Mat horiPoints, int maxIteration, double p, double disThreshold, int fitNum, std::vector<int> &best_inliers, std::vector<int> isExist, std::vector<int> randPoints);




