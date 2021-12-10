#include "lib.h"
#include "pcltopcl2.h"

using namespace std;
using namespace cv;

const double randRate = 0.8; 
const int maxK = 10;
const int minNum = 30;


void readMsgAndTackle(const sensor_msgs::PointCloudConstPtr &cloudin)
{   
    sensor_msgs::PointCloud2Ptr PointCloud2(new sensor_msgs::PointCloud2);
    convertPointCloudToPointCloud2(cloudin, PointCloud2);
    pcl::PointCloud<pcl::PointXYZ>::Ptr pclPointcloud(new pcl::PointCloud<pcl::PointXYZ>);
    convertPointCloud2ToPCLXYZ(PointCloud2, pclPointcloud);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	cloud = DeleteNAN(pclPointcloud);
	t[2]=clock();
	// printf("%lf s\n",(double)(t[2]-t[1])/CLOCKS_PER_SEC);
	int pointNum=cloud->points.size();
	cout << "After filtered bad points, the num of residual points is: " << pointNum << endl;

	//k nearest neighbor search
	int k=maxK;
  	int** indexKNN;
  	indexKNN=new int*[pointNum];
  	for(int i=0; i<pointNum; i++)
  	{
  		indexKNN[i]=new int[maxK];
	}
	double** distKNN;
	distKNN=new double*[pointNum];
	for(int i=0; i<pointNum; i++)
  	{
  		distKNN[i]=new double[maxK];
	}
	PointsKNN(cloud, pointNum, maxK, indexKNN, distKNN);
	t[4]=clock();
	// printf("%lf s\n",(double)(t[4]-t[3])/CLOCKS_PER_SEC);
	cv::Mat normals;
	std::vector<int> randPoints;

	normals=NormalEstimation(cloud, pointNum, maxK, k, indexKNN, distKNN, randRate, randPoints);

	cout<<endl;
	cout<<"Number of normals: "<<normals.rows<<endl;

	t[5]=clock();
	// printf("%lf s\n",(double)(t[5]-t[4])/CLOCKS_PER_SEC);

	//find the points in horizontal plane
	//int normals_size=1;
	double ab, an, bn, cosr, interAngle;

	cv::Mat sampledPoints;
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_sampled (new pcl::PointCloud<pcl::PointXYZ>);
	for (int i=0; i< normals.rows; i++)
	{
		cv::Mat pRow(1, 3, CV_64FC1);
		pRow.at<double>(0,0) = normals.at<double>(i,0);
		pRow.at<double>(0,1) = normals.at<double>(i,1);
		pRow.at<double>(0,2) = normals.at<double>(i,2);
		cloud_sampled->push_back(cloud->points[randPoints[i]]);
		sampledPoints.push_back(pRow);
	}
	cout << "the size of sampled points from input is: " << sampledPoints.rows << endl;

	//use one point ransac to find horizontal plane
	std::vector<std::vector<int> > inliers;
	std::vector<cv::Mat> planeModel;
	//std::vector<int> isExist(cloud_hori->points.size(), 1);
	std::vector<int> isExist(cloud->points.size(), 1);
	
	// only output one plane
	cv::Mat plane;
	std::vector<int>  inl;
	plane = OnePointRANSAC(cloud, cloud_sampled, sampledPoints, sampledPoints.rows, 0.99, 0.05, minNum, inl, isExist, randPoints);
	if (countNonZero(plane) != 0)
	{
		cout << "plane detected" << endl;
	}
	
	ROS_INFO("successful!!!!");
    return;
}


int main(int argc, char **argv)
{
    ros::init(argc, argv, "opsfrommsg");  
	ros::NodeHandle n;
    /* define a subscriber named _subCloud, and subscribe topic '/sensor_msg::PointCloud'*/
    ros::Subscriber _subCloud;
    _subCloud = n.subscribe<sensor_msgs::PointCloud>
      ("/radar_pcl", 1, &readMsgAndTackle);
    ros::spin();
    return 0;
}