#include "lib.h"
#include "pcltopcl2.h"

using namespace std;
using namespace cv;

void readMsgAndTackle(const sensor_msgs::PointCloudConstPtr &cloudin)
{   
    sensor_msgs::PointCloud2Ptr PointCloud2(new sensor_msgs::PointCloud2);
    convertPointCloudToPointCloud2(cloudin, PointCloud2);
    pcl::PointCloud<pcl::PointXYZ>::Ptr pclPointcloud(new pcl::PointCloud<pcl::PointXYZ>);
    convertPointCloud2ToPCLXYZ(PointCloud2, pclPointcloud);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	cloud = DeleteNAN(pclPointcloud);
	t[2]=clock();
	printf("%lf s\n",(double)(t[2]-t[1])/CLOCKS_PER_SEC);
	int pointNum=cloud->points.size();

	//k nearest neighbor search
	int maxK=30, k=30;
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
	printf("%lf s\n",(double)(t[4]-t[3])/CLOCKS_PER_SEC);
	cv::Mat normals;
	std::vector<int> randPoints;
	double randRate=0.1;
	normals=NormalEstimation(cloud, pointNum, maxK, k, indexKNN, distKNN, randRate, randPoints);

	cout<<endl;
	cout<<"Number of normals: "<<normals.rows<<endl;
	cout<<"Number of points: "<<cloud->points.size()<<endl;

	t[5]=clock();
	printf("%lf s\n",(double)(t[5]-t[4])/CLOCKS_PER_SEC);

	//find the points in horizontal plane
	//int normals_size=1;
	double ab, an, bn, cosr, interAngle;
	cv::Mat horiPoints;  //(1, 3, CV_64FC1); //the normals of the horizontal plane points 
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_hori (new pcl::PointCloud<pcl::PointXYZ>);
	//std::vector<int> indn;  //the index in normals
	for(int i=0; i<normals.rows; i++)
	{
		//compute the angle between normal and yAxis direction (0,1,0)
		ab=normals.at<double>(i,1);
		an=1;
		bn=sqrt(normals.at<double>(i,0)*normals.at<double>(i,0)+normals.at<double>(i,1)*normals.at<double>(i,1)+normals.at<double>(i,2)*normals.at<double>(i,2));
		cosr=ab/(an*bn);
		//cout<<M_PI<<endl;
		//cout<<acos(cosr)*180/M_PI<<endl;
		interAngle=acos(cosr)*180/M_PI;
		/*if(isnan(interAngle))
		{
			cout<<"i="<<i<<endl;
			cout<<normals.at<double>(i,0)<<" "<<normals.at<double>(i,1)<<" "<<normals.at<double>(i,2)<<endl;
		}*/
		if(interAngle<=7 || interAngle>=173)
		{
			cv::Mat hpRow(1, 3, CV_64FC1);
			hpRow.at<double>(0,0)=normals.at<double>(i,0);
			hpRow.at<double>(0,1)=normals.at<double>(i,1);
			hpRow.at<double>(0,2)=normals.at<double>(i,2);
			cloud_hori->push_back(cloud->points[randPoints[i]]);
			//indn.push_back(i);
			//cout<<endl;
			//cout<<hpRow<<endl;
			horiPoints.push_back(hpRow);
		}
	}
	cout<<endl;
	cout<<horiPoints.rows<<endl;
	cout<<cloud_hori->points.size()<<endl;

	t[6]=clock();
	printf("%lf s\n",(double)(t[6]-t[5])/CLOCKS_PER_SEC);

	//use one point ransac to find horizontal plane
	std::vector<std::vector<int> > inliers;
	std::vector<cv::Mat> planeModel;
	int minNum=10000;  //the minimum number of points in the plane
	//std::vector<int> isExist(cloud_hori->points.size(), 1);
	std::vector<int> isExist(cloud->points.size(), 1);
	
	while(accumulate(isExist.begin(), isExist.end(), 0)>minNum)
	{
		cv::Mat plane;  //first row is the point on the plane, second row is the normal
		std::vector<int> inl;
		//t[10]=clock();
		cout<<"isExist number: "<<endl<<accumulate(isExist.begin(), isExist.end(), 0)<<endl;
		//t[11]=clock();
		//printf("%lf s\n",(double)(t[11]-t[10])/CLOCKS_PER_SEC);
		t[10]=clock();
		plane=OnePointRANSAC(cloud, cloud_hori, horiPoints, horiPoints.rows, 0.99, 0.05, minNum, inl, isExist, randPoints);
		t[11]=clock();
		printf("OnePointRANSAC time: %lf s\n",(double)(t[11]-t[10])/CLOCKS_PER_SEC);
		if (countNonZero(plane)!=0)
		{
			planeModel.push_back(plane);
			inliers.push_back(inl);
			for(int i=0; i<inl.size(); i++)
			{
				isExist[inl[i]]=0;
			}
		}
		else
		{
			break;
		}
	}
	cout<<"Before aggregate planeModel size: "<<endl<<planeModel.size()<<endl;
	//aggregate the planes which should be the same one
	std::vector<std::vector<int> > inliers_fi;
	std::vector<cv::Mat> planeModel_fi;
	std::vector<int> indPlane;
	double disPlane;
	for(int i=0; i<planeModel.size(); i++)
	{
		indPlane.push_back(i);
	}
	for(int i=0; i<planeModel.size(); i++)
	{
		for(int j=i+1; j<planeModel.size(); j++)
		{
			disPlane=sqrt((planeModel[i].at<double>(0,0)-planeModel[j].at<double>(0,0))*(planeModel[i].at<double>(0,0)-planeModel[j].at<double>(0,0))+(planeModel[i].at<double>(0,1)-planeModel[j].at<double>(0,1))*(planeModel[i].at<double>(0,1)-planeModel[j].at<double>(0,1))+(planeModel[i].at<double>(0,2)-planeModel[j].at<double>(0,2))*(planeModel[i].at<double>(0,2)-planeModel[j].at<double>(0,2)));
			cout<<"i="<<i<<" j="<<j<<" distance: "<<disPlane<<endl;
			if (disPlane<0.01)
			{
				indPlane[j]=i;
			}
		}
	}
	for(int i=0; i<indPlane.size(); i++)
	{
		if(indPlane[i]==-1)
		{
			continue;
		}
		std::vector<int> vecfi;
		vecfi.insert(vecfi.end(), inliers[i].begin(), inliers[i].end());
		cv::Mat pm(2, 3, CV_64FC1);
		pm.at<double>(0,0)=planeModel[i].at<double>(0,0);
		pm.at<double>(0,1)=planeModel[i].at<double>(0,1);
		pm.at<double>(0,2)=planeModel[i].at<double>(0,2);
		int n=1;
		//pm.at<double>(1,0)=planeModel[i].at<double>(1,0);
		//pm.at<double>(1,1)=planeModel[i].at<double>(1,1);
		//pm.at<double>(1,2)=planeModel[i].at<double>(1,2);
		for(int j=i+1; j<indPlane.size(); j++)
		{
			if (indPlane[j]==indPlane[i])
			{
				pm.at<double>(0,0)=pm.at<double>(0,0)+planeModel[j].at<double>(0,0);
				pm.at<double>(0,1)=pm.at<double>(0,1)+planeModel[j].at<double>(0,1);
				pm.at<double>(0,2)=pm.at<double>(0,2)+planeModel[j].at<double>(0,2);
				n++;
				vecfi.insert(vecfi.end(), inliers[j].begin(), inliers[j].end());
				indPlane[j]=-1;
			}
		}
		pm.at<double>(0,0)=pm.at<double>(0,0)/n;
		pm.at<double>(0,1)=pm.at<double>(0,1)/n;
		pm.at<double>(0,2)=pm.at<double>(0,2)/n;
		cv::Mat pointsIn(vecfi.size(), 3, CV_64FC1);
		for(int j=0; j<vecfi.size(); j++)
		{
			pointsIn.at<double>(j,0)=cloud->points[vecfi[j]].x;
			pointsIn.at<double>(j,1)=cloud->points[vecfi[j]].y;
			pointsIn.at<double>(j,2)=cloud->points[vecfi[j]].z;
		}
		cv::Mat u0,w0,v0;
		SVD::compute(pointsIn, w0, u0, v0);
		pm.at<double>(1,0)=v0.at<double>(2,0);
		pm.at<double>(1,1)=v0.at<double>(2,1);
		pm.at<double>(1,2)=v0.at<double>(2,2);
		planeModel_fi.push_back(pm);
		inliers_fi.push_back(vecfi);
		indPlane[i]=-1;
	}
	cout<<"After aggregate planeModel size: "<<endl<<planeModel_fi.size()<<endl;
	t[8]=clock();
	printf("%lf s\n",(double)(t[8]-t[7])/CLOCKS_PER_SEC);
	
	ROS_INFO("pclPointCloud converted successfully");
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