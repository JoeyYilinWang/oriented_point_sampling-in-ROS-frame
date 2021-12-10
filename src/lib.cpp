#include "lib.h"

using namespace std;
using namespace cv;

//read 3D points from pcd file
pcl::PointCloud<pcl::PointXYZ>::Ptr ReadPCD(char* filePath)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
	// pcl::io library used for load PCD file 
	if (pcl::io::loadPCDFile<pcl::PointXYZ> (filePath, *cloud) == -1)
	{
		PCL_ERROR("Couldn't read file box_cloud.pcd \n");
	}
	return (cloud);
}

// delete the bad points.
pcl::PointCloud<pcl::PointXYZ>::Ptr DeleteNAN(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
	std::vector<int> indices;
	pcl::PointCloud<pcl::PointXYZ>::Ptr out(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::removeNaNFromPointCloud(*cloud, *out, indices);
	return out;
}

//k nearest neighbor search
void PointsKNN(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, int pointNum, int maxK, int** indexKNN, double** distKNN)
{
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	kdtree.setInputCloud (cloud);
	pcl::PointXYZ searchPoint;
	std::vector<int> pointIdxNKNSearch(maxK);
  	std::vector<float> pointNKNSquaredDistance(maxK);
	for(int i=0; i<pointNum; i++)
	{
		searchPoint.x=cloud->points[i].x;
		searchPoint.y=cloud->points[i].y;
		searchPoint.z=cloud->points[i].z;
		if(kdtree.nearestKSearch (searchPoint, maxK, pointIdxNKNSearch, pointNKNSquaredDistance)>0)
		{
			for(int j=0; j<maxK; j++)
			{
				indexKNN[i][j]=pointIdxNKNSearch[j];
				distKNN[i][j]=pointNKNSquaredDistance[j];
			}
			
	}
		}
}

//compute sigma and NWR matrix, then get the normals of the points
cv::Mat NormalEstimation(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, int pointNum, int maxK, int k, int** indexKNN, double** distKNN, double randRate, std::vector<int> &randPoints)
{
	cv::Mat diffM(3, 1, CV_64FC1);
	cv::Mat referP(3, 1, CV_64FC1);
	cv::Mat neighP(3, 1, CV_64FC1);
	cv::Mat nwrSample(3, 3, CV_64FC1);
	cv::Mat nwrMat(3, 3, CV_64FC1);
	cv::Mat normals(pointNum*randRate, 3, CV_64FC1);
	double sigma=0.2;
	int nn;    //number of nonzero distance
	t[10]=clock();
	for(int i=0; i<pointNum; i++)
	{
		randPoints.push_back(i);
	}
	std::random_shuffle(randPoints.begin(), randPoints.end());
	pointNum=pointNum*randRate;
	for(int i=0; i<pointNum; i++)
	{
		referP.at<double>(0,0)=cloud->points[randPoints[i]].x;
		referP.at<double>(1,0)=cloud->points[randPoints[i]].y;
		referP.at<double>(2,0)=cloud->points[randPoints[i]].z;
		nwrMat=cv::Mat::zeros(cv::Size(3,3),CV_64FC1);
		nn=0;
		for(int j=0; j<maxK; j++)
		{
			if(distKNN[randPoints[i]][j]!=0)
			{
				neighP.at<double>(0,0)=cloud->points[indexKNN[randPoints[i]][j]].x;
				neighP.at<double>(1,0)=cloud->points[indexKNN[randPoints[i]][j]].y;
				neighP.at<double>(2,0)=cloud->points[indexKNN[randPoints[i]][j]].z;
				diffM=neighP-referP;
				nwrSample=diffM*diffM.t();
				nwrSample=nwrSample/(distKNN[randPoints[i]][j]*distKNN[randPoints[i]][j]);
				nwrSample=nwrSample*exp(-(distKNN[randPoints[i]][j]*distKNN[randPoints[i]][j])/(2*sigma*sigma));
				nwrMat=nwrMat+nwrSample;
				nn++;
			}
			if(nn==k)
			{
				break;
			}
			
		}
		//compute eigen value
		cv::Mat eigenvalue, eigenvector;
		cv::eigen(nwrMat, eigenvalue, eigenvector);
		normals.at<double>(i,0)=eigenvector.at<double>(2,0);
		normals.at<double>(i,1)=eigenvector.at<double>(2,1);
		normals.at<double>(i,2)=eigenvector.at<double>(2,2);
	}
	t[11]=clock();
	return (normals);
}

cv::Mat OnePointRANSAC(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_hori, cv::Mat horiPoints, int maxIteration, double p, double disThreshold, int fitNum, std::vector<int> &best_inliers, std::vector<int> isExist, std::vector<int> randPoints)
{
	int iter=0;
	cv::Mat planeModel = cv::Mat::zeros(2, 3, CV_64FC1); //The first row is the point on the plane, the second row is the normal of the plane
	int inNum=0;
	std::vector<int> inliers;
	int randInd;
	double e;
	while(iter<maxIteration)
	{
		randInd=rand()%cloud_hori->points.size();
		inliers.clear();
		for(int i=0; i<cloud->points.size(); i++)
		{
			
			if(i==randPoints[randInd] || isExist[i]==0)
			{
				continue;
			}
			//compute the distance between every point to the plane
			double x=cloud->points[i].x-cloud_hori->points[randInd].x;
			double y=cloud->points[i].y-cloud_hori->points[randInd].y;
			double z=cloud->points[i].z-cloud_hori->points[randInd].z;
			double xn=horiPoints.at<double>(randInd,0);
			double yn=horiPoints.at<double>(randInd,1);
			double zn=horiPoints.at<double>(randInd,2);
			double norm_n=sqrt(xn*xn+yn*yn+zn*zn);
			double dis=abs(x*xn+y*yn+z*zn)/norm_n;
			if (dis<disThreshold)
			{
				inliers.push_back(i);
			}
		}
		if (inliers.size()>fitNum)
		{
			inliers.push_back(randPoints[randInd]);
			if (inliers.size()>inNum)
			{
				inNum=inliers.size();
				best_inliers=inliers;
				cout<<"inliers number: "<<inliers.size()<<endl;
				e=1-(double(inliers.size())/cloud->points.size());
				cout<<"e: "<<e<<endl;
				maxIteration=log(1-p)/log(1-(1-e));
			}
		}
		iter++;
	}
	cout<<"iteration times: "<<iter<<endl;
	cout<<"maxIteration: "<<maxIteration<<endl;
	//if do not find any plane
	if (inNum==0)
	{
		return planeModel;
	}
	//re-estimate the plane using all the inliers points
	cv::Mat pointsInlier(best_inliers.size(), 3,  CV_64FC1);
	for(int i=0; i<best_inliers.size(); i++)
	{
		pointsInlier.at<double>(i,0)=cloud->points[best_inliers[i]].x;
		pointsInlier.at<double>(i,1)=cloud->points[best_inliers[i]].y;
		pointsInlier.at<double>(i,2)=cloud->points[best_inliers[i]].z;
	}
	cv::Mat u,w,v;
	SVD::compute(pointsInlier, w, u, v);
	planeModel.at<double>(1,0)=v.at<double>(2,0);
	planeModel.at<double>(1,1)=v.at<double>(2,1);
	planeModel.at<double>(1,2)=v.at<double>(2,2);

	double colMean;
	cv::Mat col(best_inliers.size(), 1,  CV_64FC1);
	for(int i=0; i<3; i++)
	{
		col=pointsInlier.col(i);
		colMean=mean(col)[0];
		planeModel.at<double>(0,i)=colMean;
	}
	return planeModel;
}