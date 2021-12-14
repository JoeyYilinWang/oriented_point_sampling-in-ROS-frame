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
	for(int i=0; i<pointNum; i++)
	{
		randPoints.push_back(i);
	}
	/* after shuffled, the randPoints showed in this way  
	  0   1   2   3   4  ...
	 079|001|010|100|051|---|
	*/
	std::random_shuffle(randPoints.begin(), randPoints.end());
	pointNum=pointNum*randRate;
	for(int i=0; i<pointNum; i++)
	{
		referP.at<double>(0,0)=cloud->points[randPoints[i]].x;
		referP.at<double>(1,0)=cloud->points[randPoints[i]].y;
		referP.at<double>(2,0)=cloud->points[randPoints[i]].z;
		nwrMat=cv::Mat::zeros(cv::Size(3,3),CV_64FC1);
		nn=0;
		/* there are two ways to jump out this loop.
		1) j = maxK
		2) nn == k, happens only when k <= maxK. Because if  k > maxK, it jump out only when j = maxK  
		*/
		for(int j=0; j<maxK; j++)
		{
			// some points don't have enough neighbors, so it is needed to check at this point.
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
	return (normals);
}

cv::Mat OnePointRANSAC(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_sampled, cv::Mat sampledPoints, int maxIteration, double p, double disThreshold, int fitNum, std::vector<int> &best_inliers, std::vector<int> isExist, std::vector<int> randPoints)
{
	int iter=0;
	cv::Mat planeModel = cv::Mat::zeros(2, 3, CV_64FC1); //The first row is the point on the plane, the second row is the normal of the plane
	int inNum=0;
	std::vector<int> inliers;
	int randInd;
	double e;
	while(iter<maxIteration)
	{
		randInd=rand()%cloud_sampled->points.size();
		inliers.clear();
		// search for inliners in all points.
		for(int i=0; i<cloud->points.size(); i++)
		{
			// i should not be the index of sampled points. because sampled point is certainly the inliner of itself.
			if(i==randPoints[randInd] || isExist[i]==0)
			{
				continue;
			}
			//compute the distance between every point to the plane
			double x=cloud->points[i].x-cloud_sampled->points[randInd].x;
			double y=cloud->points[i].y-cloud_sampled->points[randInd].y;
			double z=cloud->points[i].z-cloud_sampled->points[randInd].z;
			double xn=sampledPoints.at<double>(randInd,0);
			double yn=sampledPoints.at<double>(randInd,1);
			double zn=sampledPoints.at<double>(randInd,2);
			double norm_n=sqrt(xn*xn+yn*yn+zn*zn);
			double dis=abs(x*xn+y*yn+z*zn)/norm_n;
			if (dis<disThreshold)
			{
				inliers.push_back(i);
			}
		}
		if (inliers.size()>fitNum)
		{
			// As illustrated before, sampled point is certainly the inlier of itself.
			inliers.push_back(randPoints[randInd]);
			if (inliers.size()>inNum)
			{
				inNum=inliers.size();
				best_inliers=inliers;
				cout<<"inliers number: "<<inliers.size()<<endl;
				/*
				e is the percentage of outliers in all points. if num of inliers larger, e will become smaller, and resultant maxIteration wil be larger.
				It means if more inliers founded, we should increase the maxIteration, which further means we have more confidence to find more inliers. 
				If less inliers founed, we should decrease the maxIteration, which further means we have less confidence to find more inliers.
				The reason for this manipulation is that we should keep the convergence of this process. 
				*/
				e=1-(double(inliers.size())/cloud->points.size());
				maxIteration=log(1-p)/log(1-(1-e));
			}
		}
		iter++;
	}
	// means every random sampled point have inliers less than fitNum, which further means we have no plane detected!
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
	// get the plane center point by computing the means of three of pointsInlier cols.
	for(int i=0; i<3; i++)
	{
		col=pointsInlier.col(i);
		colMean=mean(col)[0];
		planeModel.at<double>(0,i)=colMean;
	}
	return planeModel;
}