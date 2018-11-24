#include "opencv2/opencv.hpp"
#include "carID_detect.h"

//using namespace cv;
//#include <iostream>

int main()
{
	FileStorage fs("ann_xml.xml", FileStorage::WRITE); ;
	if (!fs.isOpened())
	{
		std::cerr << "failed to open " << std::endl;
	}

	Mat  trainData;
	Mat classes = Mat::zeros(1, 1700, CV_8UC1);
	char path[90];
	Mat img_read;
	for (int i = 0; i < 34; i++)  //第i类
	{
		for (int j = 1; j < 51; ++j)  //i类中第j个
		{
			sprintf(path, "F:\\opencv作品\\ANN_train\\ANN_train\\charSamples\\%d\\%d (%d).png", i, i, j);
			img_read = imread(path, -1);

			equalizeHist(img_read, img_read);
			Mat img_threshold;
			threshold(img_read, img_threshold, 180, 255, CV_THRESH_BINARY);


			Mat img_contours;
			img_threshold.copyTo(img_contours);
			vector < vector <Point> > contours;
			findContours(img_contours, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
			vector<Rect> ploy_rects(1);
			ploy_rects[0] = boundingRect(contours[0]);
			Mat dst_rect(Mat(img_threshold, ploy_rects[0]));

			Mat dst_mat;
			Mat train_mat(2, 3, CV_32FC1);
			int length;
			Point2f srcTri[3];
			Point2f dstTri[3];

			srcTri[0] = Point2f(0, 0);
			srcTri[1] = Point2f(dst_rect.cols - 1, 0);
			srcTri[2] = Point2f(0, dst_rect.rows - 1);
			length = dst_rect.rows > dst_rect.cols ? dst_rect.rows : dst_rect.cols;
			dstTri[0] = Point2f(0.0, 0.0);
			dstTri[1] = Point2f(length, 0.0);
			dstTri[2] = Point2f(0.0, length);
			train_mat = getAffineTransform(srcTri, dstTri);
			dst_mat = Mat::zeros(length, length, img_threshold.type());
			warpAffine(dst_rect, dst_mat, train_mat, dst_mat.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(0));
			resize(dst_mat, dst_mat, Size(30,30));  //尺寸调整为30*30

			

			Mat dst_feature;
			features(dst_mat, dst_feature,20); //生成1*460特征向量

			trainData.push_back(dst_feature);
			classes.at<uchar>(i * 50 + j - 1) = i;
		}

	}

	Mat trainClasses;
	trainClasses.create(trainData.rows, 34, CV_32FC1);
	for (int i = 0; i< trainData.rows; i++)
	{
		for (int k = 0; k< trainClasses.cols; k++)
		{
			if (k == (int)classes.at<uchar>(i))
			{
					trainClasses.at<float>(i, k) = 0.9;
			}
			else
				trainClasses.at<float>(i, k) = 0.1;
		}
	}

	Ptr<ANN_MLP> ann = ANN_MLP::create();
	Mat layerSizes(1, 3, CV_32SC1);
	layerSizes.at<int>(0) = trainData.cols;
	layerSizes.at<int>(1) = 48; //隐藏神经元数
	layerSizes.at<int>(2) = 34; //样本类数为34
	ann->setLayerSizes(layerSizes);
	//激活函数
	ann->setActivationFunction(ANN_MLP::SIGMOID_SYM);
	//MLP的训练方法
	ann->setTrainMethod(ANN_MLP::BACKPROP, 0.1,0.1);
	Ptr<TrainData> traindata = TrainData::create(trainData, ROW_SAMPLE, trainClasses);
	bool trained=ann->train(traindata);
	if (trained)   
	{  
		ann->save("F:\\opencv作品\\ANN_train\\ANN_train\\bp_param.xml");
	} 
	/*  opencv 2.版本
	Mat layerSizes(1, 3, CV_32SC1);
	layerSizes.at<int>(0) = trainData.cols;
	layerSizes.at<int>(1) = nlayers; //隐藏神经元数，可设为3
	layerSizes.at<int>(2) = numCharacters; //样本类数为34
	ann.create(layerSizes, ANN_MLP::SIGMOID_SYM, 1, 1);  //初始化ann
	*/


	
	for (int j =0; j < 34; ++j)  //i类中第j个
	{
		sprintf(path, "F:\\opencv作品\\ANN_train\\ANN_train\\测试样本\\%d.bmp",j);
		img_read = imread(path, -1);

		Mat img_threshold;
		threshold(img_read, img_threshold, 180, 255, CV_THRESH_BINARY);

		Mat dst_mat;
		Mat train_mat(2, 3, CV_32FC1);
		int length;
		Point2f srcTri[3];
		Point2f dstTri[3];
		srcTri[0] = Point2f(0, 0);
		srcTri[1] = Point2f(img_threshold.cols - 1, 0);
		srcTri[2] = Point2f(0, img_threshold.rows - 1);
		length = img_threshold.rows > img_threshold.cols ? img_threshold.rows : img_threshold.cols;
		dstTri[0] = Point2f(0.0, 0.0);
		dstTri[1] = Point2f(length, 0.0);
		dstTri[2] = Point2f(0.0, length);
		train_mat = getAffineTransform(srcTri, dstTri);
		dst_mat = Mat::zeros(length, length, img_threshold.type());
		warpAffine(img_threshold, dst_mat, train_mat, dst_mat.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(0));
		resize(dst_mat, dst_mat, Size(30, 30));  //尺寸调整为20*20

		Mat dst_feature;
		features(dst_mat, dst_feature, 20); //生成1*480特征向量

		Mat output(1, 34, CV_32FC1); //1*34矩阵
		ann->predict(dst_feature, output);
		Point maxLoc;
		double maxVal;
		minMaxLoc(output, 0, &maxVal, 0, &maxLoc);
		int result = maxLoc.x;
		cout << result << endl;
	}
		waitKey(0);	

	//fs << "TrainingData" << trainData;
	// << "classes" << classes;
	//fs.release();
}