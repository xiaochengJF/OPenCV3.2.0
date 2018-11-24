#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>

using namespace cv;

int main(int argc, char** argv){
	Mat  src, dst;
	src = imread("lena.jpg");
	if (!src.data){
		printf("could not load image.....\n");
		return -1;
	}
	namedWindow("yuantu",CV_WINDOW_AUTOSIZE);
	imshow ("yuantu", src);
	/*
	int rows = src.rows;
	int offsetx = src.channels();
	int cols = (src.cols-1) * src.channels();
	dst = Mat::zeros(src.size(),src.type());

	for (int row = 1; row < (rows - 1); row++){
		const uchar* previous = src.ptr<uchar>(row - 1);
		const uchar* current = src.ptr<uchar>(row);
		const uchar* next = src.ptr<uchar>(row + 1);
		uchar* output = dst.ptr<uchar>(row);
		for (int col = offsetx; col < cols; col++){
			output[col] = saturate_cast<uchar>(5 * current[col] - (current[col - offsetx] + current[col + offsetx] + previous[col] + next[col]));
		}
	}
	
	Mat kernel =  (Mat_<char>(3,3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
	filter2D(src, dst, src.depth(), kernel);
	namedWindow("duibi",CV_WINDOW_AUTOSIZE);
	imshow ("duibi", dst);
	dst = Mat(src.size(), src.type());//�����հ�ͼ��
	dst = Scalar(255, 255, 255);
	namedWindow("output", CV_WINDOW_AUTOSIZE);
	imshow("output", dst);
	//dst = src.clone();//��ȫ����
	*/
	cvtColor(src, dst, CV_BGR2GRAY);//ת����ɫ�ռ�
	printf("input image channels : %d\n", src.channels());
	printf("output image channels : %d\n", dst.channels());
	const uchar* firstRow = dst.ptr<uchar>(0);//ָ��ָ���һ�����ص�
	printf("value: %d", *firstRow );//��һ�����ص�Ҷ�ֵ
	imshow("output", dst);
	Mat M(100, 100, CV_8UC1, Scalar(255,0,255));//����һ��100X100��ͼ��
	namedWindow("output2", CV_WINDOW_AUTOSIZE);
	imshow("output2", M);


	waitKey();
	return 0;
}

	