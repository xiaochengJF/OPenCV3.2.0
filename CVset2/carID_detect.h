#pragma once
#include "opencv2/opencv.hpp"
#include <opencv2/ml.hpp>  
#include <opencv2/core.hpp>  
#include <opencv2/imgproc.hpp>  
#include "opencv2/imgcodecs.hpp" 
#include <time.h>
#include <stdlib.h>
#include <iostream>
using namespace std;
using namespace cv;
using namespace cv::ml;

void RgbConvToGray(const Mat& inputImage, Mat & outpuImage); //rgbתΪ�Ҷ�


void posDetect(Mat &, vector <RotatedRect> &,Mat &); //�ֲ�ѡȡ��ѡ��������
bool verifySizes(const RotatedRect &);  //����������Ҫ���������
void posDetect_closeImg(Mat &inputImage, vector <RotatedRect> & rects,Mat &);  //���ǵ����ƾ���ǳ�����ʱ������
bool verifySizes_closeImg(const RotatedRect & candidate); //�����ʱ�ĳ���������Ҫ���������     


void optimPosDetect(vector <RotatedRect> & rects_sImg, vector <RotatedRect> & rects_grayImg, //���������һ����λ
	vector <RotatedRect> & rects_closeImg, vector <RotatedRect> & rects_optimal);
float calOverlap(const Rect& box1, const Rect& box2);  //����2��������ص�����


void normalPosArea(Mat &intputImg, vector<RotatedRect> &rects_optimal, vector <Mat>& output_area);  //���Ʋü�����׼��Ϊ144*33


void svm_train();  //ȡ��SVM.xml�е���������ͱ�ǩ�������ѵ��


void char_segment(const Mat & inputImg, vector <Mat>&); //�Գ��������е��ַ����зָ�
bool char_verifySizes(const RotatedRect &);   //�ַ�������Ҫ���������
void char_sort(vector <RotatedRect > & in_char); //���ַ������������


void features(const Mat & in, Mat & out, int sizeData);  //���һ���ַ������Ӧ����������
Mat projectHistogram(const Mat& img, int t);             //����ˮƽ���ۼ�ֱ��ͼ��ȡ����tΪ0����1


void ann_train(ANN_MLP *ann, int numCharacters, int nlayers); //ȡ��ann_xml�е����ݣ�������������ѵ��
void classify(ANN_MLP* ann, vector<Mat> &char_feature, vector<int> & char_result); //ʹ��������ģ��Ԥ�⳵���ַ�������ӡ����Ļ