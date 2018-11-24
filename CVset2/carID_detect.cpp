#include "carID_detect.h"



void RgbConvToGray(const Mat& inputImage, Mat & outpuImage)  //g = 0.3R+0.59G+0.11B
{
	outpuImage = Mat(inputImage.rows, inputImage.cols, CV_8UC1);

	for (int i = 0; i<inputImage.rows; ++i)
	{
		uchar *ptrGray = outpuImage.ptr<uchar>(i);
		const Vec3b * ptrRgb = inputImage.ptr<Vec3b>(i);
		for (int j = 0; j<inputImage.cols; ++j)
		{
			ptrGray[j] = 0.3*ptrRgb[j][2] + 0.59*ptrRgb[j][1] + 0.11*ptrRgb[j][0];
		}
	}
}

void posDetect_closeImg(Mat &inputImage, vector <RotatedRect> & rects,Mat &drawImage)   //�����ҵ���ѡ���� rects
{

	Mat img_canny;
	Canny(inputImage, img_canny, 150, 220);
	Mat img_threshold;
	threshold(img_canny, img_threshold, 0, 255, CV_THRESH_OTSU + CV_THRESH_BINARY); //otsu�㷨�Զ������ֵ

	Mat element = getStructuringElement(MORPH_RECT, Size(15, 3));  //����̬ѧ�ĽṹԪ��
	morphologyEx(img_threshold, img_threshold, CV_MOP_CLOSE, element);  //��̬ѧ����

																		//Ѱ�ҳ������������
	vector< vector <Point> > contours;
	findContours(img_threshold, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);//ֻ���������
																				  //�Ժ�ѡ���������н�һ��ɸѡ
	vector< vector <Point> > ::iterator itc = contours.begin();

	while (itc != contours.end())
	{
		RotatedRect mr = minAreaRect(Mat(*itc)); //����ÿ����������С�н��������
		if (!verifySizes_closeImg(mr))  //�жϾ��������Ƿ����Ҫ��
		{
			itc = contours.erase(itc);
		}
		else
		{

			rects.push_back(mr);
			++itc;
		}
	}
	//���ƾ���
	RNG rng(12345);

	inputImage.copyTo(drawImage);
	Point2f pts[4];
	for (size_t t = 0; t < rects.size(); t++) {
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		rects[t].points(pts);
		for (int r = 0; r < 4; r++) {
			line(drawImage, pts[r], pts[(r + 1) % 4], color, 1, 8);
		}

	}

}


bool verifySizes_closeImg(const RotatedRect & candidate)
{
	float error = 0.4;
	const float aspect = 44 / 14; //�����
	int min = 100 * aspect * 100; //��С����
	int max = 180 * aspect * 180;  //�������
	float rmin = aspect - aspect*error; //�����������С�����
	float rmax = aspect + aspect*error; //�����������󳤿��

	int area = candidate.size.height * candidate.size.width;
	float r = (float)candidate.size.width / (float)candidate.size.height;
	if (r <1)
		r = 1 / r;

	if ((area < min || area > max) || (r< rmin || r > rmax))
		return false;
	else
		return true;
}

void posDetect(Mat &inputImage, vector <RotatedRect> & rects,Mat &drawImage)   //�����ҵ���ѡ���� rects
{
	Mat img_sobel;
	Sobel(inputImage, img_sobel, CV_8U, 1, 0, 3, 1, 0);

	Mat img_threshold;
	threshold(img_sobel, img_threshold, 0, 255, CV_THRESH_OTSU + CV_THRESH_BINARY); //otsu�㷨�Զ������ֵ

	Mat element = getStructuringElement(MORPH_RECT, Size(15, 3));  //����̬ѧ�ĽṹԪ��
	morphologyEx(img_threshold, img_threshold, CV_MOP_CLOSE, element);

	//Ѱ�ҳ������������
	vector< vector <Point> > contours;
	findContours(img_threshold, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);//ֻ���������
																				  //�Ժ�ѡ���������н�һ��ɸѡ
	vector< vector <Point> > ::iterator itc = contours.begin();

	while (itc != contours.end())
	{
		RotatedRect mr = minAreaRect(Mat(*itc)); //����ÿ����������С�н��������
		if (!verifySizes(mr))  //�жϾ��������Ƿ����Ҫ��
		{
			itc = contours.erase(itc);
		}
		else
		{
			rects.push_back(mr);
			++itc;
		}
	}
	//���ƾ���
	RNG rng(12345);
	
	inputImage.copyTo(drawImage);
	Point2f pts[4];
	for (size_t t = 0; t < rects.size(); t++) {
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		rects[t].points(pts);
		for (int r = 0; r < 4; r++) {
			line(drawImage, pts[r], pts[(r + 1) % 4], color, 1, 8);
		}

	}
}

bool verifySizes(const RotatedRect & candidate)
{
	float error = 0.4;
	const float aspect = 44 / 14; //�����
	int min = 20 * aspect * 20; //��С����
	int max = 180 * aspect * 180;  //�������
	float rmin = aspect - 2 * aspect*error; //�����������С�����
	float rmax = aspect + 2 * aspect*error; //�����������󳤿��

	int area = candidate.size.height * candidate.size.width;
	float r = (float)candidate.size.width / (float)candidate.size.height;
	if (r <1)
		r = 1 / r;

	if ((area < min || area > max) || (r< rmin || r > rmax)) //�������������Ϊ��candidateΪ��������
		return false;
	else
		return true;
}

void optimPosDetect(vector <RotatedRect> & rects_sImg, vector <RotatedRect> & rects_grayImg,
	vector <RotatedRect> & rects_closeImg, vector <RotatedRect> & rects_optimal)
{
	for (int i = 0; i<rects_sImg.size(); ++i)
	{
		for (int j = 0; j<rects_grayImg.size(); ++j)
		{
			if (calOverlap(rects_sImg[i].boundingRect(), rects_grayImg[j].boundingRect()) > 0.2)
			{
				if (rects_sImg[i].boundingRect().width * rects_sImg[i].boundingRect().height
					>= rects_grayImg[j].boundingRect().width * rects_grayImg[j].boundingRect().height)
					rects_optimal.push_back(rects_sImg[i]);
				else
					rects_optimal.push_back(rects_grayImg[j]);
			}
		}
	}

	if (rects_closeImg.size()<2)  //ֻ����1����Ϊ���ٶ�
	{
		for (int i = 0; i < rects_optimal.size(); ++i)
			for (int j = 0; j < rects_closeImg.size(); ++j)
			{
				if ((calOverlap(rects_optimal[i].boundingRect(), rects_closeImg[j].boundingRect()) < 0.2 &&
					calOverlap(rects_optimal[i].boundingRect(), rects_closeImg[j].boundingRect()) > 0.05))
				{
					rects_optimal.push_back(rects_closeImg[j]);
				}
			}
	}

}

float calOverlap(const Rect& box1, const Rect& box2)
{
	if (box1.x > box2.x + box2.width) { return 0.0; }
	if (box1.y > box2.y + box2.height) { return 0.0; }
	if (box1.x + box1.width < box2.x) { return 0.0; }
	if (box1.y + box1.height < box2.y) { return 0.0; }
	float colInt = min(box1.x + box1.width, box2.x + box2.width) - max(box1.x, box2.x);
	float rowInt = min(box1.y + box1.height, box2.y + box2.height) - max(box1.y, box2.y);
	float intersection = colInt * rowInt;
	float area1 = box1.width*box1.height;
	float area2 = box2.width*box2.height;
	return intersection / (area1 + area2 - intersection);
}

void normalPosArea(Mat &intputImg, vector<RotatedRect> &rects_optimal, vector <Mat>& output_area)
{
	float r, angle;
	for (int i = 0; i< rects_optimal.size(); ++i)
	{
		//��ת����
		angle = rects_optimal[i].angle;
		r = (float)rects_optimal[i].size.width / (float)(float)rects_optimal[i].size.height;
		if (r<1)
			angle = 90 + angle;
		Mat rotmat = getRotationMatrix2D(rects_optimal[i].center, angle, 1);//��ñ��ξ������
		Mat img_rotated;
		warpAffine(intputImg, img_rotated, rotmat, intputImg.size(), CV_INTER_CUBIC);

		//�ü�ͼ��
		Size rect_size = rects_optimal[i].size;
		if (r<1)
			swap(rect_size.width, rect_size.height);
		Mat  img_crop;
		getRectSubPix(img_rotated, rect_size, rects_optimal[i].center, img_crop);

		//�ù���ֱ��ͼ�������вü��õ���ͼ��ʹ������ͬ��Ⱥ͸߶ȣ�������ѵ���ͷ���
		Mat resultResized;
		resultResized.create(33, 144, CV_8UC3);
		resize(img_crop, resultResized, resultResized.size(), 0, 0, INTER_CUBIC);
		Mat grayResult;
		RgbConvToGray(resultResized, grayResult);
		//blur(grayResult ,grayResult,Size(3,3));
 		equalizeHist(grayResult, grayResult);
		output_area.push_back(grayResult);
	}
}

void svm_train()
{
	FileStorage fs;
	fs.open("SVM.xml", FileStorage::READ);
	Mat SVM_TrainningData;
	Mat SVM_Classes;
	

	fs["TrainingData"] >> SVM_TrainningData;
	fs["classes"] >> SVM_Classes;
	/*SVMParams SVM_params;
	SVM_params.kernel_type = SVM::LINEAR;

	svmClassifier.train(SVM_TrainningData, SVM_Classes, Mat(), Mat(), SVM_params); //SVMѵ��ģ��*/

	// ���������������ò���

	//SVM  svmClassifier;
	Ptr<SVM> svmClassifier = SVM::create();

	svmClassifier->setType(SVM::C_SVC);
	svmClassifier->setKernel(SVM::LINEAR);  //�˺���
	//svmClassifier->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
	//����ѵ������ 
	Ptr<TrainData> tData = TrainData::create(SVM_TrainningData, ROW_SAMPLE, SVM_Classes);

	// ѵ��������
	svmClassifier->train(tData);
	fs.release();
	svmClassifier->save("d:\\data.xml");
}

void char_segment(const Mat & inputImg, vector <Mat>& dst_mat)//�õ�20*20�ı�׼�ַ��ָ�ͼ��
{
	Mat img_threshold;
	threshold(inputImg, img_threshold, 180, 255, CV_THRESH_BINARY);
	Mat img_contours;
	img_threshold.copyTo(img_contours);

	vector < vector <Point> > contours;
	findContours(img_contours, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

	vector< vector <Point> > ::iterator itc = contours.begin();
	vector<RotatedRect> char_rects;

	while (itc != contours.end())
	{
		RotatedRect minArea = minAreaRect(Mat(*itc)); //����ÿ����������С�н��������

		Point2f vertices[4];
		minArea.points(vertices);

		if (!char_verifySizes(minArea))  //�жϾ��������Ƿ����Ҫ��
		{
			itc = contours.erase(itc);
		}
		else
		{
			++itc;
			char_rects.push_back(minArea);

		}
	}
	//���ƾ���
	RNG rng(12345);

	//inputImage.copyTo(drawImage);
	Point2f pts[4];
	for (size_t t = 0; t < char_rects.size(); t++) {
		//Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		char_rects[t].points(pts);
		for (int r = 0; r < 4; r++) {
			line(img_threshold, pts[r], pts[(r + 1) % 4], Scalar(255), 1, 8);
		}

	}
	namedWindow("Char image", CV_WINDOW_AUTOSIZE);
	imshow("Char image", img_threshold);
	char_sort(char_rects); //���ַ�����

	vector <Mat> char_mat;
	for (int i = 0; i<char_rects.size(); ++i)
	{
		char_mat.push_back(Mat(img_threshold, char_rects[i].boundingRect()));

	}

	Mat train_mat(2, 3, CV_32FC1);
	int length;
	dst_mat.resize(7);
	Point2f srcTri[3];
	Point2f dstTri[3];

	for (int i = 0; i< char_mat.size(); ++i)
	{
		srcTri[0] = Point2f(0, 0);
		srcTri[1] = Point2f(char_mat[i].cols - 1, 0);
		srcTri[2] = Point2f(0, char_mat[i].rows - 1);
		length = char_mat[i].rows > char_mat[i].cols ? char_mat[i].rows : char_mat[i].cols;
		dstTri[0] = Point2f(0.0, 0.0);
		dstTri[1] = Point2f(length, 0.0);
		dstTri[2] = Point2f(0.0, length);
		train_mat = getAffineTransform(srcTri, dstTri);
		dst_mat[i] = Mat::zeros(length, length, char_mat[i].type());
		warpAffine(char_mat[i], dst_mat[i], train_mat, dst_mat[i].size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(0));
		resize(dst_mat[i], dst_mat[i], Size(20, 20));  //�ߴ����Ϊ20*20

	}

}

bool char_verifySizes(const RotatedRect & candidate)
{
	float aspect = 33.0f / 20.0f;
	float charAspect = (float)candidate.size.width / (float)candidate.size.height; //��߱�
	float error = 0.35;
	float minHeight = 11;  //��С�߶�11
	float maxHeight = 33;  //���߶�33

	float minAspect = 0.20;  //���ǵ�����1����С�����Ϊ0.15
	float maxAspect = aspect + aspect*error;

	if (charAspect > minAspect && charAspect < maxAspect
		&& candidate.size.height >= minHeight && candidate.size.width< maxHeight) //��0���ص���������ȡ��߶�����������
		return true;
	else
		return false;
}

void char_sort(vector <RotatedRect > & in_char) //���ַ������������
{
	vector <RotatedRect >  out_char;
	const int length = 7;           //7���ַ�
	int index[length] = { 0,1,2,3,4,5,6 };
	float centerX[length];
	for (int i = 0; i < length; ++i)
	{
		centerX[i] = in_char[i].center.x;
	}

	for (int j = 0; j <length; j++) {
		for (int i = length - 2; i >= j; i--)
			if (centerX[i] > centerX[i + 1])
			{
				float t = centerX[i];
				centerX[i] = centerX[i + 1];
				centerX[i + 1] = t;

				int tt = index[i];
				index[i] = index[i + 1];
				index[i + 1] = tt;
			}
	}

	for (int i = 0; i<length; i++)
		out_char.push_back(in_char[(index[i])]);

	in_char.clear();     //���in_char
	in_char = out_char; //������õ��ַ������������¸�ֵ��in_char
}

void features(const Mat & in, Mat & out, int sizeData)
{
	Mat vhist = projectHistogram(in, 1); //ˮƽֱ��ͼ
	Mat hhist = projectHistogram(in, 0);  //��ֱֱ��ͼ

	Mat lowData;
	resize(in, lowData, Size(sizeData, sizeData));
	int numCols = vhist.cols + hhist.cols + lowData.cols * lowData.cols;
	out = Mat::zeros(1, numCols, CV_32F);

	int j = 0;
	for (int i = 0; i<vhist.cols; ++i)
	{
		out.at<float>(j) = vhist.at<float>(i);
		j++;
	}
	for (int i = 0; i < hhist.cols; ++i)
	{
		out.at<float>(j) = hhist.at<float>(i);
	}
	for (int x = 0; x<lowData.rows; ++x)
	{
		for (int y = 0; y < lowData.cols; ++y)
		{
			out.at<float>(j) = (float)lowData.at<unsigned char>(x, y);
			j++;
		}
	}

}

Mat projectHistogram(const Mat& img, int t)  //ˮƽ��ֱֱ��ͼ,0Ϊ����ͳ��
{                                            //1Ϊ����ͳ��
	int sz = (t) ? img.rows : img.cols;
	Mat mhist = Mat::zeros(1, sz, CV_32F);

	for (int j = 0; j < sz; j++)
	{
		Mat data = (t) ? img.row(j) : img.col(j);
		mhist.at<float>(j) = countNonZero(data);
	}

	double min, max;
	minMaxLoc(mhist, &min, &max);

	if (max > 0)
		mhist.convertTo(mhist, -1, 1.0f / max, 0);

	return mhist;
}

void ann_train(ANN_MLP *ann, int numCharacters, int nlayers)
{
	Mat trainData, classes;
	FileStorage fs;
	fs.open("ann_xml.xml", FileStorage::READ);

	fs["TrainingData"] >> trainData;
	fs["classes"] >> classes;


	
	Mat layerSizes(1, 3, CV_32SC1);
	layerSizes.at<int>(0) = trainData.cols;
	layerSizes.at<int>(1) = nlayers; //������Ԫ��������Ϊ3
	layerSizes.at<int>(2) = numCharacters; //��������Ϊ34
	//���ø������Ԫ����
	ann->setLayerSizes(layerSizes);
	//�����
	ann->setActivationFunction(ANN_MLP::SIGMOID_SYM);
	//MLP��ѵ������
	ann->setTrainMethod(ANN_MLP::BACKPROP, 0.1, 0.9);

	/*Mat layerSizes(1, 3, CV_32SC1);
	layerSizes.at<int>(0) = trainData.cols;
	layerSizes.at<int>(1) = nlayers; //������Ԫ��������Ϊ3
	layerSizes.at<int>(2) = numCharacters; //��������Ϊ34
	ann.create(layerSizes, ANN_MLP::SIGMOID_SYM, 1, 1);  //��ʼ��ann
	*/
	Mat trainClasses;
	trainClasses.create(trainData.rows, numCharacters, CV_32FC1);
	for (int i = 0; i< trainData.rows; i++)
	{
		for (int k = 0; k< trainClasses.cols; k++)
		{
			if (k == (int)classes.at<uchar>(i))
			{
				trainClasses.at<float>(i, k) = 1;
			}
			else
				trainClasses.at<float>(i, k) = 0;
		}
	}

	Mat weights(1, trainData.rows, CV_32FC1, Scalar::all(1));
	//ѵ��ģ��
	Ptr<TrainData> traindata = TrainData::create(trainData, ROW_SAMPLE, trainClasses);
	ann->train(traindata);
	//ann.train(trainData, trainClasses, weights);
}

void classify(ANN_MLP* ann, vector<Mat> &char_feature, vector<int> & char_result)
{
	char_result.resize(char_feature.size());
	for (int i = 0; i<char_feature.size(); ++i)
	{
		Mat output(1, 34, CV_32FC1); //1*34����
		ann->predict(char_feature[i], output);
		Point maxLoc;
		double maxVal;
		minMaxLoc(output, 0, &maxVal, 0, &maxLoc);
		char_result[i] = maxLoc.x;
	}

	std::cout << "�ó��ƺ�6λΪ��";
	char  s[] = { '0','1','2','3','4','5','6','7','8','9','A','B',
		'C','D','E','F','G','H','J','K','L','M','N','P','Q',
		'R','S','T','U','V','W','X','Y','Z' };
	for (int i = 1; i<char_result.size(); ++i)   //��һλ�Ǻ��֣�����ûʵ�ֶԺ��ֵ�Ԥ��
	{
		std::cout << s[char_result[i]];
	}
	std::cout << '\n';
}