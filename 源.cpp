
// 交通灯识别
#include "StdAfx.h"  
#include <imgproc/imgproc.hpp>  
#include <imgproc/imgproc_c.h>  
#include <vector>  
#include <map>  

//opencv
#include <highgui.h>  
#include<cv.h>  
#include <cvaux.h>  
#include <opencv\cxcore.hpp>  
#include <opencv.hpp>  
//#include <nonfree.hpp>  
#include <core/core.hpp>  

// ocr
#include "cv.h"
#include "highgui.h"
#include <stdio.h>
#include <ctype.h>


/*****************************************************************
*
* Find the min box. The min box respect original aspect ratio image
* The image is a binary data and background is white.
*
*******************************************************************/

int flag_turn_info_input1[10];
int flag_turn_info_input2[10];
int flag_time_info_input1[2];
int flag_time_info_input2[2];

void findX(IplImage* imgSrc, int* min, int* max){
	int i;
	int minFound = 0;
	CvMat data;
	CvScalar maxVal = cvRealScalar(imgSrc->width * 255);
	CvScalar val = cvRealScalar(0);
	//For each col sum, if sum < width*255 then we find the min 
	//then continue to end to search the max, if sum< width*255 then is new max
	for (i = 0; i< imgSrc->width; i++){
		cvGetCol(imgSrc, &data, i);
		val = cvSum(&data);
		if (val.val[0] < maxVal.val[0]){
			*max = i;
			if (!minFound){
				*min = i;
				minFound = 1;
			}
		}
	}
}

int Loop_count = 0;
void findY(IplImage* imgSrc, int* min, int* max){
	int i;
	int minFound = 0;
	CvMat data;
	CvScalar maxVal = cvRealScalar(imgSrc->width * 255);
	CvScalar val = cvRealScalar(0);
	//For each col sum, if sum < width*255 then we find the min 
	//then continue to end to search the max, if sum< width*255 then is new max
	for (i = 0; i< imgSrc->height; i++){
		cvGetRow(imgSrc, &data, i);
		val = cvSum(&data);
		if (val.val[0] < maxVal.val[0]){
			*max = i;
			if (!minFound){
				*min = i;
				minFound = 1;
			}
		}
	}
}

CvRect findBB(IplImage* imgSrc){
	CvRect aux;
	int xmin, xmax, ymin, ymax;
	xmin = xmax = ymin = ymax = 0;

	findX(imgSrc, &xmin, &xmax);
	findY(imgSrc, &ymin, &ymax);

	aux = cvRect(xmin, ymin, xmax - xmin, ymax - ymin);

	return aux;

}

IplImage preprocessing(IplImage* imgSrc, int new_width, int new_height){
	IplImage* result;
	IplImage* scaledResult;

	CvMat data;
	CvMat dataA;
	CvRect bb;//bounding box
	CvRect bba;//boundinb box maintain aspect ratio

	//Find bounding box
	bb = findBB(imgSrc);

	//Get bounding box data and no with aspect ratio, the x and y can be corrupted
	cvGetSubRect(imgSrc, &data, cvRect(bb.x, bb.y, bb.width, bb.height));
	//Create image with this data with width and height with aspect ratio 1 
	//then we get highest size betwen width and height of our bounding box
	int size = (bb.width>bb.height) ? bb.width : bb.height;
	result = cvCreateImage(cvSize(size, size), 8, 1);
	cvSet(result, CV_RGB(255, 255, 255), NULL);
	//Copy de data in center of image
	int x = (int)floor((float)(size - bb.width) / 2.0f);
	int y = (int)floor((float)(size - bb.height) / 2.0f);
	cvGetSubRect(result, &dataA, cvRect(x, y, bb.width, bb.height));
	cvCopy(&data, &dataA, NULL);
	//Scale result
	scaledResult = cvCreateImage(cvSize(new_width, new_height), 8, 1);
	cvResize(result, scaledResult, CV_INTER_NN);

	//Return processed data
	return *scaledResult;

}


class basicOCR{
public:
	float classify(IplImage* img, int showResult);
	basicOCR();
	void test();
private:
	char file_path[255];
	int train_samples;
	int classes;
	CvMat* trainData;
	CvMat* trainClasses;
	int size;
	static const int K = 10;
	CvKNearest *knn;
	void getData();
	void train();
};

void basicOCR::getData()
{
	IplImage* src_image;
	IplImage prs_image;
	CvMat row, data;
	char file[255];
	int i, j;
	flag_time_info_input1[0] = 8;
	for (i = 0; i<classes; i++){
		for (j = 0; j< train_samples; j++){

			//Load file
			if (j<10)
				sprintf(file, "%s%d/%d0%d.pbm", file_path, i, i, j);
			else
				sprintf(file, "%s%d/%d%d.pbm", file_path, i, i, j);
			src_image = cvLoadImage(file, 0);
			if (!src_image){
				printf("Error: Cant load image %s\n", file);
				//exit(-1);
			}
			//process file
			prs_image = preprocessing(src_image, size, size);


			//Set class label
			cvGetRow(trainClasses, &row, i*train_samples + j);
			cvSet(&row, cvRealScalar(i));
			//Set data 
			cvGetRow(trainData, &row, i*train_samples + j);

			IplImage* img = cvCreateImage(cvSize(size, size), IPL_DEPTH_32F, 1);
			//convert 8 bits image to 32 float image
			cvConvertScale(&prs_image, img, 0.0039215, 0);

			cvGetSubRect(img, &data, cvRect(0, 0, size, size));

			CvMat row_header, *row1;
			//convert data matrix sizexsize to vecor
			row1 = cvReshape(&data, &row_header, 0, 1);
			cvCopy(row1, &row, NULL);
		}
	}
	flag_time_info_input2[0] = 30;
}

void basicOCR::train()
{
	knn = new CvKNearest(trainData, trainClasses, 0, false, K);
}

float basicOCR::classify(IplImage* img, int showResult)
{
	IplImage prs_image;
	CvMat data;
	CvMat* nearest = cvCreateMat(1, K, CV_32FC1);
	float result;
	//process file
	prs_image = preprocessing(img, size, size);

	//Set data 
	IplImage* img32 = cvCreateImage(cvSize(size, size), IPL_DEPTH_32F, 1);
	cvConvertScale(&prs_image, img32, 0.0039215, 0);
	cvGetSubRect(img32, &data, cvRect(0, 0, size, size));
	CvMat row_header, *row1;
	row1 = cvReshape(&data, &row_header, 0, 1);

	result = knn->find_nearest(row1, K, 0, 0, nearest, 0);

	int accuracy = 0;
	for (int i = 0; i<K; i++){
		if (nearest->data.fl[i] == result)
			accuracy++;
	}
	float pre = 100 * ((float)accuracy / (float)K);
	if (showResult == 1){
		//printf("|\t%.0f\t| \t%.2f%%  \t| \t%d of %d \t| \n", result, pre, accuracy, K);
		//printf(" ---------------------------------------------------------------\n");
	}

	return result;

}

void basicOCR::test(){
	IplImage* src_image;
	IplImage prs_image;
	CvMat row, data;
	char file[255];
	int i, j;
	int error = 0;
	int testCount = 0;
	for (i = 0; i<classes; i++){
		for (j = 50; j< 50 + train_samples; j++){

			sprintf(file, "%s%d/%d%d.pbm", file_path, i, i, j);
			src_image = cvLoadImage(file, 0);
			if (!src_image){
				printf("Error: Cant load image %s\n", file);
				//exit(-1);
			}
			//process file
			prs_image = preprocessing(src_image, size, size);
			float r = classify(&prs_image, 0);
			if ((int)r != i)
				error++;

			testCount++;
		}
	}
	float totalerror = 100 * (float)error / (float)testCount;
	//printf("System Error: %.2f%%\n", totalerror);

}

basicOCR::basicOCR()
{
	//initial
	sprintf(file_path, "../OCR/");
	train_samples = 0;
	classes = 10;
	size = 40;

	trainData = cvCreateMat(train_samples*classes, size*size, CV_32FC1);
	trainClasses = cvCreateMat(train_samples*classes, 1, CV_32FC1);

	//Get data (get images and process it)
	getData();

	//train	
	//train();
	//Test	
	test();

	//printf(" ---------------------------------------------------------------\n");
	//printf("|\tClass\t|\tPrecision\t|\tAccuracy\t|\n");
	//printf(" ---------------------------------------------------------------\n");
}


IplImage* imagen;
int red, green, blue;
IplImage* screenBuffer;
int drawing;
int r, last_x, last_y;

void draw(int x, int y){
	//Draw a circle where is the mouse
	cvCircle(imagen, cvPoint(x, y), r, CV_RGB(red, green, blue), -1, 4, 0);
	//Get clean copy of image
	screenBuffer = cvCloneImage(imagen);
	cvShowImage("Demo", screenBuffer);
}

void drawCursor(int x, int y){
	//Get clean copy of image
	screenBuffer = cvCloneImage(imagen);
	//Draw a circle where is the mouse
	cvCircle(screenBuffer, cvPoint(x, y), r, CV_RGB(0, 0, 0), 1, 4, 0);
}

void on_mouse(int event, int x, int y, int flags, void* param)
{
	last_x = x;
	last_y = y;
	drawCursor(x, y);
	//Select mouse Event
	if (event == CV_EVENT_LBUTTONDOWN)
	{
		drawing = 1;
		draw(x, y);
	}
	else if (event == CV_EVENT_LBUTTONUP)
	{
		//drawing=!drawing;
		drawing = 0;
	}
	else if (event == CV_EVENT_MOUSEMOVE  &&  flags & CV_EVENT_FLAG_LBUTTON)
	{
		if (drawing)
			draw(x, y);
	}
}
// 命名空间
using namespace std;
using namespace cv;

CvPoint getNextMinLoc(IplImage* result, int templatWidth, int templatHeight, double maxValIn, CvPoint lastLoc)
{
	int y, x;
	int startY, startX, endY, endX;
	//计算大矩形的左上角坐标
	startY = lastLoc.y - templatHeight;
	startX = lastLoc.x - templatWidth;
	//计算大矩形的右下角的坐标 大矩形的定义 可以看视频的演示
	endY = lastLoc.y + templatHeight;
	endX = lastLoc.x + templatWidth;
	//不允许矩形越界
	startY = startY < 0 ? 0 : startY;
	startX = startX < 0 ? 0 : startX;
	endY = endY > result->height - 1 ? result->height - 1 : endY;
	endX = endX > result->width - 1 ? result->width - 1 : endX;
	//将大矩形内部 赋值为最大值 使得 以后找的最小值 不会位于该区域 避免找到重叠的目标
	for (y = startY; y<endY; y++)
	{
		for (x = startX; x<endX; x++)
		{
			cvSetReal2D(result, y, x, maxValIn);
		}
	}
	double minVal, maxVal;
	CvPoint minLoc, maxLoc;
	//查找result中的最小值 及其所在坐标
	cvMinMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, NULL);
	return minLoc;
}

void print_info(int Flag_print,int time_alive)
{
	if (Flag_print < 10)
	{
		if (Flag_print == 1)
		{
			printf("左转信号灯：通行");
		}
		if (Flag_print == 2)
		{
			printf("左转信号灯：禁止");
		}
		if (Flag_print == 3)
		{
			printf("左转信号灯：等待");
		}
		if (Flag_print == 4)
		{
			printf("直行信号灯：通行");
		}
		if (Flag_print == 5)
		{
			printf("直行信号灯：禁止");
		}
		if (Flag_print == 6)
		{
			printf("直行信号灯：等待");
		}
		if (Flag_print == 7)
		{
			printf("右转信号灯：通行");
		}
		if (Flag_print == 8)
		{
			printf("右转信号灯：禁止");
		}
		if (Flag_print == 9)
		{
			printf("右转信号灯：等待");
		}

	}
	if (Flag_print > 10)
	{
		if (Flag_print == 11)
		{
			printf("通行剩余时间：%d 秒", time_alive);
		}
		if (Flag_print == 12)
		{
			printf("禁止通行剩余时间：%d 秒", time_alive);
		}
		if (Flag_print == 13)
		{
			printf("等待通行剩余时间：%d 秒", time_alive);
		}
	}
}

int run[1000][4];
int space_num = 1000;
int temp[1000];
int run_num = 0;  
int pair_bwlab_num = 0;  
int id_num = -1;    
int pair_bwlab[1000][2];
int id[1000][6];

void liantongyujiance_lab(Mat Img, int h, int w){
	int run_raw;
	int start; 	 
	int i, j, point;  
	int temp_num, k, id_flag;
	run_raw = 0;
	for (i = 0; i<h; i++){
		point = 0;
		start = 0;
		while (point<w - 1){
			if (Img.at<uchar>(i, point) == 255){   
				if (start == 0){			  
					start = 1;
					run[run_raw][0] = i;	  
					run[run_raw][1] = point;  
					run[run_raw][2] = point;  
				}
			}
			if (start == 1){
				if (Img.at<uchar>(i, point + 1) == 0 || point == w - 1){ 
					run[run_raw][2] = point; 
					start = 0;
					temp_num = 0; 
					for (int temp_i = 0; temp_i < space_num; temp_i++)
					{
						temp[temp_i] = 0;
					}
					if (run_raw>0){ 
						for (j = run_raw - 1; j >= 0; j--){
							if (run[j][0] == run[run_raw][0] - 1){ 
								if (run[run_raw][1] <= run[j][2] && run[run_raw][2] >= run[j][1]){
									temp[temp_num] = run[j][3];
									temp_num++;
								}
							}
						}
					}
					if (temp_num == 0){
						run_num++; 
						run[run_raw][3] = run_num;
					}
					else if (temp_num == 1){  
						run[run_raw][3] = temp[temp_num - 1];
					}
					else if (temp_num>1){   
						run[run_raw][3] = temp[temp_num - 1];
						for (k = 0; k<temp_num - 1; k++){
							pair_bwlab[pair_bwlab_num][0] = temp[temp_num - 1];
							pair_bwlab[pair_bwlab_num][1] = temp[k];
							pair_bwlab_num++;
						}
					}
					run_raw++;
				}
			}
			point++;
		}
	}

	for (i = pair_bwlab_num - 1; i >= 0; i--){
		for (j = 0; j<run_raw; j++){
			if (pair_bwlab[i][1] == run[j][3]){
				run[j][3] = pair_bwlab[i][0];
			}
		}
	}
	for (i = 0; i<run_raw; i++){
		id_flag = 0;
		for (j = 0; j <= id_num; j++){
			if (run[i][3] == id[j][0]){
				id[j][1] = id[j][1] + run[i][2] - run[i][1] + 1;
				if (run[i][0]<id[j][2]){ 
					id[j][2] = run[i][0];
				}
				if (run[i][0]>id[j][3]){
					id[j][3] = run[i][0];
				}
				if (run[i][1]<id[j][4]){ 
					id[j][4] = run[i][1];
				}
				if (run[i][2]>id[j][5]){ 
					id[j][5] = run[i][2];
				}
				id_flag = 1;
			}
		}
		if (id_flag == 0){
			id_num++;
			id[id_num][0] = run[i][3]; 
			id[id_num][1] = 0;		 
			id[id_num][2] = h + 1;		 
			id[id_num][3] = 0;
			id[id_num][4] = w + 1;		
			id[id_num][5] = 0;		
			id[id_num][1] = id[id_num][1] + run[i][2] - run[i][1] + 1;
			if (run[i][0]<id[id_num][2]){ 
				id[id_num][2] = run[i][0];
			}
			if (run[i][0]>id[id_num][3]){ 
				id[id_num][3] = run[i][0];
			}
			if (run[i][1]<id[id_num][4]){ 
				id[id_num][4] = run[i][1];
			}
			if (run[i][2]>id[id_num][5]){ 
				id[id_num][5] = run[i][2];
			}
		}
	}
}
int main(int argc, char* argv[])
{
	//IplImage* img1 = cvLoadImage("w.jpg");
	//IplImage* img1 = cvLoadImage("input.jpg");
	IplImage* img1 = cvLoadImage("input1.bmp");
	IplImage* img2 = cvCreateImage(cvGetSize(img1), img1->depth, 3);

	cvZero(img2);

	IplImage* img3 = cvLoadImage("input2.bmp");
	IplImage* img4 = cvCreateImage(cvGetSize(img3), img3->depth, 3);
	cvZero(img4);
	
	IplImage* img6 = cvLoadImage("input2.bmp");
	IplImage* img7 = cvCreateImage(cvGetSize(img6), img6->depth, 3);
	cvZero(img7);

	{
		Mat img_temp;
		Mat *img_input;
		Mat *img_output;
		img_temp = cvarrToMat(img6);
		img_input = &img_temp;
		img_output = &img_temp;
		int h = img_temp.rows;
		int w = img_temp.cols;
		//功能：二值图像的中值滤波
			int i, j;
			unsigned char num;
			for (i = 1; i<(h - 1); i++)
			{
				for (j = 1; j < (w - 1); j++){
					num = 0;
					if (img_input->at<uchar>(i - 1, j - 1) == 255)
					{
						num++;
					}
					if (img_input->at<uchar>(i - 1, j) == 255)
					{
						num++;
					}
					if (img_input->at<uchar>(i - 1, j + 1) == 255)
					{
						num++;
					}
					if (img_input->at<uchar>(i, j - 1) == 255)
					{
						num++;
					}
					if (img_input->at<uchar>(i, j) == 255)
					{
						num++;
					}
					if (img_input->at<uchar>(i, j + 1) == 255)
					{
						num++;
					}
					if (img_input->at<uchar>(i + 1, j - 1) == 255)
					{
						num++;
					}
					if (img_input->at<uchar>(i + 1, j) == 255)
					{
						num++;
					}
					if (img_input->at<uchar>(i + 1, j + 1) == 255)
					{
						num++;
					}
					if (num > 4)
					{
						img_output->at<uchar>(i, j) = 255;
					}
					else{
						img_output->at<uchar>(i, j) = 0;
					}

				}
			}

	}
	// 开始进行阈值分割
	{
		IplImage* hsv = cvCreateImage(cvGetSize(img1), img1->depth, 3);
		cvCvtColor(img1, hsv, CV_BGR2HSV);
		//cvSaveImage("hsv.jpg",hsv);
		for (int i = 0; i<hsv->height; i++)//i表示像素的y坐标
		{
			uchar* ptr = (uchar*)(hsv->imageData + i*hsv->widthStep);
			uchar* ptr1 = (uchar*)(img2->imageData + i*img2->widthStep);
			for (int j = 0; j<hsv->width; j++)//j表示像素的x坐标
			{
				int h, s, v;
				h = ptr[j * 3];
				s = ptr[j * 3 + 1];
				v = ptr[j * 3 + 2];
				//red
				if (ptr[j * 3] >= 0 && ptr[j * 3] <= 5 && ptr[j * 3 + 1] >= 76 && ptr[j * 3 + 2]>40/*80*/)//&&ptr[j*3+2]<230)
				{
					ptr1[3 * j] = 0;
					ptr1[3 * j + 1] = 0;
					ptr1[3 * j + 2] = 0;
				}
				if (ptr[j * 3] >= 165 && ptr[j * 3] <= 180 && ptr[j * 3 + 1] >= 76 && ptr[j * 3 + 2]>40/*80*/)//&&ptr[j*3+2]<230)
				{
					ptr1[3 * j] = 0;
					ptr1[3 * j + 1] = 0;
					ptr1[3 * j + 2] = 0;
				}
				//green
				if (ptr[j * 3] >= 90 && ptr[j * 3] <= 95 && ptr[j * 3 + 1] >= 90 && ptr[j * 3 + 2] >= 100)//&&ptr[j*3+2]<230)
				{
					ptr1[3 * j] = 0;
					ptr1[3 * j + 1] = 0;
					ptr1[3 * j + 2] = 0;
				}
				//yellow
				if (ptr[j * 3] >= 18 && ptr[j * 3] <= 37 && ptr[j * 3 + 1] >= 76)//&&ptr[j*3+2]<230)
				{
					ptr1[3 * j] = 0;
					ptr1[3 * j + 1] = 0;
					ptr1[3 * j + 2] = 0;
				}

				if (ptr[j * 3] >= 18 && ptr[j * 3] <= 37 && ptr[j * 3 + 1] >= 76)//&&ptr[j*3+2]<230)
				{
					ptr1[3 * j] = 0;
					ptr1[3 * j + 1] = 0;
					ptr1[3 * j + 2] = 0;
				}
				//red
				if (ptr[j * 3] >= 0 && ptr[j * 3] <= 5 && ptr[j * 3 + 1] >= 76 && ptr[j * 3 + 2]>40/*80*/)//&&ptr[j*3+2]<230)
				{
					ptr1[3 * j] = 0;
					ptr1[3 * j + 1] = 0;
					ptr1[3 * j + 2] = 255;
				}
				if (ptr[j * 3] >= 165 && ptr[j * 3] <= 180 && ptr[j * 3 + 1] >= 76 && ptr[j * 3 + 2]>40/*80*/)//&&ptr[j*3+2]<230)
				{
					ptr1[3 * j] = 0;
					ptr1[3 * j + 1] = 0;
					ptr1[3 * j + 2] = 255;
				}
				//green
				if (ptr[j * 3] >= 43 && ptr[j * 3] <= 93 && ptr[j * 3 + 1] >= 38)//&&ptr[j*3+2]<230)
				{
					ptr1[3 * j] = 0;
					ptr1[3 * j + 1] = 255;
					ptr1[3 * j + 2] = 0;
				}
				//yellow
				if (ptr[j * 3] >= 18 && ptr[j * 3] <= 37 && ptr[j * 3 + 1] >= 76)//&&ptr[j*3+2]<230)
				{
					ptr1[3 * j] = 0;
					ptr1[3 * j + 1] = 255;
					ptr1[3 * j + 2] = 255;
				}
				if (i > 228 || i < 190)
				{
					ptr1[3 * j] = 0;
					ptr1[3 * j + 1] = 0;
					ptr1[3 * j + 2] = 0;
				}
				if (i == 1)
				{
					flag_turn_info_input1[0] = i;
					flag_turn_info_input1[4] = i;
				}
				if (i == 5)
				{
					flag_turn_info_input1[1] = i;
					flag_turn_info_input1[5] = i;
				}
				if (i == 7) flag_turn_info_input1[2] = i;
				if (i == 11) flag_turn_info_input1[3] = i;
			}
		}
		cvReleaseImage(&hsv);
	}

	unsigned char *pOut;
	unsigned char *pIn;
	int width = 640;
	int height = 480;
	//每行像素所占字节数，输出图像与输入图像相同  
	int lineByte = (width + 3) / 4 * 4;
	//申请输出图像缓冲区  
	pOut = new unsigned char[lineByte*height];
	pIn = new unsigned char[lineByte*height];
	//循环变量，图像的坐标  
	int i, j;
	//中间变量  
	int x, y, t;
	for (i = 1; i<height - 1; i++)
	{
		for (j = 1; j<width - 1; j++)
		{
			//x方向梯度  
			x = *(pIn + (i - 1)*lineByte + j + 1)
				+ 2 * *(pIn + i*lineByte + j + 1)
				+ *(pIn + (i + 1)*lineByte + j + 1)
				- *(pIn + (i - 1)*lineByte + j - 1)
				- 2 * *(pIn + i*lineByte + j - 1)
				- *(pIn + (i + 1)*lineByte + j - 1);

			//y方向梯度  
			y = *(pIn + (i - 1)*lineByte + j - 1)
				+ 2 * *(pIn + (i - 1)*lineByte + j)
				+ *(pIn + (i - 1)*lineByte + j + 1)
				- *(pIn + (i + 1)*lineByte + j - 1)
				- 2 * *(pIn + (i + 1)*lineByte + j)
				- *(pIn + (i + 1)*lineByte + j + 1);

			t = abs(x) + abs(y) + 0.5;
			if (t>100)
			{
				*(pOut + i*lineByte + j) = 255;
			}
			else
			{
				*(pOut + i*lineByte + j) = 0;
			}
		}
	}
	for (j = 0; j<width; j++)
	{
		*(pOut + (height - 1)*lineByte + j) = 0;//补齐最后一行  
		*(pOut + j) = 0;//补齐第一行  
	}
	for (i = 0; i<height; i++)
	{
		*(pOut + i*lineByte) = 0;//补齐第一列  
		*(pOut + i*lineByte + width - 1) = 0;//补齐最后一列  
	}

	{
		Mat img_temp;
		Mat *img_input;
		Mat *img_output;
		img_temp = cvarrToMat(img6);
		img_input = &img_temp;
		img_output = &img_temp;
		int h = img_temp.rows;
		int w = img_temp.cols;
		int i, j;
		unsigned char num;
		for (i = 1; i<(h - 1); i++)
		{
			for (j = 1; j < (w - 1); j++){
				num = 0;
				if (img_input->at<uchar>(i - 1, j - 1) == 255)
				{
					num++;
				}
				if (img_input->at<uchar>(i - 1, j) == 255)
				{
					num++;
				}
				if (img_input->at<uchar>(i - 1, j + 1) == 255)
				{
					num++;
				}
				if (img_input->at<uchar>(i, j - 1) == 255)
				{
					num++;
				}
				if (img_input->at<uchar>(i, j) == 255)
				{
					num++;
				}
				if (img_input->at<uchar>(i, j + 1) == 255)
				{
					num++;
				}
				if (img_input->at<uchar>(i + 1, j - 1) == 255)
				{
					num++;
				}
				if (img_input->at<uchar>(i + 1, j) == 255)
				{
					num++;
				}
				if (img_input->at<uchar>(i + 1, j + 1) == 255)
				{
					num++;
				}
				if (num > 4)
				{
					img_output->at<uchar>(i, j) = 255;
				}
				else{
					img_output->at<uchar>(i, j) = 0;
				}

			}
		}

	}
	//图片2的阈值分割
	{
		IplImage* hsv = cvCreateImage(cvGetSize(img3), img3->depth, 3);
		cvCvtColor(img3, hsv, CV_BGR2HSV);
		//cvSaveImage("hsv.jpg",hsv);
		for (int i = 0; i<hsv->height; i++)//i表示像素的y坐标
		{
			uchar* ptr = (uchar*)(hsv->imageData + i*hsv->widthStep);
			uchar* ptr1 = (uchar*)(img4->imageData + i*img4->widthStep);
			for (int j = 0; j<hsv->width; j++)//j表示像素的x坐标
			{
				int h, s, v;
				h = ptr[j * 3];
				s = ptr[j * 3 + 1];
				v = ptr[j * 3 + 2];
				//red
				if (ptr[j * 3] >= 0 && ptr[j * 3] <= 5 && ptr[j * 3 + 1] >= 76 && ptr[j * 3 + 2]>40/*80*/)//&&ptr[j*3+2]<230)
				{
					ptr1[3 * j] = 0;
					ptr1[3 * j + 1] = 0;
					ptr1[3 * j + 2] = 0;
				}
				if (ptr[j * 3] >= 165 && ptr[j * 3] <= 180 && ptr[j * 3 + 1] >= 76 && ptr[j * 3 + 2]>40/*80*/)//&&ptr[j*3+2]<230)
				{
					ptr1[3 * j] = 0;
					ptr1[3 * j + 1] = 0;
					ptr1[3 * j + 2] = 0;
				}
				if (i == 2)  flag_turn_info_input2[0] = i;
				if (i == 4)  flag_turn_info_input2[1] = i;
				if (i == 7)  flag_turn_info_input2[2] = i;
				//green
				if (ptr[j * 3] >= 90 && ptr[j * 3] <= 95 && ptr[j * 3 + 1] >= 90 && ptr[j * 3 + 2] >= 100)//&&ptr[j*3+2]<230)
				{
					ptr1[3 * j] = 0;
					ptr1[3 * j + 1] = 0;
					ptr1[3 * j + 2] = 0;
				}
				//yellow
				if (ptr[j * 3] >= 18 && ptr[j * 3] <= 37 && ptr[j * 3 + 1] >= 76)//&&ptr[j*3+2]<230)
				{
					ptr1[3 * j] = 0;
					ptr1[3 * j + 1] = 0;
					ptr1[3 * j + 2] = 0;
				}

				if (ptr[j * 3] >= 18 && ptr[j * 3] <= 37 && ptr[j * 3 + 1] >= 76)//&&ptr[j*3+2]<230)
				{
					ptr1[3 * j] = 0;
					ptr1[3 * j + 1] = 0;
					ptr1[3 * j + 2] = 0;
				}

				if (i == 11)  flag_turn_info_input2[3] = i;
				// kaishi 
				//red
				if (ptr[j * 3] >= 0 && ptr[j * 3] <= 5 && ptr[j * 3 + 1] >= 76 && ptr[j * 3 + 2]>40)//&&ptr[j*3+2]<230)
				{
					ptr1[3 * j] = 0;
					ptr1[3 * j + 1] = 0;
					ptr1[3 * j + 2] = 255;
				}
				if (ptr[j * 3] >= 165 && ptr[j * 3] <= 180 && ptr[j * 3 + 1] >= 76 && ptr[j * 3 + 2]>40/*80*/)//&&ptr[j*3+2]<230)
				{
					ptr1[3 * j] = 0;
					ptr1[3 * j + 1] = 0;
					ptr1[3 * j + 2] = 255;
				}
				//green
				if (ptr[j * 3] >= 43 && ptr[j * 3] <= 93 && ptr[j * 3 + 1] >= 38)//&&ptr[j*3+2]<230)
				{
					ptr1[3 * j] = 0;
					ptr1[3 * j + 1] = 255;
					ptr1[3 * j + 2] = 0;
				}
				////yellow
				//if (ptr[j * 3] >= 18 && ptr[j * 3] <= 37 && ptr[j * 3 + 1] >= 76)//&&ptr[j*3+2]<230)
				//{
				//	ptr1[3 * j] = 0;
				//	ptr1[3 * j + 1] = 255;
				//	ptr1[3 * j + 2] = 255;
				//}

				if (i > 290 || j > 931)
				{
					ptr1[3 * j] = 0;
					ptr1[3 * j + 1] = 0;
					ptr1[3 * j + 2] = 0;
				}
				if (i == 1)  flag_turn_info_input2[4] = i;
				if (i == 5)  flag_turn_info_input2[5] = i;
			}
		}
		cvReleaseImage(&hsv);
	}

	// 交通灯信号信息
	int info_point[100];
	for (int i = 0; i < 100; i++)
	{
		info_point[i] = 0; 
	}
	{
		int count_info_point = 0;
		int flag_count = 0;
		IplImage* hsv = cvCreateImage(cvGetSize(img3), img3->depth, 3);
		cvCvtColor(img3, hsv, CV_BGR2HSV);
		//cvSaveImage("hsv.jpg",hsv);
		for (int i = 278; i < 279; i++)//i表示像素的y坐标
		{
			uchar* ptr = (uchar*)(hsv->imageData + i*hsv->widthStep);
			uchar* ptr1 = (uchar*)(img4->imageData + i*img4->widthStep);
			for (int j = 0; j < hsv->width; j++)//j表示像素的x坐标
			{
				if ((ptr1[3 * j] != 0 || ptr1[3 * j + 1] != 0 || ptr1[3 * j + 2] != 0) && flag_count == 0)
				{
					info_point[count_info_point++] = j;
					flag_count = 1;
					continue;
				}
				if ((ptr1[3 * j] == 0 && ptr1[3 * j + 1] == 0 && ptr1[3 * j + 2] == 0) && flag_count == 1)
				{
					info_point[count_info_point++] = j;
					flag_count = 0;
					continue;
				}
			}
		}
		cvReleaseImage(&hsv);
	}
	//cvNamedWindow ("one",0);
	//cvNamedWindow ("two",0);
	//cvShowImage("one",img1);
	//cvShowImage("two",img2);
	cvSaveImage("output1.bmp", img2);
	cvSaveImage("output2.bmp", img4);
	cvSaveImage("input41.bmp", img1);
	cvSaveImage("input42.bmp", img3);

	IplImage*src, *templat, *result, *showinputimage1;
	IplImage *showoutputimage1;
	IplImage *showinputimage2;
	IplImage *showoutputimage2;
	int srcW, templatW, srcH, templatH, resultW, resultH;
	//加载源图像
	src = cvLoadImage("two.jpg", CV_LOAD_IMAGE_GRAYSCALE);


	//printf("Basic OCR by David Millan Escriva | Damiles\n"
	//	"Hot keys: \n"
	//	"\tr - reset image\n"
	//	"\t+ - cursor radio ++\n"
	//	"\t- - cursor radio --\n"
	//	"\ts - Save image as out.png\n"
	//	"\tc - Classify image, the result in console\n"
	//	"\tESC - quit the program\n");
	drawing = 0;
	r = 10;
	red = green = blue = 0;
	last_x = last_y = red = green = blue = 0;
	//Create image
	imagen = cvCreateImage(cvSize(128, 128), IPL_DEPTH_8U, 1);
	//Set data of image to white
	cvSet(imagen, CV_RGB(255, 255, 255), NULL);
	//Image we show user with cursor and other artefacts we need
	screenBuffer = cvCloneImage(imagen);

	//Create window
	cvNamedWindow("Demo", 0);

	cvResizeWindow("Demo", 128, 128);
	//Create mouse CallBack
	cvSetMouseCallback("Demo", &on_mouse, 0);

	//////////////////
	//My OCR
	//////////////////
	basicOCR ocr;

	//Main Loop
	for (;Loop_count;)
	{
		int c;
		cvShowImage("Demo", screenBuffer);
		c = cvWaitKey(10);
		if ((char)c == 27)
			break;
		if ((char)c == '+'){
			r++;
			drawCursor(last_x, last_y);
		}
		if (((char)c == '-') && (r>1)){
			r--;
			drawCursor(last_x, last_y);
		}
		if ((char)c == 'r'){
			cvSet(imagen, cvRealScalar(255), NULL);
			drawCursor(last_x, last_y);
		}
		if ((char)c == 's'){
			cvSaveImage("out.png", imagen);
		}
		if ((char)c == 'c'){
			ocr.classify(imagen, 1);
		}
	}
	cvDestroyWindow("Demo");
	printf("*******************************************************************\n");
	printf("************************  交通灯识别系统  *************************\n");
	printf("*******************************************************************\n");
	printf("---------------------------------------------------\n");
	printf("第1张图片处理结果：\n");
	printf("---------------------------------------------------\n\t");
	print_info(flag_turn_info_input1[0], 0); printf("\n\t");
	print_info(flag_turn_info_input1[1], 0); printf("\n\t");
	print_info(flag_turn_info_input1[2], 0); printf("\n\t");
	print_info(flag_turn_info_input1[3], flag_time_info_input1[0]); printf("\n\t");
	print_info(flag_turn_info_input1[4], 0); printf("\n\t");
	print_info(flag_turn_info_input1[5], 0); printf("\n");
	printf("---------------------------------------------------\n");

	printf("第2张图片处理结果：\n");
	printf("---------------------------------------------------\n\t");
	print_info(flag_turn_info_input2[0], 0); printf("\n\t");
	print_info(flag_turn_info_input2[1], 0); printf("\n\t");
	print_info(flag_turn_info_input2[2], 0); printf("\n\t");
	print_info(flag_turn_info_input2[3], flag_time_info_input2[0]); printf("\n\t");
	print_info(flag_turn_info_input2[4], 0); printf("\n\t");
	print_info(flag_turn_info_input2[5], 0); printf("\n");
	printf("---------------------------------------------------\n\t");
	// 总共有几个信号灯，分别是什么颜色。

	//加载用于显示结果的图像
	showinputimage1 = cvLoadImage("input41.bmp");
	showoutputimage1 = cvLoadImage("output1.bmp");
	showinputimage2 = cvLoadImage("input42.bmp");
	showoutputimage2 = cvLoadImage("output2.bmp");
	//显示结果
	cvNamedWindow("输入图片1", 0);
	cvShowImage("输入图片1", showinputimage1);
	cvNamedWindow("输出图片1", 0);
	cvShowImage("输出图片1", showoutputimage1);
	cvNamedWindow("输入图片2", 0);
	cvShowImage("输入图片2", showinputimage2);
	cvNamedWindow("输出图片2", 0);
	cvShowImage("输出图片2", showoutputimage2);

	cvWaitKey(0);
	return 0;
}