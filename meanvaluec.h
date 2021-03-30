#pragma once

#include <iostream>
#include <cstdio>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include<omp.h>


int dataFlag[2048][2048];
class MeanValueC {

public:
	std::vector<cv::Point> contour;
	std::vector<cv::Point> ps;
	Eigen::MatrixXf lambda;
	cv::Mat mask;
	std::vector<Eigen::MatrixXf> source;
	std::vector<Eigen::MatrixXf> target;
	

	MeanValueC(cv::Mat mask , std::vector<Eigen::MatrixXf>& source, std::vector<Eigen::MatrixXf>& target) {

		this->mask = mask.clone();
		this -> source = source;
		this->target = target;
		findContours1();
		getPs();
		lambda = Eigen::MatrixXf::Zero(ps.size(),contour.size());
	}
	MeanValueC() {
		
	}
	void writeMat(cv::Mat& mat, std::string name);
	void findContours1();
	void meanValueCoordinate();
	void getContourDiff();
	void getPs();
	void readContour(std::string path, std::vector<Eigen::MatrixXf> &contour);
};
void MeanValueC::findContours1() {

	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::Mat mask1, result;
	mask1 = mask.clone();
	threshold(mask1, result, 127, 255, CV_THRESH_BINARY);
	cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
	dilate(result,mask1, element);
	element = getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
	erode(mask1, mask1, element);
	cv::findContours(mask1, contours, hierarchy, cv::RETR_EXTERNAL,cv::CHAIN_APPROX_NONE);

	int max1 = 0;
	for (int i = 0; i < contours.size(); i++) {
		if (contours[i].size() > max1) {
			max1 = contours[i].size();
			contour = contours[i];
		}
	}
	memset(dataFlag, 0, sizeof(dataFlag));
#pragma omp parallel for
	for (int i = 0; i < contour.size(); i++) {
		dataFlag[uint(contour[i].x)][uint(contour[i].y)] = 1;
	}
	std::cout << "findCountour " << contour.size() << std::endl;
}

void MeanValueC::getPs() {
	int rows = mask.rows;
	int cols = mask.cols;
	time_t start = clock();
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			int flag = 0;
			/*for (int k = 0; k < contour.size(); k++) {
				if (i == contour[k].y && j == contour[k].x) {
					flag = 1;
					break;
				}
			}*/
			if (mask.at<uchar>(i, j) == 255 && dataFlag[j][i]==0) {
				cv::Point p(j,i);
				ps.push_back(p);
			}
		}
		//std::cout << i << std::endl;
	}
	time_t end = clock();
	std::cout << end - start << " ms" << std::endl;
	//std::cout << "ps" << std::endl;
}

void MeanValueC::meanValueCoordinate() {

	time_t start = clock();
	int core_num = omp_get_num_procs();
	std::cout<< "process " << core_num <<std::endl;
	double *dista = new double[contour.size()*ps.size()];
	double *tan1 = new double[contour.size()*ps.size()];
	double *w = new double[contour.size()*ps.size()];


	double *distc = new double[contour.size()];
	for (int j = 0; j < contour.size()-1; j++) {
		distc[j] = sqrt((contour[j].x - contour[j + 1].x)*(contour[j].x - contour[j + 1].x) + (contour[j].y - contour[j + 1].y)*(contour[j].y - contour[j + 1].y));
	}
	int size1 = contour.size() - 1;
	distc[size1] = sqrt((contour[size1].x - contour[0].x)*(contour[size1].x - contour[0].x) + (contour[size1].y - contour[0].y)*(contour[size1].y - contour[0].y));
	int len = contour.size();
#pragma omp parallel for num_threads(30)
	for (int i = 0; i < ps.size(); i++) {

		//std::cout << i << endl;
		for (int j = 0; j < contour.size(); j++) {
			dista[j+i*len] = sqrt( (contour[j].x - ps[i].x)*(contour[j].x - ps[i].x) + (contour[j].y - ps[i].y)*(contour[j].y - ps[i].y) );
		}
		for (int j = 0; j < contour.size()-1; j++) {
			
			tan1[j + i * len] = sqrt(0.5 - (dista[j + i * len]* dista[j + i * len] + dista[j + i * len +1]*dista[j + i * len +1] - distc[j]*distc[j]) / (4 * dista[j + i * len] * dista[j + i * len +1]))/ sqrt(0.5 + (dista[j + i * len] * dista[j + i * len] + dista[j + i * len + 1] * dista[j + i * len + 1] - distc[j] * distc[j]) / (4 * dista[j + i * len] * dista[j + i * len + 1]));
			if (isnan(tan1[j + i * len])) {
				tan1[j + i * len] = 0;
			}
		}
		size1 = contour.size() - 1;
		tan1[size1] = sqrt(0.5 - (dista[size1 + i * len] * dista[size1 + i * len] + dista[i * len] * dista[i * len] - distc[size1] * distc[size1]) / (4 * dista[size1 + i * len] * dista[i * len]))/ sqrt(0.5 + (dista[size1 + i * len] * dista[size1 + i * len] + dista[i * len] * dista[i * len] - distc[size1] * distc[size1]) / (4 * dista[size1 + i * len] * dista[i * len]));
		if (tan1[size1]) {
			tan1[size1] = 0;
		}
		double w_total = 0;
		for (int j = 0; j < contour.size()-1; j++) {
			w[j+i*len] = (tan1[j+i*len] + tan1[j + 1+i*len]) / dista[j+i*len];
			if (dista[j + i * len] == 0) {
				w[j + i * len] = 0;
			}
			w_total += w[j+i*len];
		}
		w[contour.size() - 1+i*len] = (tan1[contour.size() - 1+i*len] + tan1[i*len]) / dista[contour.size() - 1+i*len];
		w_total += w[contour.size() - 1+i*len];
		for (int j = 0; j < contour.size(); j++) {
			lambda(i,j) = w[j+i*len] / w_total;
		}
	}
	time_t end = clock();
	std::cout << "Waste Time " << end - start << " ms" << std::endl;
	//cout << "meanValueCoordinate" << endl;
}

void  MeanValueC::getContourDiff() {

	time_t start = clock();
	std::vector<float> diff1(contour.size());
	std::vector<float> diff2(contour.size());
	std::vector<float> diff3(contour.size());
#pragma omp parallel for
	for (int i = 0; i < contour.size(); i++) {
		int y = contour[i].y;
		int x = contour[i].x;
		diff1[i] = target[0](y,x) -  source[0](y,x) ;
		diff2[i] = target[1](y, x) - source[1](y, x);
		diff3[i] = target[2](y, x) - source[2](y, x);
	}
#pragma omp parallel for num_threads(50)
	for (int i = 0; i < ps.size(); i++) {
		int x = ps[i].x;
		int y = ps[i].y;
		double r1 = 0, r2 = 0, r3 = 0;
		for (int j = 0; j < contour.size(); j++) {
			r1 += diff1[j] * lambda(i, j);
			r2 += diff2[j] * lambda(i, j);
			r3 += diff3[j] * lambda(i, j);
		}
		target[0](y, x) = source[0](y, x) + r1;
		target[1](y,x)  = source[1](y, x) + r2;
		target[2](y,x)  = source[2](y, x) + r3;
	}
	time_t end = clock();
	std::cout << "Waste Time "<<end - start <<" ms"<< std::endl;
	//cv::imwrite("D:/res1.jpg",target);*/

}


/*
void MeanValueC::writeMat(cv::Mat& mat, std::string name) {
	
	int nrows = mat.rows;
	int ncols = mat.cols;
	double *data = new double[nrows*ncols];
	int cnt = 0;
	for (int i = 0; i < ncols; i++) {
		for (int j = 0; j < nrows; j++) {
			data[cnt] = double(mat.at<float>(j, i));
			cnt++;
		}
	}
	std::string pathname = "E:/faceReconres/" + name + ".mat";
	MATFile *pmatFile = matOpen(pathname.c_str(), "w");  
	mxArray *pWriteArray = mxCreateDoubleMatrix(nrows, ncols, mxREAL);
	memcpy((void *)(mxGetPr(pWriteArray)), (void *)data, sizeof(data)*ncols*nrows);
	matPutVariable(pmatFile, name.c_str(), pWriteArray);
	matClose(pmatFile);

}*/

void MeanValueC::readContour(std::string path, std::vector<Eigen::MatrixXf> &contour) {

	std::ifstream infile(path);
	int cnt = 0;
	Eigen::MatrixXf lms(2085, 2);
	while (!infile.eof()) {
		infile >> lms(cnt, 0) >> lms(cnt, 1);
		cnt++;
		if (cnt == 2085) {
			contour.push_back(lms);
			cnt = 0;
		}
	}
}
