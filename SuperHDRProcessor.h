//#pragma once
#ifndef SUPERHDR_IPFUNCTION_H
#define SUPERHDR_IPFUNCTION_H

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

static int sd = 0;

using namespace cv;

class SuperHDRProcessor
{

public:
	static SuperHDRProcessor* GetInstance(void);
	void init(const std::string &filename);
	void run(void);
	Mat* GetOutput(void) { return &m_iOutput; };
	~SuperHDRProcessor();

private:
	SuperHDRProcessor();

	std::string filename;

	Size m_blurKernel;
	float m_blurSigmaX;
	int m_pyrOctaves;

	Mat m_visual;
	Mat m_visual32f;
	Mat m_hsv;
	Mat m_splitHSV[3];

	Mat *m_illumination;
	Mat *m_intensity;
	Mat m_intensity32f;
	Mat m_reflectance;
	Mat m_reflectance32f;
	Mat m_globalIntensity;
	Mat m_globalIntensity32f;
	Mat m_intensityClahe;
	Mat m_intensityClahe32f;
	Mat m_fullIntensity;

	Mat m_iFinal;
	Mat m_iOutput;

	Mat WNorm[3];

	void PrepareMaterials(void);

	void ExtractIllumination(void);
	void ApplyGuidedFilter(void);
	void ExtractReflectance(void);
	void ExtractGlobalIntensity(void);
	void AdaptiveHistogramEqualization(void);

	void CombineIntensities(void);
	void ApplyBrightnessWeight(const Mat &intensityChannel, Mat *weight);
	void ApplyChromaticContrastWeight(const Mat &intensityChannel, Mat *weight);
	void ApplyWeights(void);
	void ApplyPyramids(void);
	void CreateGaussPyramid(const cv::Mat& img, std::vector<cv::Mat>& pyramid, int levels);
	void CreateLaplacianPyramid(const cv::Mat& img, std::vector<cv::Mat>& pyramidLap, int levels);
	Mat PyramidReconstruct(std::vector<cv::Mat>& pyramid, int levels);
	void ComposeOutImage(void);

};

#endif //SUPERHDR_IPFUNCTION_H




