#include "SuperHDRProcessor.h"
#include "guidedfilter.h"
#include "math_functions.h"


SuperHDRProcessor::SuperHDRProcessor() {
	m_blurKernel = Size(5, 5);
	m_blurSigmaX = 0.67;
	m_pyrOctaves = 6;
};

SuperHDRProcessor::~SuperHDRProcessor() {

	delete m_illumination;
	delete m_intensity;

};

void SuperHDRProcessor::init(const std::string &f) {

	std::string filename = f;
	m_visual = imread(filename);

}

void SuperHDRProcessor::run(void) {

	PrepareMaterials();
	ExtractIllumination();
	ApplyGuidedFilter();
	ExtractReflectance();
	ExtractGlobalIntensity();
	AdaptiveHistogramEqualization();
	CombineIntensities();
	ApplyWeights();
	ApplyPyramids();
	ComposeOutImage();
}

SuperHDRProcessor* SuperHDRProcessor::GetInstance() {

	if (!sd) {
		sd = 1;
		return new SuperHDRProcessor();
	}
	return NULL;
}

void SuperHDRProcessor::PrepareMaterials(void) {

	if (m_visual.channels() != 3) {
		cvtColor(m_visual, m_visual, CV_GRAY2BGR);
	}

	m_visual.convertTo(m_visual32f, CV_32FC3);
	cvtColor(m_visual, m_hsv, CV_BGR2HSV);
	split(m_hsv, m_splitHSV);
}

void SuperHDRProcessor::ExtractIllumination(void) {

	int dilation_type = MORPH_CROSS;
	int dilation_size = 1;
	Mat element = getStructuringElement(dilation_type, Size(2 * dilation_size + 1, 2 * dilation_size + 1),
		Point(dilation_size, dilation_size));
	m_illumination = new Mat(m_visual.rows, m_visual.cols, CV_8U);
	Mat rgb_values[3];
	split(m_visual, rgb_values);
	max(rgb_values[0], rgb_values[1], *m_illumination);
	max(rgb_values[2], *m_illumination, *m_illumination);

	dilate(*m_illumination, *m_illumination, element);
	erode(*m_illumination, *m_illumination, element);
}

void SuperHDRProcessor::ApplyGuidedFilter(void) {
	int r = 3; // Should try different values
	double eps = (0.2 * 0.2) * (255 * 255); // Should be exported
	m_intensity = new Mat(guidedFilter(*m_illumination, m_splitHSV[2], r, eps));
}

void SuperHDRProcessor::ExtractReflectance(void) {

	Mat *intensity_3ch = new Mat();
	Mat int3ch[3];

	m_intensity->copyTo(int3ch[0]);
	m_intensity->copyTo(int3ch[1]);
	m_intensity->copyTo(int3ch[2]);
	merge(int3ch, 3, *intensity_3ch);

	Mat atom_ones = Mat::ones(m_intensity->rows, m_intensity->cols, CV_8UC3);
	add(atom_ones, *intensity_3ch, *intensity_3ch);
	m_visual.convertTo(m_visual32f, CV_32FC3);
	intensity_3ch->convertTo(m_intensity32f, CV_32FC3);
	divide(m_visual32f, m_intensity32f, m_reflectance32f);

	Mat atom_255 = Mat(m_intensity->rows, m_intensity->cols, CV_32FC3, Scalar(255, 255, 255));
	multiply(atom_255, m_reflectance32f, m_reflectance32f);
	m_reflectance32f.convertTo(m_reflectance, CV_8UC3);

}

void SuperHDRProcessor::ExtractGlobalIntensity(void) {

	m_intensity->convertTo(m_intensity32f, CV_32FC1);
	Scalar m = mean(m_intensity32f);
	m[0] /= 255.0;
	Scalar intensityLambda = Scalar(4 + ((1 - m[0]) / m[0]));
	Mat divider = Mat(m_intensity->rows, m_intensity->cols, CV_32FC1, Scalar(255.0));
	Mat multiplier = Mat(m_intensity->rows, m_intensity->cols, CV_32FC1, intensityLambda[0]);

	divide(m_intensity32f, divider, m_globalIntensity32f);
	multiply(m_globalIntensity32f, multiplier, m_globalIntensity32f);
	calcAtanForMat(&m_globalIntensity32f);
	Mat mul2pi = Mat(m_globalIntensity32f.rows, m_intensity->cols, CV_32FC1, Scalar(2.0 / CV_PI));
	multiply(m_globalIntensity32f, mul2pi, m_globalIntensity32f);
	multiply(m_globalIntensity32f, divider, m_globalIntensity32f);
	m_globalIntensity32f.convertTo(m_globalIntensity, CV_8U);
}

void SuperHDRProcessor::AdaptiveHistogramEqualization(void) {

	int m_claheClip = 4;
	cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
	clahe->setClipLimit(m_claheClip);

	clahe->apply(*m_intensity, m_intensityClahe);
	m_intensityClahe.convertTo(m_intensityClahe32f, CV_32FC1);

}

void SuperHDRProcessor::CombineIntensities(void) {
	Mat intChannels[3];
	m_intensity->copyTo(intChannels[0]);
	m_globalIntensity.copyTo(intChannels[1]);
	m_intensityClahe.copyTo(intChannels[2]);
	merge(intChannels, 3, m_fullIntensity);

}

void SuperHDRProcessor::ApplyBrightnessWeight(const Mat &intensityChannel, Mat *weight) {

	float brightness_mean = 0.5;
	float brightness_stdDev = 2 * std::pow(0.25, 2);
	Mat divider = Mat(m_intensity->rows, m_intensity->cols, CV_32FC1, Scalar(255.0));
	Mat brightnessMeanMat = Mat(m_intensity->rows, m_intensity->cols, CV_32FC1, Scalar(brightness_mean));
	Mat brightnessStdDevMat = Mat(m_intensity->rows, m_intensity->cols, CV_32FC1, Scalar(brightness_stdDev));
	Mat minus_one = Mat(m_intensity->rows, m_intensity->cols, CV_32FC1, Scalar(-1.0));
	divide(intensityChannel, divider, *weight);
	subtract(*weight, brightnessMeanMat, *weight);
	multiply(*weight, *weight, *weight);
	divide(*weight, brightnessStdDevMat, *weight);
	multiply(*weight, minus_one, *weight);
	exp(*weight, *weight);

}

void SuperHDRProcessor::ApplyChromaticContrastWeight(const Mat &intensityChannel, Mat *weight) {

	float alpha = 2.0;
	float phi = 4.36;

	Mat hue;
	Mat saturation;

	m_splitHSV[0].convertTo(hue, CV_32FC1);
	m_splitHSV[1].convertTo(saturation, CV_32FC1);

	Mat divider = Mat(m_intensity->rows, m_intensity->cols, CV_32FC1, Scalar(255.0));
	divide(hue, divider, hue);
	divide(saturation, divider, saturation);

	divide(intensityChannel, divider, *weight);

	Mat alpha_mat = Mat(m_intensity->rows, m_intensity->cols, CV_32FC1, Scalar(alpha));
	Mat phi_mat = Mat(m_intensity->rows, m_intensity->cols, CV_32FC1, Scalar(phi));
	Mat ones_mat = Mat(m_intensity->rows, m_intensity->cols, CV_32FC1, Scalar(1.0));

	multiply(alpha_mat, hue, hue);
	add(hue, phi_mat, hue);
	calcCosForMat(&hue);
	multiply(hue, saturation, saturation);
	add(ones_mat, saturation, saturation);
	multiply(*weight, saturation, *weight);
}

void SuperHDRProcessor::ApplyWeights(void) {

	Mat m_intensityWeights[3];
	Mat m_brightnessWeight32f;
	Mat m_contrastWeight32f;
	Mat m_mulWeight32f;

	ApplyBrightnessWeight(m_intensity32f, &m_intensityWeights[0]);
	ApplyBrightnessWeight(m_globalIntensity32f, &m_intensityWeights[1]);
	ApplyBrightnessWeight(m_intensityClahe32f, &m_intensityWeights[2]);

	merge(m_intensityWeights, 3, m_brightnessWeight32f);

	ApplyChromaticContrastWeight(m_intensity32f, &m_intensityWeights[0]);
	ApplyChromaticContrastWeight(m_globalIntensity32f, &m_intensityWeights[1]);
	ApplyChromaticContrastWeight(m_intensityClahe32f, &m_intensityWeights[2]);

	merge(m_intensityWeights, 3, m_contrastWeight32f);
	multiply(m_brightnessWeight32f, m_contrastWeight32f, m_mulWeight32f);

	Mat sumArray[3];

	split(m_mulWeight32f, sumArray);

	add(sumArray[0], sumArray[1], sumArray[1]);
	add(sumArray[1], sumArray[2], sumArray[2]);
	sumArray[0] = sumArray[2];
	sumArray[1] = sumArray[2];

	Mat sum3ch;
	merge(sumArray, 3, sum3ch);

	Mat normalize;
	divide(m_mulWeight32f, sum3ch, normalize);

	Mat Norm[3];

	split(normalize, Norm);
	split(normalize, WNorm);
	Mat denom = Mat(m_intensity->rows, m_intensity->cols, CV_32FC1, Scalar(255.0));

	divide(m_intensity32f, denom, m_intensity32f);
	divide(m_globalIntensity32f, denom, m_globalIntensity32f);
	divide(m_intensityClahe32f, denom, m_intensityClahe32f);

	multiply(Norm[0], m_intensity32f, Norm[0]);
	multiply(Norm[1], m_globalIntensity32f, Norm[1]);
	multiply(Norm[2], m_intensityClahe32f, Norm[2]);

	Mat m_intensity_fusion;
	add(Norm[0], Norm[1], Norm[1]);
	add(Norm[1], Norm[2], m_intensity_fusion);

	m_intensity_fusion = m_intensity_fusion * 255;
	Mat n;
	m_intensity_fusion.convertTo(n, CV_8UC1);
}

Mat SuperHDRProcessor::PyramidReconstruct(std::vector<cv::Mat>& pyramid, int levels)
{
	for (int i = levels - 2; i > -1; --i)
	{
		int w, h;
		cv::Mat expandedLevel;
		w = pyramid[i].cols;
		h = pyramid[i].rows;
		cv::resize(pyramid[i + 1], expandedLevel, cv::Size(w, h), 0, 0, cv::INTER_CUBIC);
		cv::GaussianBlur(expandedLevel, expandedLevel, m_blurKernel, m_blurSigmaX, 0, cv::BORDER_REPLICATE);
		cv::add(expandedLevel, pyramid[i], pyramid[i]);
	}
	return pyramid[0];
}

void SuperHDRProcessor::ApplyPyramids()
{
	std::vector<cv::Mat> gaussPyramid_A(m_pyrOctaves), gaussPyramid_B(m_pyrOctaves), gaussPyramid_C(m_pyrOctaves);
	std::vector<cv::Mat> lapPyramid_A(m_pyrOctaves), lapPyramid_B(m_pyrOctaves), lapPyramid_C(m_pyrOctaves);

	cv::Mat* intensityImage32f_ = new cv::Mat(m_intensity32f.rows, m_intensity32f.cols,
		CV_32F, m_intensity32f.data, m_intensity32f.step);
	cv::Mat* globalIntensity32f_ = new cv::Mat(m_globalIntensity32f.rows, m_globalIntensity32f.cols,
		CV_32F, m_globalIntensity32f.data, m_globalIntensity32f.step);
	cv::Mat* intensityClahe32f_ = new cv::Mat(m_intensityClahe32f.rows, m_intensityClahe32f.cols,
		CV_32F, m_intensityClahe32f.data, m_intensityClahe32f.step);



	cv::Mat* w_NormA = new cv::Mat(WNorm[0].rows, WNorm[0].cols, CV_32F, WNorm[0].data, WNorm[0].step);
	cv::Mat* w_NormB = new cv::Mat(WNorm[1].rows, WNorm[1].cols, CV_32F, WNorm[1].data, WNorm[1].step);
	cv::Mat* w_NormC = new cv::Mat(WNorm[2].rows, WNorm[2].cols, CV_32F, WNorm[2].data, WNorm[2].step);

	// Laplacian pyramid is created from intensity image, Gauss pyramid is created from normalized weight images
	CreateGaussPyramid(*w_NormA, gaussPyramid_A, m_pyrOctaves);
	CreateGaussPyramid(*w_NormB, gaussPyramid_B, m_pyrOctaves);
	CreateGaussPyramid(*w_NormC, gaussPyramid_C, m_pyrOctaves);

	CreateLaplacianPyramid(*intensityImage32f_, lapPyramid_A, m_pyrOctaves);
	CreateLaplacianPyramid(*globalIntensity32f_, lapPyramid_B, m_pyrOctaves);
	CreateLaplacianPyramid(*intensityClahe32f_, lapPyramid_C, m_pyrOctaves);

	std::vector<cv::Mat> generalPyr(m_pyrOctaves);
	for (int i = 0; i < m_pyrOctaves; ++i)
	{
		cv::Mat mulA, mulB, mulC;
		cv::Mat matSumA, matSumB;
		cv::multiply(lapPyramid_A[i], gaussPyramid_A[i], mulA);
		cv::multiply(lapPyramid_B[i], gaussPyramid_B[i], mulB);
		cv::multiply(lapPyramid_C[i], gaussPyramid_C[i], mulC);
		cv::add(mulA, mulB, matSumA);
		cv::add(mulC, matSumA, matSumB);
		generalPyr[i] = matSumB;
	}
	cv::Mat reconstructed = PyramidReconstruct(generalPyr, m_pyrOctaves);
	reconstructed.copyTo(m_iFinal);
}

void SuperHDRProcessor::CreateGaussPyramid(const cv::Mat& img, std::vector<cv::Mat>& pyramid, int levels)
{
	// Initialize
	pyramid.resize(levels);
	pyramid[0] = img;
	// For each pyramid level starting from level 1
	//#pragma omp parallel for
	for (size_t i = 1; i < levels; ++i)
	{
		cv::Mat tmp;
		int w, h;
		// Apply gaussian filter
		cv::GaussianBlur(pyramid[i - 1], tmp, m_blurKernel, m_blurSigmaX, 0, cv::BORDER_REPLICATE);
		// Resize image using bicubic interpolation
		w = (int)ceil(pyramid[i - 1].cols / 2.0f);
		h = (int)ceil(pyramid[i - 1].rows / 2.0f);
		cv::resize(tmp, pyramid[i], cv::Size(w, h), 0, 0, cv::INTER_CUBIC);
	}
}

void SuperHDRProcessor::CreateLaplacianPyramid(const cv::Mat& img, std::vector<cv::Mat>& pyramidLap, int levels)
{
	std::vector<cv::Mat> gaussPyr(levels);
	CreateGaussPyramid(img, gaussPyr, levels);
	//#pragma omp parallel for
	for (size_t i = 1; i < levels; ++i)
	{
		int w, h;
		cv::Mat resized, blurred;
		// Upscale gauss pyramid level
		w = gaussPyr[i - 1].cols;
		h = gaussPyr[i - 1].rows;
		cv::resize(gaussPyr[i], resized, cv::Size(w, h), 0, 0, cv::INTER_CUBIC);
		// Blur upscaled image
		cv::GaussianBlur(resized, blurred, m_blurKernel, m_blurSigmaX, 0, cv::BORDER_REPLICATE);
		// Subtract from original image;
		int m_haloMode = 0;

		if (m_haloMode) {
			cv::absdiff(gaussPyr[i - 1], blurred, pyramidLap[i - 1]);
		}
		else {
			cv::subtract(gaussPyr[i - 1], blurred, pyramidLap[i - 1]);
		}

	}
	pyramidLap[levels - 1] = gaussPyr[levels - 1];
}

void SuperHDRProcessor::ComposeOutImage(void) {

	Mat reflectancePlanes[3];
	Mat full_reflectanceF;

	split(m_reflectance32f, reflectancePlanes);
	multiply(m_iFinal, reflectancePlanes[0], reflectancePlanes[0]);
	multiply(m_iFinal, reflectancePlanes[1], reflectancePlanes[1]);
	multiply(m_iFinal, reflectancePlanes[2], reflectancePlanes[2]);

	merge(reflectancePlanes, 3, full_reflectanceF);
	full_reflectanceF.convertTo(m_iOutput, CV_8UC3);
	cvtColor(m_iOutput, m_iOutput, CV_BGR2RGB);

}

