//
// Created by colde on 5/3/2025.
//

#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <opencv2/opencv.hpp>
#include <vector>

cv::Mat myGaussianBlur(const cv::Mat& src, int kernelSize, double sigma);
cv::Mat myThreshold(const cv::Mat& src, int thresh);
cv::Mat myCannyEdgeDetection(const cv::Mat& gray, double lowThreshold, double highThreshold);
std::vector<cv::Rect> detectLetters(const cv::Mat& processed, cv::Mat& output);
cv::Rect groupLettersIntoPlate(const std::vector<cv::Rect>& letterRects, cv::Mat& output);
cv::Rect detectPlateContour(const cv::Mat& processed, cv::Mat& output);
cv::Mat myDilation(const cv::Mat& src, int kernelSize = 3, int iterations = 1);

#endif
