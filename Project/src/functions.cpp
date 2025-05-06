#include "functions.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;

const double PI = 3.14159265358979323846;


Mat myGaussianBlur(const Mat& src, int kernelSize, double sigma) {
    CV_Assert(src.type() == CV_8UC1);

    Mat dst = Mat::zeros(src.size(), CV_8UC1);
    vector<vector<double>> kernel(kernelSize, vector<double>(kernelSize, 0));
    int k = kernelSize / 2;
    double sum = 0.0;

    for (int i = -k; i <= k; i++) {
        for (int j = -k; j <= k; j++) {
            double value = exp(-(i * i + j * j) / (2 * sigma * sigma)) / (2 * PI * sigma * sigma);
            kernel[i + k][j + k] = value;
            sum += value;
        }
    }

    for (int i = 0; i < kernelSize; i++)
        for (int j = 0; j < kernelSize; j++)
            kernel[i][j] /= sum;

    for (int y = k; y < src.rows - k; y++) {
        for (int x = k; x < src.cols - k; x++) {
            double pixelValue = 0.0;
            for (int i = -k; i <= k; i++) {
                for (int j = -k; j <= k; j++) {
                    pixelValue += src.at<uchar>(y + i, x + j) * kernel[i + k][j + k];
                }
            }
            dst.at<uchar>(y, x) = (uchar)pixelValue;
        }
    }
    return dst;
}



Mat myThreshold(const Mat& src, int thresh) {
    CV_Assert(src.type() == CV_8UC1);
    Mat dst = src.clone();
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            uchar pixel = src.at<uchar>(i, j);
            dst.at<uchar>(i, j) = (pixel > thresh) ? 255 : 0;
        }
    }
    return dst;
}

Mat myCannyEdgeDetection(const Mat& gray, double lowThreshold, double highThreshold) {
    Mat edges;
    Canny(gray, edges, lowThreshold, highThreshold);
    return edges;
}

cv::Mat myDilation(const cv::Mat& src, int kernelSize, int iterations) {
    CV_Assert(src.type() == CV_8UC1);
    CV_Assert(kernelSize % 2 == 1);

    cv::Mat result = src.clone();
    int k = kernelSize / 2;

    for (int iter = 0; iter < iterations; iter++) {
        cv::Mat temp = result.clone();
        for (int y = k; y < src.rows - k; y++) {
            for (int x = k; x < src.cols - k; x++) {
                uchar maxVal = 0;
                for (int i = -k; i <= k; i++) {
                    for (int j = -k; j <= k; j++) {
                        uchar val = temp.at<uchar>(y + i, x + j);
                        if (val > maxVal)
                            maxVal = val;
                    }
                }
                result.at<uchar>(y, x) = maxVal;
            }
        }
    }
    return result;
}

std::vector<cv::Rect> detectLetters(const cv::Mat& processed, cv::Mat& output) {
    std::vector<std::vector<cv::Point>> contours;
    findContours(processed, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    std::vector<cv::Rect> letterCandidates;

    double totalHeight = 0.0;
    for (const auto& contour : contours) {
        cv::Rect rect = boundingRect(contour);
        totalHeight += rect.height;
    }
    double avgHeight = (contours.empty() ? 0 : totalHeight / contours.size());

    int minHeight, maxHeight, minWidth, maxWidth;
    if (avgHeight > 80) {
        minHeight = 20;
        maxHeight = 350;
        minWidth = 10;
        maxWidth = 150;
    } else {
        minHeight = 15;
        maxHeight = 120;
        minWidth = 5;
        maxWidth = 60;
    }

    for (size_t i = 0; i < contours.size(); i++) {
        cv::Rect rect = boundingRect(contours[i]);
        if (rect.height > minHeight && rect.height < maxHeight &&
            rect.width > minWidth && rect.width < maxWidth) {
            letterCandidates.push_back(rect);
            rectangle(output, rect, cv::Scalar(0, 255, 0), 1);
            }
    }
    return letterCandidates;
}


cv::Rect detectPlateContour(const cv::Mat& processed, cv::Mat& output) {
    std::vector<std::vector<cv::Point>> contours;
    findContours(processed, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    cv::Rect bestPlate;
    float bestRatioDiff = 1e9;

    for (const auto& contour : contours) {
        cv::Rect rect = boundingRect(contour);
        if (rect.width < 100 || rect.height < 30)
            continue;

        float ratio = static_cast<float>(rect.width) / rect.height;
        if (ratio > 2.5 && ratio < 5.5) {
            float ratioDiff = std::abs(ratio - 4.0f);
            if (ratioDiff < bestRatioDiff) {
                bestRatioDiff = ratioDiff;
                bestPlate = rect;
            }
        }
    }

    if (bestPlate.area() > 0) {
        rectangle(output, bestPlate, cv::Scalar(0, 255, 255), 2);
    }
    return bestPlate;
}


cv::Rect groupLettersIntoPlate(const std::vector<cv::Rect>& letterRects, cv::Mat& output) {
    if (letterRects.empty()) {
        return cv::Rect();
    }

    const int radius = 150;
    const int minLetters = 5;
    const int maxLetters = 8;
    const float minRatio = 2.5f;
    const float maxRatio = 5.5f;

    std::vector<cv::Rect> bestGroup;
    size_t maxGroupSize = 0;
    cv::Rect bestPlate;

    for (size_t i = 0; i < letterRects.size(); i++) {
        std::vector<cv::Rect> currentGroup;
        cv::Rect base = letterRects[i];

        for (size_t j = 0; j < letterRects.size(); j++) {
            if (i == j) continue;
            cv::Rect other = letterRects[j];
            cv::Point centerBase(base.x + base.width / 2, base.y + base.height / 2);
            cv::Point centerOther(other.x + other.width / 2, other.y + other.height / 2);
            double distance = cv::norm(centerBase - centerOther);
            if (distance < radius) {
                currentGroup.push_back(other);
            }
        }
        currentGroup.push_back(base);

        if (currentGroup.size() >= minLetters && currentGroup.size() <= maxLetters) {
            cv::Rect plateRect = currentGroup[0];
            for (size_t k = 1; k < currentGroup.size(); k++) {
                plateRect |= currentGroup[k];
            }
            float ratio = static_cast<float>(plateRect.width) / plateRect.height;
            if (ratio > minRatio && ratio < maxRatio) {
                if (currentGroup.size() > maxGroupSize) {
                    maxGroupSize = currentGroup.size();
                    bestGroup = currentGroup;
                    bestPlate = plateRect;
                }
            }
        }
    }

    if (!bestGroup.empty()) {
        rectangle(output, bestPlate, cv::Scalar(255, 0, 0), 2);
        return bestPlate;
    }
    return cv::Rect();
}