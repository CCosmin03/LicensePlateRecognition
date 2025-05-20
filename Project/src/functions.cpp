#include "functions.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;

Mat myGaussianBlur(const Mat& src, int kernelSize, double sigma) {
    CV_Assert(src.type() == CV_8UC1);

    Mat dst = Mat::zeros(src.size(), CV_8UC1);
    vector<vector<double>> kernel(kernelSize, vector<double>(kernelSize, 0));
    int k = kernelSize / 2;
    double sum = 0.0;

    for (int i = -k; i <= k; i++) {
        for (int j = -k; j <= k; j++) {
            double value = exp(-(i * i + j * j) / (2 * sigma * sigma)) / (2 * CV_PI * sigma * sigma);
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

Mat myCannyEdgeDetectionOpenCV(const Mat& gray, double lowThreshold, double highThreshold) {
    Mat edges;
    Canny(gray, edges, lowThreshold, highThreshold);
    return edges;
}

int sobelX[3][3] = {
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1}
};
int sobelY[3][3] = {
    {-1, -2, -1},
    {0, 0, 0},
    {1, 2, 1}
};

Mat myCannyEdgeDetection(const Mat& gray, double lowThreshold, double highThreshold) {
    CV_Assert(gray.type() == CV_8UC1);
    int rows = gray.rows;
    int cols = gray.cols;

    // Gradient Magnitude si Direction
    Mat magnitude = Mat::zeros(rows, cols, CV_32FC1);
    Mat direction = Mat::zeros(rows, cols, CV_32FC1);

    for (int y = 1; y < rows - 1; y++) {
        for (int x = 1; x < cols - 1; x++) {
            float gx = 0, gy = 0;
            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {
                    int pixel = gray.at<uchar>(y + i, x + j);
                    gx += pixel * sobelX[i + 1][j + 1];
                    gy += pixel * sobelY[i + 1][j + 1];
                }
            }
            magnitude.at<float>(y, x) = sqrt(gx * gx + gy * gy);
            direction.at<float>(y, x) = atan2(gy, gx) * 180 / CV_PI;
            if (direction.at<float>(y, x) < 0)
                direction.at<float>(y, x) += 180;
        }
    }

    // Non-Maximum Suppression
    Mat nms = Mat::zeros(rows, cols, CV_32FC1);
    for (int y = 1; y < rows - 1; y++) {
        for (int x = 1; x < cols - 1; x++) {
            float angle = direction.at<float>(y, x);
            float mag = magnitude.at<float>(y, x);
            float q = 0, r = 0;

            if ((angle >= 0 && angle < 22.5) || (angle >= 157.5 && angle <= 180)) {
                q = magnitude.at<float>(y, x + 1);
                r = magnitude.at<float>(y, x - 1);
            } else if (angle >= 22.5 && angle < 67.5) {
                q = magnitude.at<float>(y + 1, x - 1);
                r = magnitude.at<float>(y - 1, x + 1);
            } else if (angle >= 67.5 && angle < 112.5) {
                q = magnitude.at<float>(y + 1, x);
                r = magnitude.at<float>(y - 1, x);
            } else if (angle >= 112.5 && angle < 157.5) {
                q = magnitude.at<float>(y - 1, x - 1);
                r = magnitude.at<float>(y + 1, x + 1);
            }

            if (mag >= q && mag >= r)
                nms.at<float>(y, x) = mag;
            else
                nms.at<float>(y, x) = 0;
        }
    }

    // Hysteresis Thresholding
    Mat result = Mat::zeros(rows, cols, CV_8UC1);
    for (int y = 1; y < rows - 1; y++) {
        for (int x = 1; x < cols - 1; x++) {
            float val = nms.at<float>(y, x);
            if (val >= highThreshold) {
                result.at<uchar>(y, x) = 255;
            } else if (val >= lowThreshold) {
                for (int i = -1; i <= 1; i++) {
                    for (int j = -1; j <= 1; j++) {
                        if (nms.at<float>(y + i, x + j) >= highThreshold) {
                            result.at<uchar>(y, x) = 255;
                            goto break_loop;
                        }
                    }
                }
                break_loop:;
            }
        }
    }

    return result;
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

    const int radius = 100;
    const int minLetters = 6;
    const int maxLetters = 9;
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