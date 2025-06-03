#include <opencv2/opencv.hpp>
#include "src/functions.h"

using namespace cv;
using namespace std;

float computeContourScore(const cv::Rect& r, int imgHeight) {
    if (r.area() == 0) return 0;
    float ratio = (float)r.width / r.height;
    float ratioPenalty = exp(-1.25 * abs(ratio - 4.7f));
    float verticalBoost = (float)(r.y + r.height) / imgHeight;
    return ratioPenalty * sqrt((float)r.area()) * verticalBoost;
}

float computeLetterScore(const cv::Rect& r, int n, int imgHeight) {
    if (r.area() == 0 || n == 0) return 0;
    float ratio = (float)r.width / r.height;
    float ratioPenalty = exp(-1.5 * abs(ratio - 4.7f));
    float verticalBoost = (float)(r.y + r.height) / imgHeight;
    return ratioPenalty * sqrt((float)r.area()) * verticalBoost;
}

int main() {
    string basePath = "C:/Users/colde/OneDrive/Desktop/UTCN/An3Sem2/PI/ImageProcessingProject/Project/images/";

    for (int i = 1; i <= 20; i++) {
        string filename = basePath + "car" + to_string(i) + ".jpg";
        Mat image = imread(filename);
        if (image.empty()) {
            cout << "Nu s-a putut incarca: " << filename << endl;
            continue;
        }

        cout << "\nProcesare imagine: " << filename << endl;
        double t0 = getTickCount();

        Mat gray;
        cvtColor(image, gray, COLOR_BGR2GRAY);
        Mat blurred = myGaussianBlur(gray, 5, 1.0);
        Mat output = image.clone();

        vector<int> thresholds = {100, 130};
        cv::Rect bestPlate;
        string bestMethod = "N/A";
        int bestThresh = 0;
        float bestScore = 0;

        for (int t : thresholds) {
            Mat binary = myThreshold(blurred, t);
            Mat cannyEdges = myCannyEdgeDetection(binary, 50, 150);
            Mat dilated = myDilation(cannyEdges, 3, 2);
            // Contur
            Rect r1 = detectPlateContour(dilated, output);
            float ratio1 = (r1.height > 0) ? (float)r1.width / r1.height : 0;
            float score1 = computeContourScore(r1, image.rows);
            cout << "Contur (t=" << t << "): ratio=" << ratio1 << ", scor=" << score1 << endl;

            if (score1 > bestScore) {
                bestScore = score1;
                bestPlate = r1;
                bestMethod = "contur";
                bestThresh = t;
            }

            // Litere
            vector<Rect> letterRects = detectLetters(cannyEdges, output);
            Rect r2 = groupLettersIntoPlate(letterRects, output);
            float ratio2 = (r2.height > 0) ? (float)r2.width / r2.height : 0;
            float score2 = computeLetterScore(r2, (int)letterRects.size(), image.rows);
            cout << "Litere (t=" << t << "): ratio=" << ratio2 << ", scor=" << score2 << endl;

            if (score2 > bestScore) {
                bestScore = score2;
                bestPlate = r2;
                bestMethod = "litere";
                bestThresh = t;
            }
        }

        if (bestPlate.area() > 0) {
            rectangle(output, bestPlate, Scalar(0, 0, 255), 2);
            cout << "Selectat: metoda=" << bestMethod << ", threshold=" << bestThresh << ", scor=" << bestScore << endl;
        } else {
            cout << "Nicio detectare valida gasita." << endl;
        }

        double t1 = getTickCount();
        double elapsedMs = (t1 - t0) / getTickFrequency() * 1000.0;
        cout << "Timp de executie: " << elapsedMs << " ms" << endl;

        imshow("Detected Plate", output);
        waitKey(0);
    }

    return 0;
}
