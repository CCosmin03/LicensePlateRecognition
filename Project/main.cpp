#include <opencv2/opencv.hpp>
#include "src/functions.h"

using namespace cv;
using namespace std;

int main() {

    Mat image = imread("C:/Users/colde/OneDrive/Desktop/UTCN/An3Sem2/PI/ImageProcessingProject/Project/images/car1.jpg");
    if (image.empty()) {
        cout << "Imaginea nu a putut fi incarcata." << endl;
        return -1;
    }

    //Convertim la grayscale
    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);
    //imshow("Original Grayscale", gray);

    //Gaussian Blur
    Mat blurred = myGaussianBlur(gray, 5, 1.0);
    //imshow("Gaussian Blurred", blurred);

    //Threshold
    Mat binary = myThreshold(blurred, 100);  // pragul poate fi ajustat
    //imshow("Binary Image", binary);

    //Canny (pe imagine binara)
    Mat cannyEdges = myCannyEdgeDetection(binary, 100, 200);
    imshow("Canny Edges", cannyEdges);

    //Dilatare (ajuta la inchiderea contururilor)

    Mat dilated = myDilation(cannyEdges, 3, 2);
    //imshow("Dilated", dilated);

    //Output
    Mat output = image.clone();

    //Detectam mai intai contur clar al placutei (daca este)
    Rect plateByContour = detectPlateContour(dilated, output);

    //Daca nu gasim contur clar, facem detectare pe litere
    if (plateByContour.area() == 0) {
        vector<Rect> letterRects = detectLetters(cannyEdges, output);  // tot pe cannyEdges pentru litere
        Rect plateByLetters = groupLettersIntoPlate(letterRects, output);
    }

    //Afisam rezultatul final
    imshow("Detected Plate", output);

    waitKey(0);
    return 0;
}
