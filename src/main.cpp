#include "preprocess.h"
#include "segmentation.h"
#include <filesystem>
#include <iostream>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

int main(int argcc, char **argv) {
    string imagesPath = "../dataset/rgb/*";

    vector<std::string> files;
    glob(imagesPath, files);

    for (const string &file : files) {
        cout << file << endl;
        Mat image = imread(file);
        GaussianBlur(image, image, Size(5, 5), 20);
        imshow("Original", image);
        Mat output = Segmentation::ClusterWithMeanShift(image);
    }
    return -1;
}