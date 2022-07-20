#include "common.h"
#include "preprocess.h"
#include "segmentation.h"
#include <filesystem>
#include <iostream>
#include <string>

#include "dataset.cpp"
#include <dirent.h>
#include <iostream>
#include <vector>
#include <fstream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

int mainClustering(int argcc, char **argv) {
    string imagesPath = "../dataset/rgb/*";

    vector<std::string> files;
    glob(imagesPath, files);

    for (const string &file : files) {
        cout << file << endl;
        Mat input = imread(file), image;

        GaussianBlur(input, image, Size(7, 7), 10);
        // Preprocess::equalize(image, image);
        imshow("Original", input);
        imshow("Equalized", image);
        waitKey(0);

        // medianBlur(input, image, 7);
        // bilateralFilter(input, image, 25, 150, 150);
        // GaussianBlur(input, image, Size(7, 7), 10);
        Mat output = Segmentation::ClusterWithMeanShift(image);
    }
}

int main(int argcc, char **argv) {
    vector<Mat> images;
    string path = "../../Lab7/Datasets/dolomites";

    DIR *dir;
    struct dirent *diread;
    vector<char *> files;

    for(auto image : dataset){
        image.loadCoordinates();
        image.cutImage();
        image.segmentImage();
        //imshow("hand", image.src);
        //waitKey(0);
        for(auto mask : image.handSrc){
            //cout << image.id << endl;
            //imshow("hand", mask);
            //waitKey(0);
            string saveFileName = "../dataset/processed/" + to_string(image.id) + ".jpg";  
            imwrite(saveFileName, mask);
        }
        closedir(dir);
    } else {
        perror("opendir");
        return EXIT_FAILURE;
    }

    for (auto file : files)
        cout << file << "| ";
    cout << endl;
    return -1;
}