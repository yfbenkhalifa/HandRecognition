#include "common.h"
#include "dataset.h"
#include "evaluation.h"
#include "preprocess.h"
#include "segmentation.h"
#include <dirent.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <stdio.h>
#include <string>
#include <vector>

#include <iostream>

#include <Python.h>
using namespace std;
using namespace cv;

string pytohnCommand = "python3 ../python/main.py ";
string activeDir = "../activeDir/";
string saveDir = "../results/handDetection/";
int detectionWidth, detectionHeight;
int detectOffsetX, detectOffsetY;
int maxHands = 4;
double scoreThreshold = 0.95;

std::vector<int> explode(std::string const &s, char delim) {
    std::vector<int> result;
    std::istringstream iss(s);

    for (std::string token; std::getline(iss, token, delim);) {
        result.push_back(stoi(std::move(token)));
    }

    return result;
}

int mainClustering(int argcc, char **argv) {
    string imagesPath = "../dataset/rgb/*";

    vector<std::string> files;
    glob(imagesPath, files);

    vector<double> errors;

    for (const string &file : files) {
        cout << file << endl;
        Mat input = imread(file);
        Mat preprocessed = input.clone();

        // imshow("Input", input);

        // GaussianBlur(input, preprocessed, Size(7, 7), 10);
        // Preprocess::saturate(input, preprocessed);
        Mat temp;
        bilateralFilter(input, preprocessed, -1, 25, 10);
        // Preprocess::sharpenImage(preprocessed, preprocessed);
        // Preprocess::equalize(preprocessed, preprocessed);
        // Preprocess::saturate(preprocessed, preprocessed);

        // imshow("Preprocessed", preprocessed);

        // laplacian(preprocessed);
        // waitKey();

        string det_path = "../dataset/det/";
        string det_file = file.substr(15, file.length() - 19);
        det_path.append(det_file).append(".txt");
        ifstream myfile(det_path);

        vector<Rect> rectangles;
        if (myfile.is_open()) {
            string line;
            while (getline(myfile, line)) {
                vector<int> params = explode(line, '\t');
                if (params.size() < 4)
                    params = explode(line, ' ');
                rectangles.push_back(Rect(params[0], params[1], params[2], params[3]));
            }
            myfile.close();
        }

        Mat mask(input.size(), CV_8UC1);
        mask.setTo(0);

        int rectangles_area = 0;

        for (int i = 0; i < rectangles.size(); i++) {
            Mat roi = preprocessed(rectangles[i]);
            Mat mask_roi = mask(rectangles[i]);

            // bilateralFilter(roi, preprocessed, 7, 20, 150);
            // Preprocess::saturate(preprocessed, preprocessed);

            // imshow("Preprocessed", preprocessed);

            Mat color_mask = Segmentation::GetSkinMask(roi);
            // color_mask.copyTo(mask_roi);
            cvtColor(color_mask, color_mask, COLOR_GRAY2BGR);

            bitwise_and(roi, color_mask, roi);

            Mat output = Segmentation::ClusterWithMeanShift(roi, 4, 6);

            mask_roi |= output;

            rectangles_area += rectangles[i].area(); 

            // imshow("Mask", mask);
            // waitKey();
        }

        string mask_path = "../dataset/mask/";
        string mask_file = file.substr(15, file.length() - 19);
        mask_path.append(mask_file).append(".png");
        Mat gt_mask = imread(mask_path, IMREAD_GRAYSCALE);

        double error = evaluateMask(gt_mask, mask) / rectangles_area;

        errors.push_back(error);

        cout << "Error: " << error << endl;
    }
    double avg_error = 0;
    for (auto &error : errors)
        avg_error += error;
    avg_error /= errors.size();
    cout << "ERRORE MEDIO: " << avg_error << endl;
    return 0;
}

vector<Scalar> colors = {(255, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 0, 255), (0, 255, 255), (150, 200, 50)};

string exec(string command) {
    char buffer[128];
    string result = "";

    // Open pipe to file
    FILE *pipe = popen(command.c_str(), "r");
    if (!pipe) {
        return "popen failed!";
    }

    // read till end of process:
    while (!feof(pipe)) {

        // use buffer to read and add to result
        if (fgets(buffer, 128, pipe) != NULL)
            result += buffer;
    }

    pclose(pipe);
    return result;
}

vector<HandMetadata> detect(Mat src) {
    // Takes as argument the image path and reads it
    // Mat src = imread(srcPath);
    // Select detection parameters
    imwrite(activeDir + "src.jpg", src);

    string command = pytohnCommand;
    string result = exec(command);
    cout << result << endl;
    vector<int> handROICoords = readCSV(activeDir + "test.csv");
    // Perform detection
    vector<HandMetadata> hands;
    for (int i = 0; i < handROICoords.size(); i += 4) {
        vector<int> vec{&handROICoords[i], &handROICoords[i + 4]};
        HandMetadata temp = loadHandMetadata(vec);
        hands.push_back(temp);
    }

    // cout << "Found " + to_string(hands.size()) + " hand" << endl;

    return hands;
}

Mat drawROI(Mat src, vector<HandMetadata> hands) {
    Mat dst = src.clone();
    int colorId = 0;
    int thickness = 2;
    for (auto hand : hands) {
        Point p1(hand.PosX, hand.PosY);
        Point p2(hand.PosX + hand.Width, hand.PosY + hand.Height);
        rectangle(dst, p1, p2, colors.at(colorId), thickness, LINE_8);
        colorId += 1;
        if (colorId == colors.size())
            colorId = 0;
    }

    return dst;
}

Mat handDetectionModule(Mat src) {
    // PreprocessImage(src);

    vector<HandMetadata> hands = detect(src);
    Mat dst = drawROI(src, hands);
    // imshow("result", dst);
    // waitKey(0);
    return dst;
}

int main(int argcc, char **argv) {
    // vector<Mat> src = readSrcImages("../dataset/benchmark/rgb/", 30);
    int count = 0;
    string image = "10.jpg";
    Mat src = imread("../dataset/rgb/" + image, IMREAD_ANYCOLOR);
    cout << "Processing Image" << endl;
    Mat dst = handDetectionModule(src);
    cout << "Saving Image" << endl;
    imwrite(saveDir + image, dst);
    std::remove("../activeDir/test.csv");
    std::remove("../activeDir/test.jpg");
    std::remove("../activeDir/src.jpg");

    // createDataset(500);
    return -1;
}
