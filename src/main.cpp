#include "common.h"
#include "dataset.h"
#include "evaluation.h"
#include "preprocess.h"
#include "segmentation.h"
#include "utils.h"
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

//Change this command to "python ../python/main.py" if in your system the python module is not found by "python"
string pytohnCommand = "python3 ../python/main.py ";
//It is important that this is the same foolder specified in the python module
string activeDir = "../activeDir/";

string saveDir = "../results/handDetection/";


int mainClustering(int argcc, char **argv) {
    string imagesPath = "../dataset/rgb/*";

    vector<std::string> files;
    glob(imagesPath, files);

    vector<double> errors;

    for (const string &file : files) {
        cout << file << endl;
        Mat input = imread(file), preprocessed = input.clone();
        // Preprocess::sharpenImage(input, input);
        // bilateralFilter(input, preprocessed, -1, 10, 10);
        // imshow("Preprocessed", preprocessed);
        // waitKey();

        vector<Rect> gt_rectangles = Utils::getGroundTruthRois(file);
        Mat gt_mask = Utils::getGroundTruthMask(file);

        int rectangles_area = 0;

        for (int i = 0; i < gt_rectangles.size(); i++)
            rectangles_area += gt_rectangles[i].area();

        HandsSegmentation segmentor(preprocessed, gt_rectangles, 4, 5);

        Mat output = segmentor.DrawSegments();

        Utils::saveOutput(file, output);

        Mat mask = segmentor.mask;

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

vector<HandMetadata> detect(Mat src)
{
    imwrite(activeDir + "src.jpg", src);

    string command = pytohnCommand;
    string result = exec(command);
    cout << result << endl;
    vector<int> handROICoords = Dataset::readCSV(activeDir + "test.csv");
    // Perform detection
    vector<HandMetadata> hands;
    for (int i = 0; i < handROICoords.size(); i += 4)
    {
        std::vector<int> vec{&handROICoords[i], &handROICoords[i + 4]};
        HandMetadata temp = Dataset::loadHandMetadata(vec);
        hands.push_back(temp);
    }

    // cout << "Found " + to_string(hands.size()) + " hand" << endl;

    return hands;
}

Mat drawROI(Mat src, vector<HandMetadata> hands)
{
    Mat dst = src.clone();
    int thickness = 2;

    for (auto hand : hands)
    {
        Point p1(hand.PosX, hand.PosY);
        int R = rand() % 255 + 1;
        int G = rand() % 255 + 1;
        int B = rand() % 255 + 1;
        Point p2(hand.PosX + hand.Width, hand.PosY + hand.Height);
        rectangle(dst, p1, p2, Scalar(B, G, R), thickness, LINE_8);
    }

    return dst;
}


/* Hand detection module entry point:
// It takes a Mat object representing the input 
*/  

Mat handDetectionModule(Mat src)
{
    Mat srcOriginal = src.clone();
    Preprocess::equalize(src, src);
    Preprocess::fixGamma(src, src);
    Preprocess::smooth(src, src);
    Preprocess::sharpenImage(src, src);
    //imshow("test", src);
    //waitKey();
    vector<HandMetadata> hands = detect(src);
    Mat dst = drawROI(srcOriginal, hands);

    return dst;
}

int main(int argcc, char **argv) {
   
    return -1;
}
