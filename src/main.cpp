#pragma once
#include "common.h"
#include "dataset.h"
#include "evaluation.h"
#include "preprocess.h"
#include "segmentation.h"
#include "utils.h"

// Change this command to "python ../python/main.py" if in your system the python module is not found by "python"
string pythonCommand = "python3 ../python/main.py ";
// It is important that this is the same foolder specified in the python module
string activeDir = "../activeDir/";
// Directory for saving the evalutation results
string saveDir = "../results/handDetection/";
string segmentationSaveDir = "../results/handSegmentation/";

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
    imwrite(activeDir + "src.jpg", src);

    string command = pythonCommand;
    string result = exec(command);
    cout << result << endl;
    vector<int> handROICoords = Dataset::readCSV(activeDir + "test.csv");
    // Perform detection
    vector<HandMetadata> hands;
    for (int i = 0; i < handROICoords.size(); i += 4) {
        std::vector<int> vec{&handROICoords[i], &handROICoords[i + 4]};
        HandMetadata temp = Dataset::loadHandMetadata(vec);
        hands.push_back(temp);
    }

    // cout << "Found " + to_string(hands.size()) + " hand" << endl;

    return hands;
}

Mat drawROI(Mat src, vector<HandMetadata> hands) {
    Mat dst = src.clone();
    int thickness = 2;

    for (auto hand : hands) {
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
Mat handDetectionModule(Mat src) {
    Mat srcOriginal = src.clone();
    Preprocess::equalize(src, src);
    Preprocess::fixGamma(src, src);
    Preprocess::smooth(src, src);
    Preprocess::sharpenImage(src, src);
    // imshow("test", src);
    // waitKey();
    vector<HandMetadata> hands = detect(src);
    Mat dst = drawROI(srcOriginal, hands);

    // Clean up temporary files
    // std::remove("../activeDir/test.csv");
    // std::remove("../activeDir/test.jpg");
    // std::remove("../activeDir/src.jpg");

    return dst;
}

int main(int argcc, char **argv) {
    string imagesPath = "../dataset/rgb/*";

    vector<std::string> files;
    glob(imagesPath, files);

    vector<double> errors;

    std::ofstream error_file;
    error_file.open(segmentationSaveDir + "results.txt");

    for (const string &file : files) {
        cout << file << endl;
        Mat input = imread(file);

        error_file << file << endl;

        vector<Rect> gt_rectangles = Utils::getGroundTruthRois(file);

        Mat gt_mask = Utils::getGroundTruthMask(file);

        Mat temp(input.size(), CV_8UC1, Scalar(0));

        for (int i = 0; i < gt_rectangles.size(); i++)
            temp(gt_rectangles[i]).setTo(Scalar(255));

        int rectangles_area = countNonZero(temp);

        vector<Rect> detected_rectangles;
        auto hands = detect(input);
        for (int i = 0; i < hands.size(); i++) {
            detected_rectangles.push_back(hands[i].getBoundingBox());
        }

        HandsSegmentation segmentor(input, detected_rectangles, 4, 6);

        Mat output = segmentor.DrawSegments();

        Utils::saveOutput(file, segmentationSaveDir, output);

        Mat mask = segmentor.mask;

        double error = evaluateMask(gt_mask, mask) / rectangles_area;

        errors.push_back(error);

        error_file << "Error: " << error << endl << endl;
    }
    double avg_error = 0;
    for (auto &error : errors)
        avg_error += error;
    avg_error /= errors.size();
    error_file << "AVG ERROR: " << avg_error << endl;

    error_file.close();
    return 0;
}
