#include "common.h"
#include "evaluation.h"
#include "preprocess.h"
#include "segmentation.h"
#include <dirent.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

// std::vector<int> explode(std::string const &s, char delim) {
//     std::vector<int> result;
//     std::istringstream iss(s);

//     for (std::string token; std::getline(iss, token, delim);) {
//         result.push_back(stoi(std::move(token)));
//     }

//     return result;
// }

// int main(int argcc, char **argv) {
//     string imagesPath = "../dataset/rgb/*";

//     vector<std::string> files;
//     glob(imagesPath, files);

//     for (const string &file : files) {
//         cout << file << endl;
//         Mat input = imread(file);
//         Mat preprocessed = input.clone();

//         // imshow("Input", input);

//         // GaussianBlur(input, preprocessed, Size(7, 7), 10);

//         string det_path = "../dataset/det/";
//         string det_file = file.substr(15, file.length() - 19);
//         det_path.append(det_file).append(".txt");
//         ifstream myfile(det_path);

//         vector<Rect> rectangles;
//         if (myfile.is_open()) {
//             string line;
//             while (getline(myfile, line)) {
//                 vector<int> params = explode(line, '\t');
//                 rectangles.push_back(Rect(params[0], params[1], params[2], params[3]));
//             }
//             myfile.close();
//         }

//         Mat mask(input.size(), CV_8UC1);
//         mask.setTo(0);

//         for (int i = 0; i < rectangles.size(); i++) {
//             Mat roi = preprocessed(rectangles[i]);

//             // bilateralFilter(roi, preprocessed, 7, 20, 150);
//             // Preprocess::saturate(preprocessed, preprocessed);

//             // imshow("Preprocessed", preprocessed);

//             Mat output = Segmentation::ClusterWithMeanShift(roi);

//             Mat mask_roi = mask(rectangles[i]);
//             output.copyTo(mask_roi);
//         }

//         string mask_path = "../dataset/mask/";
//         string mask_file = file.substr(15, file.length() - 19);
//         mask_path.append(mask_file).append(".png");
//         Mat gt_mask = imread(mask_path, IMREAD_GRAYSCALE);

//         cout << "Error: " << evaluateMask(gt_mask, mask) << endl;
//     }
// }

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