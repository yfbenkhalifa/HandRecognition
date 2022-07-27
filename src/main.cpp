#include "common.h"
#include "dataset.h"
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

// void laplacian(const Mat &input) {
//     Mat src, src_gray, dst;
//     int kernel_size = 3;
//     int scale = 1;
//     int delta = 0;
//     int ddepth = CV_16S;

//     // Reduce noise by blurring with a Gaussian filter ( kernel size = 3 )
//     GaussianBlur(input, src, Size(3, 3), 0, 0, BORDER_DEFAULT);
//     cvtColor(src, src_gray, COLOR_BGR2GRAY); // Convert the image to grayscale
//     Mat abs_dst;
//     Laplacian(src_gray, dst, ddepth, kernel_size, scale, delta, BORDER_DEFAULT);
//     // converting back to CV_8U
//     convertScaleAbs(dst, abs_dst);

//     imshow("Laplacian", abs_dst);
// }

// int main(int argcc, char **argv) {
//     string imagesPath = "../dataset/rgb/*";

//     vector<std::string> files;
//     files = {"../dataset/rgb/08.jpg"};
//     // glob(imagesPath, files);

//     for (const string &file : files) {
//         cout << file << endl;
//         Mat input = imread(file);
//         Mat preprocessed = input.clone();

//         // imshow("Input", input);

//         // GaussianBlur(input, preprocessed, Size(7, 7), 10);
//         // Preprocess::saturate(input, preprocessed);
//         Mat temp;
//         bilateralFilter(input, preprocessed, -1, 25, 10);
//         // Preprocess::sharpenImage(preprocessed, preprocessed);
//         Preprocess::equalize(preprocessed, preprocessed);
//         // Preprocess::saturate(preprocessed, preprocessed);

//         // imshow("Preprocessed", preprocessed);

//         // laplacian(preprocessed);
//         // waitKey();

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

//             Mat color_mask = Segmentation::GetSkinMask(roi);
//             cvtColor(color_mask, color_mask, COLOR_GRAY2BGR);

//             bitwise_and(roi, color_mask, roi);

//             Mat output = Segmentation::ClusterWithMeanShift(roi, 4, 2);

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