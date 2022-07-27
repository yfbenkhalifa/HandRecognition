#include "preprocess.h"
#include "common.h"
using namespace std;

bool Preprocess::isWithin(Vec3b vec, Vec3b minThreshold, Vec3b maxThreshold) {
    for (int i = 0; i < 3; i++) {
        if (vec[i] >= minThreshold[i] || vec[i] <= maxThreshold[i])
            return false;
    }
    return true;
}

Mat Preprocess::segment(Mat img, Vec3b minThreshold, Vec3b maxThreshold, Vec3b avg) {
    Mat mask(img.rows, img.cols, CV_8U);
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j += 3) {
            if (isWithin(img.at<Vec3b>(i, j), minThreshold, maxThreshold)) {
                *(mask.ptr<Vec3b>(i, j)) = Vec3b(255, 255, 255);
            } else {
                *(mask.ptr<Vec3b>(i, j)) = Vec3b(0, 0, 0);
            }
        }
    }
    return mask;
}

void Preprocess::equalize(const Mat &input, Mat &output) {

    Mat temp;
    cvtColor(input, temp, COLOR_BGR2YCrCb);

    Mat *channels = new Mat[3];
    split(temp, channels);

    equalizeHist(channels[0], channels[0]);

    merge(channels, 3, temp);

    cvtColor(temp, output, COLOR_YCrCb2BGR);
}

void Preprocess::saturate(const Mat &input, Mat &output) {

    Mat temp;
    cvtColor(input, temp, COLOR_BGR2HSV);

    Mat *channels = new Mat[3];
    split(temp, channels);

    equalizeHist(channels[1], channels[1]);

    merge(channels, 3, temp);

    cvtColor(temp, output, COLOR_HSV2BGR);
}

void Preprocess::sharpenImage(const Mat &input, Mat &output) {
    // cv::blur(input, output, cv::Size(50, 50));
    // cv::addWeighted(input, 4, output, -4, 128, output);
    // Mat sharp;
    // Mat sharpening_kernel = (Mat_<double>(3, 3) << -1, -1, -1, -1, 9, -1, -1, -1, -1);
    // filter2D(input, output, -1, sharpening_kernel);
    double sigma = 4, amount = 1;
    Mat blurry;
    GaussianBlur(input, blurry, Size(), sigma);
    addWeighted(input, 1 + amount, blurry, -amount, 0, output);
}
