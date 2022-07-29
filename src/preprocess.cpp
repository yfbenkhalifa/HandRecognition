#include "preprocess.h"
#include "common.h"
using namespace std;

bool Preprocess::isWithin(Vec3b vec, Vec3b minThreshold, Vec3b maxThreshold)
{
    for (int i = 0; i < 3; i++)
    {
        if (vec[i] >= minThreshold[i] || vec[i] <= maxThreshold[i])
            return false;
    }
    return true;
}

Mat Preprocess::segment(Mat img, Vec3b minThreshold, Vec3b maxThreshold, Vec3b avg)
{
    Mat mask(img.rows, img.cols, CV_8U);
    for (int i = 0; i < img.rows; i++)
    {
        for (int j = 0; j < img.cols; j += 3)
        {
            if (isWithin(img.at<Vec3b>(i, j), minThreshold, maxThreshold))
            {
                *(mask.ptr<Vec3b>(i, j)) = Vec3b(255, 255, 255);
            }
            else
            {
                *(mask.ptr<Vec3b>(i, j)) = Vec3b(0, 0, 0);
            }
        }
    }
    return mask;
}

void Preprocess::equalize(const Mat &input, Mat &output)
{

    Mat temp;
    cvtColor(input, temp, COLOR_BGR2HSV);

    Mat *channels = new Mat[3];
    split(temp, channels);

    equalizeHist(channels[2], channels[2]);

    merge(channels, 3, temp);

    cvtColor(temp, output, COLOR_HSV2BGR);
}

void Preprocess::smooth(const Mat &input, Mat &output)
{
    GaussianBlur(input, output, Size(3, 3), 0, 0, BORDER_DEFAULT);
}

void Preprocess::sharpenImage(const Mat &input, Mat &output) {
    Mat temp;
    cvtColor(input, temp, COLOR_BGR2HSV);

    Mat *channels = new Mat[3];
    split(temp, channels);

    double sigma = 6, amount = 4;
    Mat blurry;

    GaussianBlur(channels[1], blurry, Size(), sigma);
    addWeighted(channels[1], 1 + amount, blurry, -amount, 0, channels[1]);

    merge(channels, 3, temp);

    cvtColor(temp, output, COLOR_HSV2BGR);
}

void Preprocess::fixGamma(const Mat &input, Mat &output)
{
    Mat dest(input.rows, input.cols, input.type());
    int x;
    double gamma = 1.0;
    for (int i = 0; i < input.rows; i++)
    {
        for (int j = 0; j < (input.cols * input.channels()); j++)
        {
            output.at<uchar>(i) = saturate_cast<uchar>(pow(input.at<uchar>(j) / 255.0, gamma) * 255.0);
        }
    }
}
