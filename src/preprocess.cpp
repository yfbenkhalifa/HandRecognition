#include "preprocess.h"
#include "common.h"
using namespace std;


bool isWithin(Vec3b vec, Vec3b minThreshold, Vec3b maxThreshold)
{
    for(int i=0; i < 3; i++){
        if (vec[i] >= minThreshold[i] || vec[i] <= maxThreshold[i])
            return false;
    }
    return true;
}



Mat segment(Mat img, Vec3b minThreshold, Vec3b maxThreshold, Vec3b avg)
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
