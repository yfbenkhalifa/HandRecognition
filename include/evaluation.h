#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>

using namespace cv;

double evaluateMask(const Mat &gt_mask, const Mat &est_mask) {
    cv::Mat error;
    bitwise_xor(gt_mask, est_mask, error);

    // double total_gt = countNonZero(gt_mask);
    double total_error = countNonZero(error);

    // imshow("Mask", est_mask);
    // imshow("GT Mask", gt_mask);
    // imshow("Error", error);
    // waitKey(0);

    return total_error;
}