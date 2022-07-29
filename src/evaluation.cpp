#include "dataset.h"
#include "common.h"
//Class to evaluate results of our code
double evaluateBox(vector<int> detected, vector<int> label){
   double score = 0;
   double tot;
   cout << detected.size() << endl;
   cout << label.size() << endl;
   for(int i = 0; i<detected.size(); i++){
        if(i+1 % 2 == 0) tot  = 720;
        else tot = 1280;
        score += 1 - abs(detected.at(i) - label.at(i))/tot;
   }

    score /= detected.size();

    return score;
}

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