#pragma once

#include "common.h"

class Preprocess {
  public:
    static Mat segment(Mat, Vec3b, Vec3b, Vec3b);

    /* Image Pre-processing Functions */

    // Returns smoothed image using Gaussian blurring with given Kernel and value for sigma
    static Mat smoothImage(Mat, Mat, double);

    // Returns sharpened image using derivative filter with given kernel
    static void sharpenImage(const Mat &input, Mat &output);

    static void equalize(const Mat &input, Mat &output);

    static void saturate(const Mat &input, Mat &output);

    /* Utility Functions */

    // Returns true if the pixel BGR intensity is within the given thresholds
    static bool isWithin(Vec3b, Vec3b, Vec3b);

    void sharpen(Mat);

    void laplacian(const Mat &input, Mat &output);
};
