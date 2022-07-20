#pragma once

#include "sample.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

using namespace cv;

struct Cluster {
    Sample mode;
    std::vector<Sample> shifted_points;
};

class MeanShift {
  public:
    MeanShift(const Mat &_image) {
        image = _image.clone();
        set_kernel(nullptr);

        window_size = sqrt(image.rows * image.cols) / 6;
        printf("WINDOW SIZE = %i\n", window_size);
    }
    MeanShift(const Mat &_image, double (*_kernel_func)(double, double)) {
        image = _image.clone();
        set_kernel(kernel_func);

        window_size = sqrt(image.rows * image.cols) / 6;
        printf("WINDOW SIZE = %i\n", window_size);
    }

    std::vector<Sample> meanshift(const std::vector<Sample> &points, double _kernel_bandwidth, double _epsilon = 0.00001);
    std::vector<Cluster> cluster(const std::vector<Sample> &, double);

  private:
    double kernel_bandwidth;
    double EPSILON;

    mutex shift_lock;
    int MAX_ITERATIONS = 100;

    int window_size = 1;

    Sample pruneSample = Sample({0, 0, 0}, {20, 20});

    double (*kernel_func)(double, double);
    Mat image;

    void set_kernel(double (*_kernel_func)(double, double));

    Sample shift_point(const Sample &point);
    std::vector<Sample> meanshift(const std::vector<Sample> &_points);
    std::vector<Cluster> cluster(const std::vector<Sample> &_points, const std::vector<Sample> &shifted_points);

    Sample meanshiftSinglePoint(const Sample &point);
};