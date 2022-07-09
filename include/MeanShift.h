#pragma once

#include "sample.h"
#include <vector>

struct Cluster {
    Sample mode;
    std::vector<Sample> original_points;
    std::vector<Sample> shifted_points;
};

class MeanShift {
  public:
    MeanShift(const std::vector<Sample> &_density_points) {
        density_samples = _density_points;
        set_kernel(nullptr);
    }
    MeanShift(const std::vector<Sample> &_density_samples, double (*_kernel_func)(double, double)) {
        density_samples = _density_samples;
        set_kernel(kernel_func);
    }

    std::vector<Sample> meanshift(const std::vector<Sample> &points, double _kernel_bandwidth, double _epsilon = 0.00001);
    std::vector<Cluster> cluster(const std::vector<Sample> &, double);

  private:
    double kernel_bandwidth;
    double EPSILON;

    double (*kernel_func)(double, double);
    std::vector<Sample> density_samples;

    void set_kernel(double (*_kernel_func)(double, double));

    Sample shift_point(const Sample &point);
    std::vector<Sample> meanshift(const std::vector<Sample> &_points);
    std::vector<Cluster> cluster(const std::vector<Sample> &_points, const std::vector<Sample> &shifted_points);
};