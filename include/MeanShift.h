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
  private:
    static const int MAX_THREADS = 12;
    static const int MAX_ITERATIONS = 5;
    static const int MIN_COLOR_SHIFT = 0.3;
    static const int MIN_LOCATION_SHIFT = 0.3;

    double color_bandwidth = 1;
    int spatial_bandwidth = 1;

    Sample *shifted_points;

    const int neighbors[8][2] = {{-1, 0}, {-1, -1}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}};
    Mat image;

    Sample shift_point(const Sample &point);
    void meanshiftSinglePoint(const Sample &point);
    void meanshift(const std::vector<Sample> &points);
    
    std::vector<Cluster> cluster(const Sample *shifted_points);

    vector<Point> grow(const Sample *shifted_points, const vector<Point> &points, int *mask, const int &clusterIndex);

  public:
    MeanShift(const Mat &_image, int _spatial_bandwidth = 4, double _color_bandwidth = 4) {
        image = _image.clone();

        spatial_bandwidth = _spatial_bandwidth;
        color_bandwidth = _color_bandwidth;

        printf("Spatial bandwidth = %i\n", spatial_bandwidth);
        printf("Color bandwidth = %f\n", color_bandwidth);
    }

    std::vector<Cluster> cluster(const std::vector<Sample> &_points);
};