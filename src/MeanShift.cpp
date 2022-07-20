#include "MeanShift.h"
#include "sample.h"
#include <algorithm>
#include <execution>
#include <list>
#include <math.h>
#include <stdio.h>

using namespace std;

#define CLUSTER_EPSILON 100

double gaussian_kernel(double distance, double kernel_bandwidth) {
    return exp(-1.0 / 2.0 * (distance * distance) / (kernel_bandwidth * kernel_bandwidth));

    // if (distance < kernel_bandwidth)
    //     return 1;
    // return 0;
}

void MeanShift::set_kernel(double (*_kernel_func)(double, double)) {
    if (!_kernel_func)
        kernel_func = gaussian_kernel;
    else
        kernel_func = _kernel_func;
}

Sample MeanShift::shift_point(const Sample &point) {
    Sample shifted_point(point.color.size(), point.location.size(), point.originalLocation);

    int skipped = 0;
    double total_weight = 0;

    int x1 = clamp((int)point.location[0] - window_size / 2, 0, image.cols - 1);
    int x2 = clamp((int)point.location[0] + window_size / 2, 0, image.cols - 1);
    int y1 = clamp((int)point.location[1] - window_size / 2, 0, image.rows - 1);
    int y2 = clamp((int)point.location[1] + window_size / 2, 0, image.rows - 1);
    Rect roi(x1, y1, x2 - x1, y2 - y1);

    Mat imageRoi = image(roi);

    for (int r = 0; r < imageRoi.rows; r += window_size / 5)
        for (int c = 0; c < imageRoi.cols; c += window_size / 5) {
            Sample current = Sample(imageRoi, r, c, x1, y1);
            double colorDistance = point.colorDistanceFrom(current);
            double locationDistance = point.locationDistanceFrom(current);

            double colorWeight = kernel_func(colorDistance, 15), locationWeight = kernel_func(locationDistance, window_size / 5);
            double weight = colorWeight * locationWeight;

            if (weight == 0)
                continue;

            Sample temp = current * weight;
            shifted_point = shifted_point + temp;

            total_weight += weight;
        };
    if (total_weight != 0)
        shifted_point = shifted_point / total_weight;

    return shifted_point;
}

vector<Sample> MeanShift::meanshift(const vector<Sample> &_points, double _kernel_bandwidth, double _epsilon) {
    kernel_bandwidth = _kernel_bandwidth;
    EPSILON = _epsilon;
    return meanshift(_points);
}

Sample MeanShift::meanshiftSinglePoint(const Sample &point) {
    Sample prev_point(point);
    Sample point_new = shift_point(point);
    for (int i = 0; i < MAX_ITERATIONS; i++) {
        double shift_distance = point_new.distanceFrom(prev_point);

        if (shift_distance <= EPSILON)
            break;

        prev_point = point_new;
        point_new = shift_point(point_new);
    }

    return point_new;
}

vector<Sample> MeanShift::meanshift(const vector<Sample> &_points) {

    vector<Sample> shifted_points;
    int current_progress = 10;

    for_each(execution::par, _points.begin(), _points.end(), [&](auto &&point) {
        Sample point_new = meanshiftSinglePoint(point);
        // point_new += pruneSample;
        // Sample point_new = meanshiftSinglePoint(point);

        shift_lock.lock();
        shifted_points.push_back(point_new);
        int progress = (int)(100.0 * shifted_points.size() / _points.size());
        if (progress > current_progress) {
            printf("Progress: %d% \n", current_progress);
            current_progress = progress - (progress % 10) + 10;
        }
        shift_lock.unlock();
    });
    return shifted_points;
}

vector<Cluster> MeanShift::cluster(const std::vector<Sample> &points, const std::vector<Sample> &shifted_points) {
    vector<Cluster> clusters;

    for (int i = 0; i < shifted_points.size(); i++) {
        int c = 0;
        for (; c < clusters.size(); c++) {
            double color_distance = shifted_points[i].colorDistanceFrom(clusters[c].mode);
            double location_distance = shifted_points[i].locationDistanceFrom(clusters[c].mode);

            if (color_distance < 50 && location_distance < window_size)
                break;

            // double distance = shifted_points[i].distanceFrom(clusters[c].mode);

            // if (distance <= CLUSTER_EPSILON)
            //     break;
        }

        if (c == clusters.size()) {
            Cluster clus;
            clus.mode = shifted_points[i];
            clusters.push_back(clus);
        } else {
            clusters[c].mode = (clusters[c].mode * clusters[c].shifted_points.size() + shifted_points[i]) / (clusters[c].shifted_points.size() + 1);
        }
        clusters[c].shifted_points.push_back(shifted_points[i]);
    }

    return clusters;
}

vector<Cluster> MeanShift::cluster(const std::vector<Sample> &points, double _kernel_bandwidth) {
    kernel_bandwidth = _kernel_bandwidth;
    std::vector<Sample> shifted_points = meanshift(points, kernel_bandwidth, 0.1);
    return cluster(points, shifted_points);
}