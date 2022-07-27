#include "MeanShift.h"
#include "sample.h"
#include <algorithm>
#include <chrono>
#include <execution>
#include <iostream>
#include <list>
#include <math.h>
#include <stdio.h>
#include <thread>

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

    int x1 = max((int)point.location[0] - spatial_bandwidth, 0);
    int x2 = min((int)point.location[0] + spatial_bandwidth, image.cols - 1);
    int y1 = max((int)point.location[1] - spatial_bandwidth, 0);
    int y2 = min((int)point.location[1] + spatial_bandwidth, image.rows - 1);
    Rect roi(x1, y1, x2 - x1, y2 - y1);

    Mat imageRoi = image(roi);

    for (int r = 0; r < imageRoi.rows; r++)
        for (int c = 0; c < imageRoi.cols; c++) {
            Sample current = Sample(imageRoi, r, c, x1, y1);
            double colorDistance = point.colorDistanceFrom(current);
            // double locationDistance = point.locationDistanceFrom(current);

            double weight = gaussian_kernel(colorDistance, color_bandwidth); // colorDistance < color_bandwidth ? 1 : 0;

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

Sample MeanShift::meanshiftSinglePoint(const Sample &point) {
    Sample prev_point(point);
    Sample point_new = shift_point(point);
    int i = 0;
    for (; i < MAX_ITERATIONS; i++) {
        double color_shift_distance = point_new.colorDistanceFrom(prev_point);
        double location_shift_distance = point_new.locationDistanceFrom(prev_point);

        if (color_shift_distance < MIN_COLOR_SHIFT || location_shift_distance < MIN_LOCATION_SHIFT)
            break;

        prev_point = point_new;
        point_new = shift_point(point_new);
    }

    return point_new;
}

Sample *MeanShift::meanshift(const vector<Sample> &_points) {

    Sample *shifted_points = new Sample[image.cols * image.rows];
    int current_progress = 10;
    double progress = 0;

    for_each(/*execution::par,*/ _points.begin(), _points.end(), [&](auto &&point) {
        // std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

        Sample point_new = meanshiftSinglePoint(point);
        // point_new += pruneSample;
        // Sample point_new = meanshiftSinglePoint(point);
        int index = point_new.originalLocation[0] + point_new.originalLocation[1] * image.cols;
        shifted_points[index] = point_new;

        shift_lock.lock();
        progress += 100.0 / _points.size();
        if (progress > current_progress) {
            cout << "Progress: " << current_progress << "%" << endl;
            current_progress = (int)progress - ((int)progress % 10) + 10;
        }
        shift_lock.unlock();
        // std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        // std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
        // std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    });
    return shifted_points;
}

vector<Point> MeanShift::grow(const Sample *shifted_points, const vector<Point> &points, int *mask, const int &clusterIndex) {
    vector<Point> new_points;
    for (int i = 0; i < points.size(); i++) {
        int index = points[i].x + points[i].y * image.cols;
        for (int k = 0; k < 8; k++) {
            int nIndex = index + neighbors[k][0] + neighbors[k][1] * image.cols;
            if (nIndex >= 0 && (nIndex < image.rows * image.cols) && (mask[nIndex] < 0)) {
                double color_distance = shifted_points[nIndex].colorDistanceFrom(shifted_points[index]);
                if (color_distance < color_bandwidth) {
                    mask[nIndex] = clusterIndex;
                    Point new_point(points[i]);
                    new_point.x += neighbors[k][0];
                    new_point.y += neighbors[k][1];
                    new_points.push_back(new_point);
                }
            }
        }
    }
    return new_points;
}

vector<Cluster> MeanShift::cluster(const Sample *shifted_points) {
    vector<Cluster> clusters;
    int clusterIndex = 0;
    int mask[image.cols * image.rows];

    memset(mask, -1, image.cols * image.rows * sizeof(int));

    for (int x = 0; x < image.cols; x++) {
        for (int y = 0; y < image.rows; y++) {
            int index = x + y * image.cols;
            if (mask[index] >= 0)
                continue;

            clusterIndex = clusters.size();
            Cluster newCluster;
            newCluster.mode = Sample(shifted_points[index]);
            newCluster.shifted_points.push_back(shifted_points[index]);

            vector<Point> growingVector = {{Point(x, y)}};
            mask[index] = clusterIndex;

            while (growingVector.size() > 0) {

                growingVector = grow(shifted_points, growingVector, mask, clusterIndex);

                for (int i = 0; i < growingVector.size(); i++) {
                    int newIndex = growingVector[i].x + growingVector[i].y * image.cols;
                    newCluster.mode += shifted_points[newIndex];
                    newCluster.shifted_points.push_back(shifted_points[newIndex]);
                }
            }

            clusters.push_back(newCluster);
        }
    }

    vector<Cluster> validClusters;

    for (int i = 0; i < clusters.size(); i++) {
        if (clusters[i].shifted_points.size() > 50) {
            clusters[i].mode = clusters[i].mode / clusters[i].shifted_points.size();
            validClusters.push_back(clusters[i]);
        }
    }

    return validClusters;
}

vector<Cluster> MeanShift::cluster(const std::vector<Sample> &points) {
    Sample *shifted_points = meanshift(points);
    return cluster(shifted_points);
}