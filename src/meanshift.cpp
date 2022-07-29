#include "meanshift.h"
#include "sample.h"
#include <algorithm>
#include <iostream>
#include <list>
#include <math.h>
#include <stdio.h>
#include <thread>

using namespace std;

double gaussian_kernel(double distance, double kernel_bandwidth) {
    return exp(-1.0 / 2.0 * (distance * distance) / (kernel_bandwidth * kernel_bandwidth));
}

Sample MeanShift::shift_point(const Sample &point) {

    Sample shifted_point(point.color.size(), point.location.size(), point.originalLocation);

    int skipped = 0;
    double total_weight = 0;

    int x1 = min(max((int)point.location[0] - spatial_bandwidth, 0), image.cols - 1);
    int x2 = min(max((int)point.location[0] + spatial_bandwidth, 0), image.cols - 1);
    int y1 = min(max((int)point.location[1] - spatial_bandwidth, 0), image.rows - 1);
    int y2 = min(max((int)point.location[1] + spatial_bandwidth, 0), image.rows - 1);
    Rect roi(x1, y1, x2 - x1, y2 - y1);

    if (!(0 <= roi.x && 0 <= roi.width && roi.x + roi.width <= image.cols && 0 <= roi.y && 0 <= roi.height && roi.y + roi.height <= image.rows)) {
        cout << point.originalLocation[0] << "-" << point.originalLocation[1] << endl;
        cout << point.location[0] << "-" << point.location[1] << endl;
        return point;
    }

    Mat imageRoi = image(roi);

    for (int r = 0; r < imageRoi.rows; r++)
        for (int c = 0; c < imageRoi.cols; c++) {
            Sample current = Sample(imageRoi, r, c, x1, y1);
            double colorDistance = point.colorDistanceFrom(current);
            double locationDistance = point.locationDistanceFrom(current);

            double weight = gaussian_kernel(colorDistance, color_bandwidth);

            if (1 / weight == 1 / 0.0)
                continue;

            Sample temp = current * weight;
            shifted_point = shifted_point + temp;

            total_weight += weight;
        };
    if (total_weight != 0)
        shifted_point = shifted_point / total_weight;

    return shifted_point;
}

void MeanShift::meanshiftSinglePoint(const Sample &point) {
    int index = point.originalLocation[0] + point.originalLocation[1] * image.cols;

    Sample prev_point(point);
    Sample point_new = shift_point(point);
    int i = 0;
    for (; i < MAX_ITERATIONS; i++) {
        double color_shift_distance = point_new.colorDistanceFrom(prev_point);
        double location_shift_distance = point_new.locationDistanceFrom(prev_point);

        if (color_shift_distance < MIN_COLOR_SHIFT && location_shift_distance < MIN_LOCATION_SHIFT)
            break;

        prev_point = point_new;
        point_new = shift_point(point_new);
    }

    shifted_points[index] = point_new;
}

void MeanShift::meanshift(const vector<Sample> &_points) {
    vector<thread> threads;
    int current_progress = 10;
    double progress = 0;
    cout << "Progress: ";
    for_each(_points.begin(), _points.end(), [&](auto &&point) {
        bool started = false;
        while (!started) {
            if (threads.size() < MAX_THREADS) {
                threads.push_back(thread(&MeanShift::meanshiftSinglePoint, this, point));
                started = true;

                progress += 100.0 / _points.size();
                if (progress > current_progress) {
                    cout << current_progress << "%"
                         << "\t" << flush;
                    current_progress = (int)progress - ((int)progress % 10) + 10;
                }

            } else {
                threads[0].join();
                threads.erase(threads.begin());
            }
        }
    });
    cout << endl;

    while (threads.size() > 0) {
        threads[0].join();
        threads.erase(threads.begin());
    }
}

vector<Point> MeanShift::GrowRegion(const Sample *shifted_points, const vector<Point> &points, int *mask, const int &clusterIndex) {
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

vector<Cluster> MeanShift::RGClustering(const Sample *shifted_points) {
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

                growingVector = GrowRegion(shifted_points, growingVector, mask, clusterIndex);

                for (int i = 0; i < growingVector.size(); i++) {
                    int newIndex = growingVector[i].x + growingVector[i].y * image.cols;
                    newCluster.mode += shifted_points[newIndex];
                    newCluster.shifted_points.push_back(shifted_points[newIndex]);
                }
            }

            clusters.push_back(newCluster);
        }
    }
    for (int i = 0; i < clusters.size(); i++)
        clusters[i].mode = clusters[i].mode / clusters[i].shifted_points.size();

    return clusters;
}

vector<Cluster> MeanShift::cluster(const std::vector<Sample> &points) {
    shifted_points = new Sample[image.cols * image.rows];
    for (int i = 0; i < image.cols * image.rows; i++) {
        double x = i % image.cols;
        double y = i / image.cols;
        shifted_points[i] = Sample({0, 0, 0}, {x, y});
    }

    meanshift(points);
    return RGClustering(shifted_points);
}