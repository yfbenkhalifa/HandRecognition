#include "MeanShift.h"
#include <algorithm>
#include <execution>
#include <math.h>
#include <stdio.h>

using namespace std;

#define CLUSTER_EPSILON 50
#define WINDOW_SIZE 10

double gaussian_kernel(double distance, double kernel_bandwidth) {
    double temp = exp(-1.0 / 2.0 * (distance * distance) / (kernel_bandwidth * kernel_bandwidth));
    return temp;
}

void MeanShift::set_kernel(double (*_kernel_func)(double, double)) {
    if (!_kernel_func)
        kernel_func = gaussian_kernel;
    else
        kernel_func = _kernel_func;
}

Sample MeanShift::shift_point(const Sample &point) {
    Sample shifted_point(point.color.size(), point.location.size());

    int skipped = 0;
    double total_weight = 0;

    for (int i = 0; i < density_samples.size(); i++) {
        double distance = point.distanceFrom(density_samples[i]);

        if (distance > WINDOW_SIZE) {
            skipped++;
            continue;
        }

        double weight = 1; // kernel_func(distance, kernel_bandwidth);

        Sample temp = density_samples[i] * weight;
        shifted_point = shifted_point + temp;

        total_weight += weight;
    };

    // printf("Skipped ratio: %f\n", 100.0 * skipped / points.size());

    shifted_point = shifted_point / total_weight;

    return shifted_point;
}

std::vector<Sample> MeanShift::meanshift(const std::vector<Sample> &_points, double _kernel_bandwidth, double _epsilon) {
    kernel_bandwidth = _kernel_bandwidth;
    EPSILON = _epsilon;
    return meanshift(_points);
}

mutex max_distance_lock;

// std::vector<MeanShift::Sample> MeanShift::meanshift(const std::vector<Sample> &_points) {
//     double max_shift_distance;
//     vector<bool> stop_moving(_points.size(), false);
//     vector<Sample> shifted_points = _points;
//     Sample new_point;

//     vector<int> indeces(_points.size());
//     iota(indeces.begin(), indeces.end(), 1);

//     do {
//         max_shift_distance = 0;
//         for_each(indeces.begin(), indeces.end(), [&](auto &&i) {
//             if (stop_moving[i])
//                 return;

//             new_point = shift_point(shifted_points[i]);
//             double shift_distance_sqr = euclidean_distance_sqr(new_point, shifted_points[i]);

//             max_distance_lock.lock();
//             if (shift_distance_sqr > max_shift_distance) {
//                 max_shift_distance = shift_distance_sqr;
//             }
//             max_distance_lock.unlock();

//             if (shift_distance_sqr <= EPSILON_SQUARE) {
//                 stop_moving[i] = true;
//             }
//             shifted_points[i] = new_point;
//         });
//         printf("max_shift_distance: %f\n", sqrt(max_shift_distance));
//     } while (max_shift_distance > EPSILON_SQUARE);
//     return shifted_points;
// }

std::vector<Sample> MeanShift::meanshift(const std::vector<Sample> &_points) {
    vector<Sample> shifted_points(_points.size());
    vector<int> indeces(_points.size());
    iota(indeces.begin(), indeces.end(), 1);

    // for_each(execution::par, indeces.begin(), indeces.end(), [&](auto &&i) {
    for (int i = 0; i < _points.size(); i++) {
        Sample prev_point = _points[i];
        Sample point_new = shift_point(_points[i]);
        while (true) {
            double shift_distance = point_new.distanceFrom(prev_point);

            if (shift_distance <= EPSILON)
                break;
            prev_point = point_new;
            point_new = shift_point(point_new);
        }
        shifted_points[i] = point_new;
    }
    // });
    return shifted_points;
}

vector<Cluster> MeanShift::cluster(const std::vector<Sample> &points, const std::vector<Sample> &shifted_points) {
    vector<Cluster> clusters;

    for (int i = 0; i < shifted_points.size(); i++) {
        int c = 0;
        for (; c < clusters.size(); c++) {
            double distance = shifted_points[i].distanceFrom(clusters[c].mode);
            if (distance <= CLUSTER_EPSILON)
                break;
        }

        if (c == clusters.size()) {
            Cluster clus;
            clus.mode = shifted_points[i];
            clusters.push_back(clus);
        }

        clusters[c].original_points.push_back(points[i]);
        clusters[c].shifted_points.push_back(shifted_points[i]);
    }

    return clusters;
}

vector<Cluster> MeanShift::cluster(const std::vector<Sample> &points, double _kernel_bandwidth) {
    kernel_bandwidth = _kernel_bandwidth;
    std::vector<Sample> shifted_points = meanshift(points, kernel_bandwidth);
    return cluster(points, shifted_points);
}