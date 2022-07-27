#include "segmentation.h"
#include "MeanShift.h"
#include "sample.h"
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

using namespace cv;
using namespace std;

Mat Segmentation::SegmentByColor(const Mat &input) {
    Mat converted;
    cvtColor(input, converted, COLOR_BGR2YCrCb);

    std::vector<cv::Mat> converted_channels;

    split(converted, converted_channels);

    threshold(converted_channels[1], converted_channels[1], 131, 255, THRESH_TOZERO);
    imshow("Intersection1", converted_channels[1]);
    waitKey(0);
    threshold(converted_channels[1], converted_channels[1], 185, 255, THRESH_TOZERO_INV);

    threshold(converted_channels[1], converted_channels[1], 0, 255, THRESH_BINARY);

    threshold(converted_channels[2], converted_channels[2], 80, 255, THRESH_TOZERO);
    threshold(converted_channels[2], converted_channels[2], 135, 255, THRESH_TOZERO_INV);

    threshold(converted_channels[2], converted_channels[2], 0, 255, THRESH_BINARY);

    imshow("Intersection2", converted_channels[2]);

    Mat intersection;

    bitwise_and(converted_channels[1], converted_channels[2], intersection);

    imshow("Intersection", intersection);
    waitKey(0);

    return intersection;
}

Mat Segmentation::ClusterWithMeanShift(const Mat &input) {

    int resize_scale = 1;

    Mat converted;

    resize(input, input, Size(input.cols / resize_scale, input.rows / resize_scale));
    cvtColor(input, converted, COLOR_BGR2YCrCb);

    vector<Sample> samples;

    for (int r = 0; r < converted.rows; r += 1)
        for (int c = 0; c < converted.cols; c += 1)
            samples.push_back(Sample(converted, r, c));

    MeanShift *c = new MeanShift(converted, 4, 2.5);
    vector<Cluster> clusters = c->cluster(samples);

    int n_clusters = (int)clusters.size();

    cout << n_clusters << endl;

    // Mat temp(converted.size(), CV_8UC3);
    // temp.setTo(0);

    // for (int c = 0; c < n_clusters; c++) {
    //     Sample *mode = &clusters[c].mode;
    //     for (int i = 0; i < clusters[c].shifted_points.size(); i++) {

    //         Sample *current = &clusters[c].shifted_points[i];
    //         Point originalLocation(current->originalLocation[0], current->originalLocation[1]);

    //         temp.at<Vec3b>(originalLocation) = Vec3b(mode->color[0], mode->color[1], mode->color[2]);
    //     }
    // }

    // cvtColor(temp, temp, COLOR_YCrCb2BGR);
    // imshow("Temp", temp);

    Mat output(converted.size(), CV_8UC1);
    output.setTo(0);

    vector<Cluster> validClusters;

    // for (int i = 0; i < clusters.size(); i++) {
    //     Sample mode = clusters[i].mode;
    //     if (mode.color[1] - 133 <= (173 - 133) && mode.color[2] - 77 <= (127 - 77)) {
    //         validClusters.push_back(clusters[i]);
    //     }
    // }

    int max_size = -1;
    int max_index = -1;
    for (int c = 0; c < n_clusters; c++) {
        int current_size = clusters[c].shifted_points.size();
        if (current_size > max_size) {
            max_index = c;
            max_size = current_size;
        }
    }

    validClusters.push_back(clusters[max_index]);

    for (int c = 0; c < validClusters.size(); c++) {
        Sample *mode = &validClusters[c].mode;
        for (int i = 0; i < validClusters[c].shifted_points.size(); i++) {

            Sample *current = &validClusters[c].shifted_points[i];
            Point originalLocation(current->originalLocation[0], current->originalLocation[1]);

            output.at<uchar>(originalLocation) = 255;
        }
    }

    auto erosion_type = MORPH_ELLIPSE;
    auto erosion_size = 7;
    Mat element = getStructuringElement(erosion_type, Size(2 * erosion_size + 1, 2 * erosion_size + 1), Point(erosion_size, erosion_size));
    dilate(output, output, element);
    // erode(output, output, element);
    // erode(output, output, element);
    // dilate(output, output, element);

    return output;
}