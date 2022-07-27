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

Mat Segmentation::GetSkinMask(const Mat &input) {
    Mat ycb_range, hsv_range, hsv_ms_image, ycr_ms_image;
    cvtColor(input, hsv_ms_image, COLOR_BGR2HSV);
    cvtColor(input, ycr_ms_image, COLOR_BGR2YCrCb);
    int min_h = 0, max_h = 25, min_s = 15, max_s = 190;
    inRange(ycr_ms_image, Scalar(0, 135, 85), Scalar(255, 180, 135), ycb_range);
    inRange(hsv_ms_image, Scalar(min_h, min_s, 0), Scalar(max_h, max_s, 255), hsv_range);

    imshow("HSV_Output", hsv_range);

    createTrackbar("min_h", "HSV_Output", &min_h, 255);
    createTrackbar("max_h", "HSV_Output", &max_h, 255);
    createTrackbar("min_s", "HSV_Output", &min_s, 255);
    createTrackbar("max_s", "HSV_Output", &max_s, 255);

    imshow("Input", input);

    while (true) {
        inRange(hsv_ms_image, Scalar(min_h, min_s, 0), Scalar(max_h, max_s, 255), hsv_range);
        imshow("HSV_Output", hsv_range);
        imshow("YCrCb_Output", ycb_range);
        char key = (char)waitKey(50);
        if (key == 27)
            break;
    }

    Mat mask = ycb_range & hsv_range;
    medianBlur(mask, mask, 5);

    return mask;
}

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

Mat Segmentation::ClusterWithMeanShift(const Mat &input, const int &spatial_bandwidth, const int &color_bandwidth) {

    int resize_scale = 1;

    Mat converted;

    resize(input, input, Size(input.cols / resize_scale, input.rows / resize_scale));
    cvtColor(input, converted, COLOR_BGR2YCrCb);

    vector<Sample> samples;

    for (int r = 0; r < converted.rows; r += 1)
        for (int c = 0; c < converted.cols; c += 1)
            samples.push_back(Sample(converted, r, c));

    MeanShift *c = new MeanShift(converted, spatial_bandwidth, color_bandwidth);
    vector<Cluster> clusters = c->cluster(samples);
    int n_clusters = (int)clusters.size();

    cout << n_clusters << endl;

    Mat ycr_ms_image(converted.size(), CV_8UC3), clusterized;
    ycr_ms_image.setTo(0);

    for (int c = 0; c < n_clusters; c++) {
        // if (clusters[c].shifted_points.size() < max_size * 0.1)
        //     continue;
        Sample *mode = &clusters[c].mode;
        for (int i = 0; i < clusters[c].shifted_points.size(); i++) {

            Sample *current = &clusters[c].shifted_points[i];
            Point originalLocation(current->originalLocation[0], current->originalLocation[1]);

            ycr_ms_image.at<Vec3b>(originalLocation) = Vec3b(mode->color[0], mode->color[1], mode->color[2]);
        }
    }

    cvtColor(ycr_ms_image, clusterized, COLOR_YCrCb2BGR);
    imshow("Clusterized", clusterized);

    Mat output(converted.size(), CV_8UC1);
    output.setTo(0);

    int max_size = -1;
    int max_index = -1;
    for (int c = 0; c < clusters.size(); c++) {
        // Black cluster
        if (clusters[c].mode.color[0] < 1 || clusters[c].shifted_points.size() < 10) {
            clusters.erase(clusters.begin() + c);
            c--;
            continue;
        }
        int current_size = clusters[c].shifted_points.size();
        if (current_size > max_size) {
            max_index = c;
            max_size = current_size;
        }
    }

    cout << clusters.size() << endl;

    vector<Cluster> validClusters;
    validClusters.push_back(clusters[max_index]);
    clusters.erase(clusters.begin() + max_index);

    double ratio = 1.0 * validClusters[0].shifted_points.size() / samples.size();
    while (ratio < 0.6 && clusters.size() > 0) {
        double min_distance = DBL_MAX;
        double min_distance_index = -1;
        for (int c = 0; c < clusters.size(); c++) {
            // for (int i = 0; i < validClusters.size(); i++) {
                double distance = validClusters[0].mode.distanceFrom(clusters[c].mode);
                if (distance < min_distance) {
                    min_distance = distance;
                    min_distance_index = c;
                }
            // }
        }
        validClusters.push_back(clusters[min_distance_index]);
        ratio += 1.0 * clusters[min_distance_index].shifted_points.size() / samples.size();
        clusters.erase(clusters.begin() + min_distance_index);
    }

    for (int c = 0; c < validClusters.size(); c++) {
        Sample *mode = &validClusters[c].mode;
        for (int i = 0; i < validClusters[c].shifted_points.size(); i++) {

            Sample *current = &validClusters[c].shifted_points[i];
            Point originalLocation(current->originalLocation[0], current->originalLocation[1]);

            output.at<uchar>(originalLocation) = 255;
        }
    }

    auto erosion_type = MORPH_ELLIPSE;
    auto erosion_size = 3;
    Mat element = getStructuringElement(erosion_type, Size(2 * erosion_size + 1, 2 * erosion_size + 1), Point(erosion_size, erosion_size));
    // dilate(output, output, element);
    // erode(output, output, element);
    // erode(output, output, element);
    // dilate(output, output, element);

    morphologyEx(output, output, MORPH_CLOSE, element);
    morphologyEx(output, output, MORPH_OPEN, element);
    morphologyEx(output, output, MORPH_OPEN, element);
    morphologyEx(output, output, MORPH_CLOSE, element);

    imshow("Output", output);
    waitKey();

    return output;
}