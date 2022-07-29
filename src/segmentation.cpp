#include "segmentation.h"
#include "meanshift.h"
#include "preprocess.h"
#include "sample.h"
#include "utils.h"
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

using namespace cv;
using namespace std;

Mat HandsSegmentation::GetSkinMask() {
    int min_cr = 135, max_cr = 180, min_cb = 85, max_cb = 135;
    Mat ycrcb_range, ycrcb_ms_image;
    cvtColor(image, ycrcb_ms_image, COLOR_BGR2YCrCb);
    inRange(ycrcb_ms_image, Scalar(0, min_cr, min_cb), Scalar(255, max_cr, max_cb), ycrcb_range);

    Mat mask;
    auto element_type = MORPH_ELLIPSE;
    auto element_size = 6;
    Mat element = getStructuringElement(element_type, Size(element_size * 2 + 1, element_size * 2 + 1));
    morphologyEx(ycrcb_range, mask, MORPH_CLOSE, element);

    return mask;
}

vector<Sample> HandsSegmentation::GetSamples(const Mat &from) {
    vector<Sample> samples;

    for (int r = 0; r < from.rows; r += 1)
        for (int c = 0; c < from.cols; c += 1) {
            Sample temp = Sample(from, r, c);
            if (temp.color[0] >= 1)
                samples.push_back(temp);
        }

    return samples;
}

void HandsSegmentation::ShowClusterizedImage(const Mat &image, const vector<Cluster> &clusters) {
    Mat ycrcb_ms_image(image.size(), CV_8UC3), clusterized;
    ycrcb_ms_image.setTo(0);

    for (int c = 0; c < clusters.size(); c++) {
        const Sample *mode = &clusters[c].mode;
        for (int i = 0; i < clusters[c].shifted_points.size(); i++) {

            const Sample *current = &clusters[c].shifted_points[i];
            Point originalLocation(current->originalLocation[0], current->originalLocation[1]);

            ycrcb_ms_image.at<Vec3b>(originalLocation) = Vec3b(mode->color[0], mode->color[1], mode->color[2]);
        }
    }

    cvtColor(ycrcb_ms_image, clusterized, COLOR_YCrCb2BGR);
    imshow("Clusterized", clusterized);
    waitKey();
}

Mat HandsSegmentation::CreateMaskFromCluster(const Cluster &max_cluster, const Size &image_size) {
    Mat output(image_size, CV_8UC1);
    output.setTo(0);

    const Sample *mode = &max_cluster.mode;
    for (int i = 0; i < max_cluster.shifted_points.size(); i++) {

        const Sample *current = &max_cluster.shifted_points[i];
        Point originalLocation(current->originalLocation[0], current->originalLocation[1]);

        output.at<uchar>(originalLocation) = 255;
    }

    auto element_type = MORPH_ELLIPSE;
    auto element_size = 3;
    Mat element = getStructuringElement(element_type, Size(2 * element_size + 1, 2 * element_size + 1), Point(element_size, element_size));
    morphologyEx(output, output, MORPH_OPEN, element);

    element_size = 6;
    element = getStructuringElement(element_type, Size(2 * element_size + 1, 2 * element_size + 1), Point(element_size, element_size));
    morphologyEx(output, output, MORPH_CLOSE, element);

    return output;
}

Cluster HandsSegmentation::SelectLargestCluster(const vector<Cluster> clusters) {

    vector<Cluster> validClusters;

    for (int c = 0; c < clusters.size(); c++) {
        if (clusters[c].mode.color[0] < 1)
            continue;

        validClusters.push_back(clusters[c]);
    }

    int max_size = -1;
    int max_index = -1;

    for (int c = 0; c < validClusters.size(); c++) {
        int current_size = validClusters[c].shifted_points.size();
        if (current_size > max_size) {
            max_index = c;
            max_size = current_size;
        }
    }
    return validClusters[max_index];
}

Mat HandsSegmentation::MSSegment(const Mat &input, const int &spatial_bandwidth, const double &color_bandwidth) {
    vector<Sample> samples = GetSamples(input);

    MeanShift *c = new MeanShift(input, spatial_bandwidth, color_bandwidth);

    vector<Cluster> clusters = c->cluster(samples);

    // ShowClusterizedImage(input, clusters);

    Cluster max_cluster = SelectLargestCluster(clusters);

    if (max_cluster.shifted_points.size() < 0.4 * samples.size()) {
        cout << "Regeneration" << endl;
        return MSSegment(input, spatial_bandwidth, color_bandwidth + 0.5);
    }

    return CreateMaskFromCluster(max_cluster, input.size());
}

Mat HandsSegmentation::DrawSegments() {
    Mat output = image.clone();
    mask = Mat(image.size(), CV_8UC1);
    mask.setTo(0);

    Mat preprocessed, sharp;
    GaussianBlur(image, sharp, Size(3, 3), 4);
    Preprocess::equalize(sharp, sharp);
    Preprocess::sharpenImage(sharp, sharp);
    bilateralFilter(sharp, preprocessed, -1, 10, 10);

    // imshow("Image", image);
    // imshow("Preprocessed", preprocessed);
    // waitKey();

    // return output;

    if (use_skin_detector) {
        Mat color_mask = HandsSegmentation::GetSkinMask();
        cvtColor(color_mask, color_mask, COLOR_GRAY2BGR);
        bitwise_and(image, color_mask, preprocessed);
    }

    sort(handRois.begin(), handRois.end(), [](Rect a, Rect b) { return a.area() < b.area(); });

    for (int i = 0; i < handRois.size(); i++) {
        Mat prepared_roi = preprocessed(handRois[i]), input;

        cvtColor(prepared_roi, input, COLOR_BGR2YCrCb);
        Mat hand_mask = HandsSegmentation::MSSegment(input, ms_spatial_bandwidth, ms_color_bandwidth);

        Mat mask_roi = mask(handRois[i]);
        bitwise_and(~mask_roi, hand_mask, hand_mask);
        mask_roi |= hand_mask;

        Scalar color = Utils::getColor(i);
        Mat overlay(handRois[i].height, handRois[i].width, CV_8UC3, color), hand_mask_rgb, output_roi = output(handRois[i]);
        cvtColor(hand_mask, hand_mask_rgb, COLOR_GRAY2BGR);

        Mat image_roi = image(handRois[i]);
        addWeighted(image_roi, 0.5, overlay, 0.5, 0, overlay);
        overlay.copyTo(output_roi, hand_mask);
    }

    // imshow("Output", output);
    // waitKey();

    return output;
}