#include "segmentation.h"
#include "MeanShift.h"
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

    // int min_h = 135, max_h = 180, min_s = 85, max_s = 135;
    // Mat hsv_range, hsv_ms_image ;
    // cvtColor(image, hsv_ms_image, COLOR_BGR2HSV);
    // inRange(hsv_ms_image, Scalar(min_h, min_s, 0), Scalar(max_h, max_s, 255), hsv_range);

    // imshow("Image", image);
    // imshow("HSV_Output", hsv_range);
    // imshow("YCrCb_Output", hsv_range);

    // createTrackbar("min_h", "HSV_Output", &min_h, 255);
    // createTrackbar("max_h", "HSV_Output", &max_h, 255);
    // createTrackbar("min_s", "HSV_Output", &min_s, 255);
    // createTrackbar("max_s", "HSV_Output", &max_s, 255);

    // createTrackbar("min_cr", "YCrCb_Output", &min_cr, 255);
    // createTrackbar("max_cr", "YCrCb_Output", &max_cr, 255);
    // createTrackbar("min_cb", "YCrCb_Output", &min_cb, 255);
    // createTrackbar("max_cb", "YCrCb_Output", &max_cb, 255);

    Mat mask;
    auto erosion_type = MORPH_ELLIPSE;
    auto closing_size = 6;
    Mat element = getStructuringElement(erosion_type, Size(2 * closing_size + 1, 2 * closing_size + 1), Point(closing_size, closing_size));

    // while (true) {
    //     inRange(hsv_ms_image, Scalar(min_h, min_s, 0), Scalar(max_h, max_s, 255), hsv_range);
    //     inRange(ycrcb_ms_image, Scalar(0, min_cr, min_cb), Scalar(255, max_cr, max_cb), ycrcb_range);
    //     imshow("HSV_Output", hsv_range);
    //     imshow("YCrCb_Output", ycrcb_range);
    //     morphologyEx(ycrcb_range, mask, MORPH_CLOSE, element);
    //     imshow("Output", ycrcb_range);
    //     char key = (char)waitKey(50);
    //     if (key == 27)
    //         break;
    // }

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

    auto erosion_type = MORPH_ELLIPSE;
    auto closing_size = 6;
    Mat element = getStructuringElement(erosion_type, Size(2 * closing_size + 1, 2 * closing_size + 1), Point(closing_size, closing_size));

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

    if (max_cluster.shifted_points.size() < 0.5 * samples.size()) {
        cout << "Regeneration" << endl;
        return MSSegment(input, spatial_bandwidth, color_bandwidth + 1);
    }

    return CreateMaskFromCluster(max_cluster, input.size());
}

Mat HandsSegmentation::DrawSegments() {
    Mat color_mask = HandsSegmentation::GetSkinMask();
    cvtColor(color_mask, color_mask, COLOR_GRAY2BGR);

    Mat prepared;
    bitwise_and(image, color_mask, prepared);

    cvtColor(prepared, prepared, COLOR_BGR2YCrCb);

    Mat output = image.clone();
    mask = Mat(image.size(), CV_8UC1);
    mask.setTo(0);

    sort(handRois.begin(), handRois.end(), [](Rect a, Rect b) { return a.area() > b.area(); });

    for (int i = 0; i < handRois.size(); i++) {
        Mat prepared_roi = prepared(handRois[i]);

        Mat hand_mask = HandsSegmentation::MSSegment(prepared_roi, ms_spatial_bandwidth, ms_color_bandwidth);

        Mat mask_roi = mask(handRois[i]);
        bitwise_and(~mask_roi, hand_mask, hand_mask);
        mask_roi |= hand_mask;

        Scalar color = Utils::getColor(i);
        Mat overlay(prepared_roi.rows, prepared_roi.cols, CV_8UC3, color), hand_mask_rgb, output_roi = output(handRois[i]);
        cvtColor(hand_mask, hand_mask_rgb, COLOR_GRAY2BGR);

        Mat image_roi = image(handRois[i]);
        addWeighted(image_roi, 0.5, overlay, 0.5, 0, overlay);
        overlay.copyTo(output_roi, hand_mask);
    }

    // imshow("Output", output);
    // waitKey();

    return output;
}