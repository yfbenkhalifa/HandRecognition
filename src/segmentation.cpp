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

Mat Segmentation::ClusterWithMeanShift(Mat input) {

    int resize_scale = 2;

    resize(input, input, Size(input.cols / resize_scale, input.rows / resize_scale));
    cvtColor(input, input, COLOR_BGR2HSV);

    vector<Sample> samples;

    for (int r = 0; r < input.rows; r += 1)
        for (int c = 0; c < input.cols; c += 1)
            samples.push_back(Sample(input, r, c));

    MeanShift *c = new MeanShift(input, 3, 6);
    vector<Cluster> clusters = c->cluster(samples);

    int n_clusters = (int)clusters.size();

    cout << n_clusters << endl;

    Mat output = input.clone();
    output.setTo(0);

    for (int c = 0; c < n_clusters; c++) {
        // uchar intesity = (uchar)(c * 255.0 / (n_clusters - 1));
        Sample *mode = &clusters[c].mode;
        for (int i = 0; i < clusters[c].shifted_points.size(); i++) {
            // Point p1(location[0], location[1]), p2(originalLocation[0], originalLocation[1]);
            // drawMarker(output, p1, Scalar(255, 0, 0), MARKER_SQUARE);
            // line(output, p1, p2, Scalar(255, 0, 0));
            Sample *current = &clusters[c].shifted_points[i];
            Point originalLocation(current->originalLocation[0], current->originalLocation[1]);
            Point location(mode->location[0], mode->location[1]);
            // drawMarker(output, location, Scalar(255, 0, 0), MARKER_STAR / 2);
            output.at<Vec3b>(originalLocation) = Vec3b(mode->color[0], mode->color[1], mode->color[2]);
        }
    }

    cvtColor(output, output, COLOR_HSV2BGR);
    resize(output, output, Size(input.cols, input.rows));
    imshow("Output", output);
    waitKey(0);

    // imshow("Output", output);

    return output;
}